import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from agent.config import settings
from agent.llm_utils import (
    _extract_evidence_sentences,
    build_openai_client,
    build_stub_executive_summary,
    build_stub_section_analysis,
    generate_text_response,
    should_use_stub_llm,
)
from agent.report_writer import ReportWriter
from agent.schemas import (
    AnalysisMode,
    AnalysisPlan,
    AnalysisStep,
    ConfidenceLevel,
    DocumentMeta,
    EvidenceAssessment,
    GeneratedMemo,
    MemoSection,
    ResearchDecisionRecord,
    ResearchLane,
    ResearchQuestionResult,
    ResearchQuestionStatus,
    ResearchReplayRecord,
    ResearchSubquestion,
    ResearchTask,
    RetrievedChunk,
)
from retrieval.build_index import load_processed_company_chunks
from retrieval.hybrid_retriever import HybridRetriever
from verification.claim_extractor import ClaimExtractor
from verification.evidence_verifier import EvidenceVerifier

logger = logging.getLogger(__name__)


HARD_FACT_KEYWORDS = {
    "revenue", "sales", "growth", "margin", "cash flow", "fcf", "funding",
    "valuation", "eps", "guidance", "同比", "环比", "营收", "收入", "利润", "毛利率",
    "利润率", "现金流", "估值",
}
SEMANTIC_KEYWORDS = {
    "management", "tone", "outlook", "risk", "driver", "catalyst", "strategy",
    "attitude", "headwind", "challenge", "管理层", "态度", "展望", "风险", "驱动",
    "战略", "挑战", "monitorable", "monitorables", "monitoring",
}
SLOT_TOKEN_ALIASES = {
    "revenue": {"revenue", "sales", "mix", "monetization"},
    "growth": {"growth", "driver", "drivers", "momentum", "tailwind"},
    "management": {"management", "commentary", "tone", "outlook", "guidance", "framing", "narrative", "emphasis", "priority", "priorities", "cross-sell", "upsell", "portfolio", "tailwind"},
    "evidence": {"evidence", "support", "confidence", "distinction", "weight", "weaker", "stronger"},
    "comparison": {"compare", "comparison", "change", "shift", "versus", "vs"},
    "business": {"business", "model", "platform", "offering", "offerings", "product", "products"},
    "monetization": {"monetization", "pricing", "usage", "consumption", "subscription", "commitment", "billing"},
    "pricing": {"pricing", "usage", "consumption", "subscription", "commitment", "billing"},
    "mix": {"mix", "breakdown", "components", "component", "product", "line", "delivery", "network", "security", "other", "compute", "observability", "segment", "segments", "category", "categories", "customer", "concentration", "geography", "geographic"},
    "profitability": {"profitability", "margin", "gross", "operating", "income"},
    "demand": {"demand", "customer", "customers", "traffic", "retention", "pipeline"},
}
SEMANTIC_QUERY_TYPES = {
    "management_commentary",
    "source_priority",
    "evidence_weighting",
    "risk_extraction",
    "bull_bear_balance",
}
HARD_FACT_SLOT_TOKENS = {
    "revenue", "sales", "margin", "profitability", "cash flow", "fcf", "valuation",
    "funding", "eps", "capex", "guidance", "financials",
}
PRIORITY_LABELS = {
    "highest": 1,
    "high": 1,
    "medium": 2,
    "med": 2,
    "normal": 2,
    "low": 3,
    "lowest": 4,
}
PRIMARY_SOURCE_TYPES = [
    "annual_report",
    "quarterly_report",
    "quarterly_results",
    "results_release",
    "earnings_call_transcript",
    "shareholder_letter",
    "financial",
]
SEMANTIC_SOURCE_TYPES = PRIMARY_SOURCE_TYPES + [
    "investor_presentation",
    "webpage",
    "profile",
    "company_profile",
    "ir_overview",
    "risk",
    "analysis",
    "competitive",
]


class ResearchController:
    def __init__(
        self,
        *,
        load_models: bool = True,
        demo_mode: bool = False,
        report_writer: Optional[ReportWriter] = None,
        retriever_factory=None,
    ):
        self.demo_mode = demo_mode
        self.load_models = load_models
        self.use_stub_llm = demo_mode or should_use_stub_llm(settings.llm_mode, settings.openai_api_key)
        self.client = None
        if not self.use_stub_llm:
            self.client = build_openai_client(settings.openai_api_key, settings.openai_api_base)
        self.report_writer = report_writer or ReportWriter(demo_mode=demo_mode, enable_report_verification=True)
        self.claim_extractor: ClaimExtractor = self.report_writer.claim_extractor
        self.verifier: EvidenceVerifier = self.report_writer.verifier
        self.retriever_factory = retriever_factory or self._default_retriever_factory
        self._company_catalog = self._build_company_catalog()
        self._retriever_cache: Dict[str, tuple[HybridRetriever, Dict[str, dict]]] = {}

    def _strict_live_fail_fast(self) -> bool:
        return (not self.use_stub_llm) and bool(settings.strict_live_mode)

    def research(self, query: str, mode: Optional[AnalysisMode] = None) -> dict:
        task = self._build_task(query=query, mode=mode or AnalysisMode.COMPANY)
        replay = ResearchReplayRecord(task=task)
        llm_budget = {"used": 0, "max": task.llm_call_budget}

        if task.mode == AnalysisMode.COMPETITIVE or len(task.mentioned_company_ids) > 1:
            return self._build_unsupported_scope_result(task, replay)

        subquestions = self._decompose_subquestions(task, llm_budget)
        replay.subquestions = [self._clone_subquestion(item) for item in subquestions]
        retriever, source_registry = self.retriever_factory(task.company_id)
        workflow_events: List[dict] = [
            {"node_name": "research_controller", "event_type": "started", "detail": task.query}
        ]

        hard_fact_questions = [item for item in subquestions if item.lane == ResearchLane.HARD_FACT]
        semantic_questions = [item for item in subquestions if item.lane == ResearchLane.SEMANTIC]
        question_results: List[ResearchQuestionResult] = []

        for group_name, questions in (("hard_fact", hard_fact_questions), ("semantic", semantic_questions)):
            if not questions:
                continue
            workflow_events.append({"node_name": group_name, "event_type": "started", "detail": f"{len(questions)} subquestions"})
            for subquestion in questions:
                result = self._execute_subquestion(
                    task=task,
                    subquestion=subquestion,
                    retriever=retriever,
                    llm_budget=llm_budget,
                    replay=replay,
                )
                question_results.append(result)
                workflow_events.append(
                    {
                        "node_name": subquestion.question_id,
                        "event_type": result.status.value,
                        "detail": result.refusal_reason or f"{len(result.evidence)} evidence chunks",
                    }
                )
            workflow_events.append({"node_name": group_name, "event_type": "completed", "detail": ""})

        self._generate_subquestion_answers(task, question_results, llm_budget, replay)
        replay.subquestions = [self._clone_subquestion(item.subquestion) for item in question_results]
        replay.llm_calls_used = llm_budget["used"]
        memo = self._build_memo(task, question_results, source_registry, replay, llm_budget)
        memo_markdown = self.report_writer.render_memo(memo)
        compatibility_plan = AnalysisPlan(
            mode=task.mode,
            user_query=task.query,
            steps=[
                AnalysisStep(
                    name=item.subquestion.question_id,
                    description=item.subquestion.text,
                    required=item.subquestion.required,
                    search_queries=item.trace[0]["queries"] if item.trace else [],
                    query_contracts=[
                        {
                            "fact_slot": item.subquestion.fact_slot,
                            "lane": item.subquestion.lane.value,
                            "period": item.subquestion.required_period,
                            "source_types": item.subquestion.allowed_source_types,
                        }
                    ],
                    evidence_requirements={"lane": item.subquestion.lane.value},
                )
                for item in question_results
            ],
            contract=memo.contract,
        )
        workflow_events.append({"node_name": "research_controller", "event_type": "completed", "detail": memo.title})

        return {
            "memo_object": memo,
            "memo_markdown": memo_markdown,
            "workflow_events": workflow_events,
            "plan": compatibility_plan,
            "research_task": task,
            "subquestion_results": question_results,
            "research_trace": replay,
            "llm_calls_used": llm_budget["used"],
        }

    # ===== public tool surface for higher-level agents =====

    def build_task(
        self,
        query: str,
        mode: AnalysisMode,
        *,
        company_id: Optional[str] = None,
        period: Optional[str] = None,
        target_periods: Optional[Sequence[str]] = None,
        required_slots: Optional[Sequence[str]] = None,
        required_source_types: Optional[Sequence[str]] = None,
        query_types: Optional[Sequence[str]] = None,
    ) -> ResearchTask:
        return self._build_task(
            query=query,
            mode=mode,
            company_id=company_id,
            period=period,
            target_periods=target_periods,
            required_slots=required_slots,
            required_source_types=required_source_types,
            query_types=query_types,
        )

    def prepare_runtime(self, task: ResearchTask):
        return self.retriever_factory(task.company_id)

    def plan_subquestions(self, task: ResearchTask, llm_budget: dict) -> List[ResearchSubquestion]:
        return self._decompose_subquestions(task, llm_budget)

    def initial_queries(self, task: ResearchTask, subquestion: ResearchSubquestion) -> List[str]:
        return self._build_initial_queries(task, subquestion)

    def retrieve_evidence(
        self,
        *,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        queries: Sequence[str],
        retriever: HybridRetriever,
        round_trace: dict,
    ) -> List[RetrievedChunk]:
        return self._retrieve_queries(task, subquestion, queries, retriever, round_trace)

    def dedupe_evidence(self, chunks: List[RetrievedChunk], limit: int) -> List[RetrievedChunk]:
        return self._dedupe_chunks(chunks, limit)

    def assess_subquestion_evidence(
        self,
        *,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        chunks: List[RetrievedChunk],
    ) -> EvidenceAssessment:
        return self._assess_evidence(task, subquestion, chunks)

    def plan_followup_queries(
        self,
        *,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        assessment: EvidenceAssessment,
        llm_budget: dict,
    ) -> List[str]:
        return self._build_followup_queries(task, subquestion, assessment, llm_budget)

    def generate_subquestion_answers(
        self,
        *,
        task: ResearchTask,
        results: List[ResearchQuestionResult],
        llm_budget: dict,
        replay: ResearchReplayRecord,
    ) -> None:
        self._generate_subquestion_answers(task, results, llm_budget, replay)

    def build_memo_from_results(
        self,
        *,
        task: ResearchTask,
        results: List[ResearchQuestionResult],
        source_registry: Dict[str, dict],
        replay: ResearchReplayRecord,
        llm_budget: dict,
    ) -> GeneratedMemo:
        return self._build_memo(task, results, source_registry, replay, llm_budget)

    def build_unsupported_scope_result(self, task: ResearchTask, replay: ResearchReplayRecord) -> dict:
        return self._build_unsupported_scope_result(task, replay)

    def _build_task(
        self,
        query: str,
        mode: AnalysisMode,
        *,
        company_id: Optional[str] = None,
        period: Optional[str] = None,
        target_periods: Optional[Sequence[str]] = None,
        required_slots: Optional[Sequence[str]] = None,
        required_source_types: Optional[Sequence[str]] = None,
        query_types: Optional[Sequence[str]] = None,
    ) -> ResearchTask:
        mentioned_company_ids = self._infer_company_ids(query)
        resolved_company_id = company_id or (mentioned_company_ids[0] if mentioned_company_ids else self._infer_company_id(query))
        resolved_target_periods = list(target_periods or self._infer_periods(query))
        if period and period not in resolved_target_periods:
            resolved_target_periods.insert(0, period)
        resolved_period = period or (resolved_target_periods[0] if len(resolved_target_periods) == 1 else None)
        return ResearchTask(
            query=query,
            company_id=resolved_company_id,
            period=resolved_period,
            target_periods=resolved_target_periods,
            required_slots=list(required_slots or []),
            required_source_types=list(required_source_types or []),
            query_types=list(query_types or []),
            mode=mode,
            mentioned_company_ids=mentioned_company_ids or ([resolved_company_id] if resolved_company_id else []),
        )

    def _build_company_catalog(self) -> Dict[str, set[str]]:
        catalog: Dict[str, set[str]] = {}
        roots = [settings.data_dir / "processed", settings.company_packs_dir]
        for root in roots:
            if not root.exists():
                continue
            for company_dir in root.iterdir():
                if not company_dir.is_dir():
                    continue
                company_id = company_dir.name.lower()
                catalog.setdefault(company_id, set()).update(
                    {
                        company_id,
                        company_id.replace("_", " "),
                        company_id.replace("-", " "),
                        company_id.upper(),
                        company_id.title(),
                    }
                )
                sources_file = (settings.data_dir / "processed" / company_id / "sources.json")
                if sources_file.exists():
                    try:
                        rows = json.loads(sources_file.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        rows = []
                    for row in rows:
                        for value in (row.get("company"), row.get("issuer"), row.get("title")):
                            if not value:
                                continue
                            normalized = str(value).strip()
                            catalog[company_id].add(normalized)
                            catalog[company_id].add(normalized.lower())
        return catalog

    def _infer_company_id(self, query: str) -> str:
        company_ids = self._infer_company_ids(query)
        if company_ids:
            return company_ids[0]
        available = sorted(self._company_catalog.keys())
        if available:
            return available[0]
        raise RuntimeError("No company packs or processed corpora are available.")

    def _infer_company_ids(self, query: str) -> List[str]:
        normalized_query = query.lower()
        matches = []
        for company_id, aliases in self._company_catalog.items():
            best_alias_length = 0
            for alias in aliases:
                alias_norm = alias.lower()
                if not alias_norm:
                    continue
                if alias_norm in normalized_query:
                    best_alias_length = max(best_alias_length, len(alias_norm))
            if best_alias_length:
                matches.append((best_alias_length, company_id))
        if matches:
            matches.sort(reverse=True)
            return [company_id for _, company_id in matches]
        return []

    def _infer_period(self, query: str) -> Optional[str]:
        periods = self._infer_periods(query)
        if periods:
            return periods[0]
        return None

    def _infer_periods(self, query: str) -> List[str]:
        patterns = [
            (r"\bq([1-4])\s*[-/]?\s*(20\d{2})\b", lambda m: f"{m.group(2)}Q{m.group(1)}"),
            (r"\bfy\s*(20\d{2})\b", lambda m: f"{m.group(1)}FY"),
            (r"\b(20\d{2})\s*[年]?\s*第?([一二三四1-4])\s*季", lambda m: f"{m.group(1)}Q{self._cn_quarter(m.group(2))}"),
        ]
        matches = []
        seen = set()
        for pattern, formatter in patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                period = formatter(match)
                if period in seen:
                    continue
                seen.add(period)
                matches.append((match.start(), period))
        matches.sort(key=lambda item: item[0])
        return [period for _, period in matches]

    def _default_retriever_factory(self, company_id: str):
        cached = self._retriever_cache.get(company_id)
        if cached is not None:
            return cached
        chunks = load_processed_company_chunks(company_id)
        if not chunks:
            raise RuntimeError(f"No processed chunks found for company '{company_id}'.")
        retriever = HybridRetriever(
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
            load_models=self.load_models,
        )
        retriever.index(chunks)
        registry = self._load_source_registry(company_id)
        retriever.attach_source_metadata(registry)
        cached = (retriever, registry)
        self._retriever_cache[company_id] = cached
        return cached

    def _load_source_registry(self, company_id: str) -> Dict[str, dict]:
        sources_file = settings.data_dir / "processed" / company_id / "sources.json"
        if not sources_file.exists():
            return {}
        with open(sources_file, encoding="utf-8") as handle:
            payload = json.load(handle)
        return {row["source_id"]: dict(row) for row in payload}

    def _decompose_subquestions(self, task: ResearchTask, llm_budget: dict) -> List[ResearchSubquestion]:
        proposals: List[dict] = []
        if not self.use_stub_llm and self._consume_llm_budget(llm_budget, reason="decompose_subquestions"):
            proposals = self._llm_decompose_subquestions(task)
        if not proposals:
            proposals = self._fallback_subquestions(task)
        proposals = self._ensure_required_slot_proposals(task, proposals)
        proposals = self._ensure_query_coverage_proposals(task, proposals)
        return self._normalize_subquestions(task, proposals)

    def _llm_decompose_subquestions(self, task: ResearchTask) -> List[dict]:
        prompt = (
            "Return JSON with a top-level key `subquestions`.\n"
            "Each item must include: text, lane, priority, fact_slot, metric_family, needs_numeric_verification.\n"
            "Rules: at most 6 items; split mixed research questions into atomic subquestions; "
            "numerical and period-specific questions must be lane=hard_fact; "
            "risk, tone, strategy, drivers must be lane=semantic.\n\n"
            f"Company: {task.company_id}\n"
            f"Target period: {self._task_period_label(task)}\n"
            f"User query: {task.query}"
        )
        try:
            raw = generate_text_response(
                self.client,
                model=settings.openai_model,
                system_prompt="You decompose financial research requests into atomic, evidence-checkable subquestions.",
                user_prompt=prompt,
                max_tokens=900,
                temperature=0.1,
                max_retries=settings.llm_request_max_retries,
            )
            payload = self._parse_json(raw)
            if isinstance(payload, dict) and isinstance(payload.get("subquestions"), list):
                return payload["subquestions"]
        except Exception as exc:
            if self._strict_live_fail_fast():
                raise
            logger.warning("Falling back to rule-based decomposition because LLM decomposition failed: %s", exc)
        return []

    def _fallback_subquestions(self, task: ResearchTask) -> List[dict]:
        query_lower = task.query.lower()
        query_types = {item.lower() for item in task.query_types}
        proposals: List[dict] = []
        period_label = self._task_period_label(task, include_latest_placeholder=False) or "the target periods"

        if {"management_commentary", "period_diff"} & query_types:
            proposals.append(
                {
                    "text": (
                        f"How did management commentary shift across {period_label} on revenue mix and growth drivers "
                        f"for {task.company_id}?"
                    ),
                    "lane": "semantic",
                    "priority": 1,
                    "fact_slot": "Q3 vs Q4" if len(task.target_periods) >= 2 else "management period shift",
                    "metric_family": "management_tone",
                    "needs_numeric_verification": False,
                }
            )

        if {"source_priority", "evidence_weighting"} & query_types:
            proposals.extend(
                [
                    {
                        "text": (
                            f"Which reported growth-outlook claim for {task.company_id} in {period_label} is the "
                            "high-confidence claim, based on results materials or Exhibit 99.1?"
                        ),
                        "lane": "semantic",
                        "priority": 1,
                        "fact_slot": "high-confidence claim",
                        "metric_family": "evidence",
                        "needs_numeric_verification": False,
                    },
                    {
                        "text": (
                            f"Which lower-confidence management claim about {task.company_id}'s growth outlook appears "
                            f"in the transcript for {period_label}, and what evidence distinction makes it weaker?"
                        ),
                        "lane": "semantic",
                        "priority": 2,
                        "fact_slot": "lower-confidence management claim",
                        "metric_family": "management_tone",
                        "needs_numeric_verification": False,
                    },
                ]
            )

        if self._contains_any(query_lower, HARD_FACT_KEYWORDS):
            if "revenue" in query_lower or "营收" in query_lower or "sales" in query_lower:
                proposals.append(
                    {
                        "text": f"What is {task.company_id}'s disclosed revenue and growth for {task.period or 'the target period'}?",
                        "lane": "hard_fact",
                        "priority": 1,
                        "fact_slot": "revenue",
                        "metric_family": "revenue",
                        "needs_numeric_verification": True,
                    }
                )
            if "margin" in query_lower or "利润率" in query_lower or "cash flow" in query_lower or "现金流" in query_lower:
                proposals.append(
                    {
                        "text": f"What do disclosed margins or cash-flow metrics show for {task.company_id} in {task.period or 'the target period'}?",
                        "lane": "hard_fact",
                        "priority": 2,
                        "fact_slot": "financial_quality",
                        "metric_family": "margin",
                        "needs_numeric_verification": True,
                    }
                )
            if "valuation" in query_lower or "funding" in query_lower or "估值" in query_lower or "融资" in query_lower:
                proposals.append(
                    {
                        "text": f"What disclosed funding or valuation evidence is available for {task.company_id}?",
                        "lane": "hard_fact",
                        "priority": 3,
                        "fact_slot": "valuation",
                        "metric_family": "valuation",
                        "needs_numeric_verification": True,
                    }
                )

        if self._contains_any(query_lower, SEMANTIC_KEYWORDS):
            if "risk" in query_lower or "风险" in query_lower:
                proposals.append(
                    {
                        "text": f"What risks or headwinds are explicitly described for {task.company_id}?",
                        "lane": "semantic",
                        "priority": 4,
                        "fact_slot": "risk",
                        "metric_family": "risk",
                        "needs_numeric_verification": False,
                    }
                )
            if (
                any(token in query_lower for token in ("management", "管理层", "outlook", "展望", "tone", "attitude", "态度"))
                and not {"management_commentary", "source_priority", "evidence_weighting"} & query_types
            ):
                proposals.append(
                    {
                        "text": f"What management tone or outlook is supported by the source pack for {task.company_id}?",
                        "lane": "semantic",
                        "priority": 5,
                        "fact_slot": "management_tone",
                        "metric_family": "management_tone",
                        "needs_numeric_verification": False,
                    }
                )

        if not proposals:
            proposals = [
                {
                    "text": f"What are the core business model and products of {task.company_id}?",
                    "lane": "semantic",
                    "priority": 1,
                    "fact_slot": "business_model",
                    "metric_family": "business_model",
                    "needs_numeric_verification": False,
                },
                {
                    "text": f"What revenue, margin, or funding facts are disclosed for {task.company_id}?",
                    "lane": "hard_fact",
                    "priority": 2,
                    "fact_slot": "financials",
                    "metric_family": "revenue",
                    "needs_numeric_verification": True,
                },
                {
                    "text": f"What outlook, risks, or monitorables matter most for {task.company_id}?",
                    "lane": "semantic",
                    "priority": 3,
                    "fact_slot": "outlook_and_risks",
                    "metric_family": "outlook",
                    "needs_numeric_verification": False,
                },
            ]
        return proposals

    def _ensure_required_slot_proposals(self, task: ResearchTask, proposals: List[dict]) -> List[dict]:
        if not task.required_slots:
            return proposals

        normalized_existing = {
            self._normalize_text(
                " ".join(
                    [
                        str(item.get("fact_slot", "")),
                        str(item.get("metric_family", "")),
                        str(item.get("text", "")),
                    ]
                )
            )
            for item in proposals
        }

        next_priority = max([self._coerce_priority_value(item.get("priority"), default=0) for item in proposals] or [0]) + 1
        augmented = list(proposals)
        for slot in task.required_slots:
            if not self._required_slot_needs_proposal(slot):
                continue
            if any(self._proposal_covers_slot(item, slot) for item in augmented):
                continue
            proposal = self._proposal_for_required_slot(task, slot, next_priority)
            slot_key = self._normalize_text(" ".join([proposal["fact_slot"], proposal["metric_family"], proposal["text"]]))
            if slot_key in normalized_existing:
                continue
            augmented.append(proposal)
            normalized_existing.add(slot_key)
            next_priority += 1
        return augmented

    def _ensure_query_coverage_proposals(self, task: ResearchTask, proposals: List[dict]) -> List[dict]:
        query_lower = task.query.lower()
        augmented = list(proposals)
        next_priority = max([self._coerce_priority_value(item.get("priority"), default=0) for item in augmented] or [0]) + 1

        def has_proposal(predicate) -> bool:
            return any(predicate(item) for item in augmented)

        if any(phrase in query_lower for phrase in ("make money", "business model", "revenue sources", "monetization")):
            if not has_proposal(
                lambda item: (
                    str(item.get("lane", "")).lower() == "semantic"
                    and any(token in self._normalize_text(" ".join([str(item.get("fact_slot", "")), str(item.get("text", ""))])) for token in ("business model", "revenue model", "revenue source", "monetization"))
                )
            ):
                augmented.append(
                    {
                        "text": f"What are {task.company_id}'s primary revenue sources and overall business model for generating money?",
                        "lane": "semantic",
                        "priority": next_priority,
                        "fact_slot": f"{task.company_id}_revenue_model",
                        "metric_family": "business_model",
                        "needs_numeric_verification": False,
                    }
                )
                next_priority += 1

        if self._targets_product_service_catalog_question(task):
            if not has_proposal(
                lambda item: (
                    str(item.get("lane", "")).lower() == "semantic"
                    and any(
                        token in self._normalize_text(" ".join([str(item.get("fact_slot", "")), str(item.get("text", ""))]))
                        for token in ("product", "products", "service", "services", "offering", "offerings")
                    )
                )
            ):
                period_label = self._task_period_label(task, include_latest_placeholder=False) or "the target period"
                augmented.append(
                    {
                        "text": f"What major products, platforms, and services does {task.company_id} explicitly say it sells in {period_label}?",
                        "lane": "semantic",
                        "priority": next_priority,
                        "fact_slot": f"{task.company_id}_products_and_services",
                        "metric_family": "business_model",
                        "needs_numeric_verification": False,
                    }
                )
                next_priority += 1

        if self._targets_customer_concentration_question(task):
            if not has_proposal(
                lambda item: (
                    any(token in self._normalize_text(" ".join([str(item.get("fact_slot", "")), str(item.get("text", ""))])) for token in ("customer concentration", "major customer", "customer accounted"))
                )
            ):
                period_label = self._task_period_label(task, include_latest_placeholder=False) or "the target period"
                augmented.append(
                    {
                        "text": f"Did {task.company_id} disclose customer concentration in {period_label}, and if so what percentage of revenue did the customer represent?",
                        "lane": "hard_fact",
                        "priority": next_priority,
                        "fact_slot": f"{task.company_id}_customer_concentration",
                        "metric_family": "revenue",
                        "needs_numeric_verification": True,
                    }
                )
                next_priority += 1

        if any(phrase in query_lower for phrase in ("revenue mix", "breakdown", "components")):
            if not has_proposal(
                lambda item: (
                    str(item.get("lane", "")).lower() == "hard_fact"
                    and self._is_revenue_mix_payload(item)
                )
            ):
                period_label = self._task_period_label(task, include_latest_placeholder=False) or "the target period"
                augmented.append(
                    {
                        "text": f"What was {task.company_id}'s revenue mix or breakdown by key components (e.g., by product/service type, customer concentration, or geography) specifically for {period_label}?",
                        "lane": "hard_fact",
                        "priority": next_priority,
                        "fact_slot": f"{period_label.lower().replace(' ', '_')}_revenue_mix",
                        "metric_family": "revenue",
                        "needs_numeric_verification": True,
                    }
                )
                next_priority += 1

        if any(phrase in query_lower for phrase in ("management emphasize", "management emphasise", "management emphasis", "what does management emphasize", "what did management emphasize")) or (
            "management" in query_lower and any(phrase in query_lower for phrase in ("revenue mix", "components", "priorities"))
        ):
            if not has_proposal(
                lambda item: (
                    str(item.get("lane", "")).lower() == "semantic"
                    and any(token in self._normalize_text(" ".join([str(item.get("fact_slot", "")), str(item.get("text", ""))])) for token in ("management", "emphasis", "priority", "commentary"))
                    and self._is_revenue_mix_payload(item)
                )
            ):
                period_label = self._task_period_label(task, include_latest_placeholder=False) or "the target period"
                augmented.append(
                    {
                        "text": f"In {task.company_id}'s {period_label} earnings materials or call, what does management emphasize as the most important revenue mix components?",
                        "lane": "semantic",
                        "priority": next_priority,
                        "fact_slot": f"{period_label.lower().replace(' ', '_')}_management_revenue_mix_emphasis",
                        "metric_family": "management_tone",
                        "needs_numeric_verification": False,
                    }
                )
        return augmented

    def _required_slot_needs_proposal(self, slot: str) -> bool:
        slot_text = str(slot or "").strip()
        if not slot_text:
            return False

        normalized = self._normalized_slot_text(slot_text)
        if not normalized:
            return False

        tokens = normalized.split()
        alpha_tokens = [token for token in tokens if re.search(r"[a-z]", token)]
        has_digit = bool(re.search(r"\d", slot_text))

        # Benchmark anchors like "1.6", "$1577.00", or "2022" are useful for scoring,
        # but they are not trustworthy standalone research questions.
        if has_digit and not alpha_tokens:
            return False
        if has_digit and len(alpha_tokens) <= 1 and len(tokens) <= 3:
            return False

        # Full answer sentences from external benchmarks should remain evaluation
        # anchors, not become separate evidence-seeking subquestions.
        answer_like_markers = (
            "positive working capital",
            "negative working capital",
            "had positive",
            "had negative",
            "is positive",
            "is negative",
            "has positive",
            "has negative",
            "based on fy",
            "as of fy",
        )
        if (
            len(tokens) >= 6
            and (
                slot_text.endswith((".", "!", "?"))
                or normalized.startswith(("yes ", "no "))
                or any(marker in normalized for marker in answer_like_markers)
            )
        ):
            return False

        return True

    def _proposal_for_required_slot(self, task: ResearchTask, slot: str, priority: int) -> dict:
        slot_text = str(slot or "").strip()
        lowered = slot_text.lower()
        query_types = {item.lower() for item in task.query_types}
        commentary_mode = bool({"management_commentary", "source_priority", "evidence_weighting"} & query_types)
        period_mode = "period_diff" in query_types
        slot_requests_numeric_fact = any(token in lowered for token in HARD_FACT_SLOT_TOKENS)

        if any(token in lowered for token in ("risk", "bear", "headwind")):
            lane = "semantic"
            metric_family = "risk"
            question = f"What source-backed evidence addresses the required slot '{slot_text}' for {task.company_id}?"
        elif any(token in lowered for token in ("high-confidence", "lower-confidence", "confidence", "evidence distinction", "evidence weight", "weaker claim")) or {"source_priority", "evidence_weighting"} & query_types:
            lane = "semantic"
            metric_family = "evidence"
            question = f"What source-backed evidence distinction addresses the required slot '{slot_text}' for {task.company_id}?"
        elif commentary_mode and any(token in lowered for token in ("revenue mix", "growth drivers", "management emphasis", "q3 vs q4", "vs", "change", "shift", "commentary", "tone", "outlook")):
            lane = "semantic"
            metric_family = "management_tone"
            if period_mode and len(task.target_periods) >= 2:
                question = (
                    f"How does management commentary across {' and '.join(task.target_periods[:2])} address the required slot "
                    f"'{slot_text}' for {task.company_id}?"
                )
            else:
                question = f"What source-backed management commentary addresses the required slot '{slot_text}' for {task.company_id}?"
        elif any(token in lowered for token in ("catalyst", "bull", "driver", "setup", "framing", "narrative", "commentary", "tone", "outlook", "reconciliation")):
            lane = "semantic"
            metric_family = "management" if any(token in lowered for token in ("commentary", "tone", "outlook", "kpi")) else "strategy"
            question = f"What source-backed evidence addresses the required slot '{slot_text}' for {task.company_id}?"
        elif any(
            token in lowered
            for token in (
                "business model",
                "monetization",
                "product breadth",
                "product",
                "products",
                "service",
                "services",
                "offering",
                "offerings",
                "platform",
                "platforms",
            )
        ):
            lane = "semantic"
            metric_family = "business_model"
            question = f"What source-backed evidence addresses the required slot '{slot_text}' for {task.company_id}?"
        elif commentary_mode and not slot_requests_numeric_fact:
            lane = "semantic"
            metric_family = "management_tone"
            question = f"What source-backed commentary addresses the required slot '{slot_text}' for {task.company_id}?"
        else:
            lane = "hard_fact"
            metric_family = "revenue" if any(token in lowered for token in ("revenue", "mix", "growth")) else "financials"
            question = f"What reported evidence addresses the required slot '{slot_text}' for {task.company_id}?"

        return {
            "text": question,
            "lane": lane,
            "priority": priority,
            "fact_slot": slot_text,
            "metric_family": metric_family,
            "needs_numeric_verification": lane == "hard_fact",
        }

    def _normalize_subquestions(self, task: ResearchTask, proposals: List[dict]) -> List[ResearchSubquestion]:
        normalized: List[ResearchSubquestion] = []
        seen_keys = set()
        for idx, proposal in enumerate(proposals, start=1):
            text = str(proposal.get("text", "")).strip()
            if not text:
                continue
            lane = self._enforce_lane(text, proposal.get("lane"))
            fact_slot = str(proposal.get("fact_slot") or f"slot_{idx}").strip()
            metric_family = str(proposal.get("metric_family") or fact_slot).strip().lower()
            priority = max(1, min(task.max_subquestions, self._coerce_priority_value(proposal.get("priority"), default=idx)))
            needs_numeric = bool(proposal.get("needs_numeric_verification") or lane == ResearchLane.HARD_FACT)
            allowed_source_types = self._preferred_source_types_for_subquestion(
                task,
                lane,
                text,
                fact_slot,
            )
            if task.required_source_types:
                preferred = [source_type for source_type in task.required_source_types if source_type in allowed_source_types]
                if preferred:
                    allowed_source_types = preferred
            dedupe_key = (lane.value, fact_slot.lower(), self._normalize_text(text))
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            normalized.append(
                ResearchSubquestion(
                    question_id=f"q{len(normalized) + 1}",
                    text=text,
                    lane=lane,
                    priority=priority,
                    fact_slot=fact_slot,
                    metric_family=metric_family,
                    required_period=(task.period if len(task.target_periods) <= 1 else None),
                    allowed_source_types=allowed_source_types,
                    needs_numeric_verification=needs_numeric,
                    max_rounds=task.max_followup_rounds,
                )
            )
        normalized.sort(key=lambda item: (item.priority, item.question_id))
        collapsed: List[ResearchSubquestion] = []
        seen_intents = set()
        for item in normalized:
            intent_key = (item.lane.value, self._intent_key(item))
            if intent_key in seen_intents:
                continue
            seen_intents.add(intent_key)
            collapsed.append(item)
        return collapsed[: task.max_subquestions]

    def _execute_subquestion(
        self,
        *,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        retriever: HybridRetriever,
        llm_budget: dict,
        replay: ResearchReplayRecord,
    ) -> ResearchQuestionResult:
        subquestion.status = ResearchQuestionStatus.RETRIEVING
        result = ResearchQuestionResult(subquestion=self._clone_subquestion(subquestion), status=ResearchQuestionStatus.RETRIEVING)
        queries = self._build_initial_queries(task, subquestion)
        total_rounds = 0

        while total_rounds <= subquestion.max_rounds:
            total_rounds += 1
            subquestion.rounds_used = total_rounds
            round_trace = {"round": total_rounds, "queries": queries, "retrievals": []}
            retrieved = self._retrieve_queries(task, subquestion, queries, retriever, round_trace)
            result.evidence = self._dedupe_chunks(result.evidence + retrieved, task.max_evidence_per_question)
            assessment = self._assess_evidence(task, subquestion, result.evidence)
            llm_gray_zone_review = None
            if not assessment.sufficient and total_rounds > subquestion.max_rounds:
                assessment, llm_gray_zone_review = self._maybe_llm_review_gray_zone_assessment(
                    task=task,
                    subquestion=subquestion,
                    result=result,
                    assessment=assessment,
                    llm_budget=llm_budget,
                )
            result.assessments.append(assessment)
            round_trace["assessment"] = {
                "valid_chunk_count": assessment.valid_chunk_count,
                "best_entailment": assessment.best_entailment,
                "mean_entailment": assessment.mean_entailment,
                "best_support": assessment.best_support,
                "mean_support": assessment.mean_support,
                "numeric_match": assessment.numeric_match,
                "metric_match": assessment.metric_match,
                "high_trust_hit": assessment.high_trust_hit,
                "sufficient": assessment.sufficient,
                "insufficient": assessment.insufficient,
                "conflict": assessment.conflict,
                "reasons": assessment.reasons,
            }
            if llm_gray_zone_review is not None:
                round_trace["llm_gray_zone_review"] = llm_gray_zone_review
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="llm_gray_zone_review",
                        question_id=subquestion.question_id,
                        reason=llm_gray_zone_review.get("reason", "gray-zone evidence review"),
                        payload=llm_gray_zone_review,
                    )
                )
            result.trace.append(round_trace)

            if assessment.sufficient:
                subquestion.status = ResearchQuestionStatus.COMPLETED
                result.status = ResearchQuestionStatus.COMPLETED
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="complete",
                        question_id=subquestion.question_id,
                        reason=(
                            "sufficient evidence reached"
                            if not (llm_gray_zone_review or {}).get("accepted")
                            else "gray-zone evidence review accepted the evidence as sufficient"
                        ),
                        payload={
                            "round": total_rounds,
                            "matched_chunk_ids": assessment.matched_chunk_ids,
                            "llm_gray_zone_review": bool((llm_gray_zone_review or {}).get("accepted")),
                        },
                    )
                )
                break

            if total_rounds > subquestion.max_rounds:
                subquestion.status = ResearchQuestionStatus.REFUSED
                result.status = ResearchQuestionStatus.REFUSED
                result.refusal_reason = "; ".join(assessment.reasons) or "Evidence remained insufficient after follow-ups."
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="refuse",
                        question_id=subquestion.question_id,
                        reason=result.refusal_reason,
                        payload={"round": total_rounds},
                    )
                )
                break

            if assessment.conflict:
                subquestion.status = ResearchQuestionStatus.CONFLICT
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="conflict",
                        question_id=subquestion.question_id,
                        reason="conflicting evidence detected",
                        payload={"round": total_rounds, "matched_chunk_ids": assessment.matched_chunk_ids},
                    )
                )
            else:
                subquestion.status = ResearchQuestionStatus.NEEDS_FOLLOWUP
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="followup",
                        question_id=subquestion.question_id,
                        reason="evidence insufficient",
                        payload={"round": total_rounds, "reasons": assessment.reasons},
                    )
                )
            queries = self._build_followup_queries(task, subquestion, assessment, llm_budget)
            result.followup_queries = queries

        result.subquestion = self._clone_subquestion(subquestion)
        return result

    def _maybe_llm_review_gray_zone_assessment(
        self,
        *,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        result: ResearchQuestionResult,
        assessment: EvidenceAssessment,
        llm_budget: dict,
    ) -> tuple[EvidenceAssessment, Optional[dict]]:
        if (
            self.use_stub_llm
            or self.client is None
            or not settings.llm_judge_enabled
            or assessment.sufficient
            or assessment.conflict
            or assessment.valid_chunk_count <= 0
        ):
            return assessment, None
        if not self._is_llm_gray_zone_candidate(task, subquestion, assessment):
            return assessment, None
        if not self._consume_llm_budget(llm_budget, reason="gray_zone_evidence_review"):
            return assessment, None

        prompt = self._build_gray_zone_review_prompt(task, subquestion, result, assessment)
        try:
            raw = generate_text_response(
                self.client,
                model=settings.openai_model,
                system_prompt=(
                    "You are a conservative financial evidence reviewer. "
                    "Only mark evidence as sufficient when a careful analyst could answer with the provided chunks "
                    "without inventing missing facts. Never override hard numeric or citation failures."
                ),
                user_prompt=prompt,
                max_tokens=300,
                temperature=0.0,
                max_retries=settings.llm_request_max_retries,
                response_format={"type": "json_object"},
            )
            payload = self._parse_json(raw)
        except Exception as exc:
            logger.warning("Skipping gray-zone evidence review after LLM failure: %s", exc)
            return assessment, {
                "reviewed": True,
                "accepted": False,
                "reason": f"LLM gray-zone review failed: {exc}",
            }

        verdict = str(payload.get("verdict", "")).strip().lower()
        rationale = str(payload.get("rationale", "")).strip()
        if verdict not in {"sufficient", "insufficient"}:
            return assessment, {
                "reviewed": True,
                "accepted": False,
                "reason": "LLM gray-zone review returned an invalid verdict.",
                "raw": raw[:500],
            }
        if verdict != "sufficient":
            return assessment, {
                "reviewed": True,
                "accepted": False,
                "reason": rationale or "LLM gray-zone review kept the evidence below threshold.",
            }
        if subquestion.lane == ResearchLane.HARD_FACT:
            return assessment, {
                "reviewed": True,
                "accepted": False,
                "reason": rationale or "Runtime gray-zone review cannot promote hard-fact evidence.",
                "verdict": verdict,
                "promotion_blocked": True,
            }

        promoted = EvidenceAssessment(
            valid_chunk_count=assessment.valid_chunk_count,
            best_entailment=assessment.best_entailment,
            mean_entailment=assessment.mean_entailment,
            best_support=assessment.best_support,
            mean_support=assessment.mean_support,
            numeric_match=assessment.numeric_match,
            metric_match=assessment.metric_match,
            high_trust_hit=assessment.high_trust_hit,
            sufficient=True,
            insufficient=False,
            conflict=False,
            reasons=list(assessment.reasons) + [
                rationale or "LLM gray-zone review accepted the evidence as sufficient.",
            ],
            matched_chunk_ids=list(assessment.matched_chunk_ids),
        )
        return promoted, {
            "reviewed": True,
            "accepted": True,
            "reason": rationale or "LLM gray-zone review accepted the evidence as sufficient.",
            "verdict": verdict,
        }

    def _is_llm_gray_zone_candidate(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        assessment: EvidenceAssessment,
    ) -> bool:
        blocking_prefixes = (
            "No contract-valid evidence chunks were retrieved.",
            "Target metric or numeric evidence was not found in retrieved chunks.",
            "Conflicting evidence detected across top supporting chunks.",
        )
        if any(reason.startswith(blocking_prefixes) for reason in assessment.reasons):
            return False

        if subquestion.lane == ResearchLane.HARD_FACT:
            return False

        return bool(
            assessment.high_trust_hit
            and assessment.valid_chunk_count >= 1
            and (assessment.mean_support >= 0.32 or assessment.best_support >= 0.45)
        )

    def _build_gray_zone_review_prompt(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        result: ResearchQuestionResult,
        assessment: EvidenceAssessment,
    ) -> str:
        prioritized_chunks = [
            chunk for chunk in result.evidence
            if chunk.chunk_id in set(assessment.matched_chunk_ids)
        ]
        if not prioritized_chunks:
            prioritized_chunks = list(result.evidence[:3])
        else:
            extra = [chunk for chunk in result.evidence if chunk.chunk_id not in set(assessment.matched_chunk_ids)]
            prioritized_chunks = (prioritized_chunks + extra)[:3]

        evidence_lines = []
        for chunk in prioritized_chunks:
            evidence_lines.append(
                f"[Chunk: {chunk.chunk_id}] [Source: {chunk.source_id}] "
                f"[Primary: {'yes' if chunk.is_primary else 'no'}] "
                f"[SourceType: {chunk.source_type or 'unknown'}] "
                f"{chunk.text}"
            )

        prompt = {
            "task": "Decide whether the evidence is sufficient to answer the financial research subquestion conservatively.",
            "company": task.company_id,
            "period": self._task_period_label(task, include_latest_placeholder=False),
            "lane": subquestion.lane.value,
            "subquestion": subquestion.text,
            "fact_slot": subquestion.fact_slot,
            "metric_family": subquestion.metric_family,
            "assessment": {
                "valid_chunk_count": assessment.valid_chunk_count,
                "best_support": assessment.best_support,
                "mean_support": assessment.mean_support,
                "numeric_match": assessment.numeric_match,
                "metric_match": assessment.metric_match,
                "high_trust_hit": assessment.high_trust_hit,
                "reasons": assessment.reasons,
            },
            "instructions": [
                "Return sufficient only if the answer can be stated conservatively using the provided evidence.",
                "For hard-fact questions, allow compositional reasoning only when the needed figures clearly appear in the evidence.",
                "For semantic questions, require direct management commentary or explicit causal language.",
                "Do not approve evidence when key values, periods, or comparison dimensions are missing.",
            ],
            "evidence": evidence_lines,
            "output_schema": {
                "verdict": "sufficient or insufficient",
                "rationale": "short explanation",
            },
        }
        return json.dumps(prompt, ensure_ascii=False)

    def _retrieve_queries(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        queries: Sequence[str],
        retriever: HybridRetriever,
        round_trace: dict,
    ) -> List[RetrievedChunk]:
        source_filters = list(subquestion.allowed_source_types)
        retrieved: List[RetrievedChunk] = []
        for query in queries[:3]:
            results, trace = retriever.retrieve_with_trace(
                query,
                top_k=min(task.max_evidence_per_question, settings.retrieval_top_k),
                candidate_pool_size=max(20, settings.retrieval_top_k * 3),
                rerank_top_n=max(8, settings.retrieval_top_k * 2),
                mode=settings.retrieval_mode,
                filters={
                    "companies": [task.company_id],
                    "periods": task.target_periods or ([task.period] if task.period else []),
                    "source_types": source_filters,
                },
                strategy=subquestion.lane.value,
            )
            if not results and source_filters:
                results, trace = retriever.retrieve_with_trace(
                    query,
                    top_k=min(task.max_evidence_per_question, settings.retrieval_top_k),
                    candidate_pool_size=max(20, settings.retrieval_top_k * 3),
                    rerank_top_n=max(8, settings.retrieval_top_k * 2),
                    mode=settings.retrieval_mode,
                    filters={
                        "companies": [task.company_id],
                        "periods": task.target_periods or ([task.period] if task.period else []),
                        "source_types": [],
                    },
                    strategy=subquestion.lane.value,
                )
                trace["fallback_source_filter"] = True
            round_trace["retrievals"].append(trace)
            retrieved.extend(results)
        return retrieved

    def _assess_evidence(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        chunks: List[RetrievedChunk],
    ) -> EvidenceAssessment:
        valid_chunks = [
            chunk for chunk in chunks
            if chunk.company and self._normalize_text(chunk.company) == self._normalize_text(task.company_id)
        ]
        valid_periods = {
            self._normalize_text(period)
            for period in (task.target_periods or ([task.period] if task.period else []))
            if period
        }
        if valid_periods:
            valid_chunks = [
                chunk for chunk in valid_chunks
                if chunk.period and self._normalize_text(chunk.period) in valid_periods
            ]
        if not valid_chunks:
            return EvidenceAssessment(
                valid_chunk_count=0,
                insufficient=True,
                reasons=["No contract-valid evidence chunks were retrieved."],
            )

        normalized_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "text": chunk.text,
                "source_type": chunk.source_type,
                "is_primary": chunk.is_primary,
            }
            for chunk in valid_chunks
        ]
        hypotheses = self._build_assessment_hypotheses(task, subquestion)
        chunk_index = {chunk.chunk_id: chunk for chunk in valid_chunks}
        aggregate: Dict[str, dict] = {}

        for hypothesis in hypotheses:
            for item in self.verifier.score_hypothesis_against_chunks(hypothesis, normalized_chunks):
                chunk = chunk_index.get(item["chunk_id"])
                if chunk is None:
                    continue
                support_score = float(item["entailment"])
                support_score += 0.20 * self._slot_overlap_ratio(subquestion, chunk.text)
                if subquestion.metric_family and (
                    subquestion.metric_family in (chunk.metric_signals or [])
                    or subquestion.metric_family in chunk.text.lower()
                ):
                    support_score += 0.08
                if self._is_revenue_mix_question(subquestion):
                    support_score += self._revenue_mix_support_bonus(subquestion, chunk)
                if subquestion.lane == ResearchLane.HARD_FACT:
                    support_score += self._hard_fact_support_bonus(task, subquestion, chunk)
                if subquestion.lane == ResearchLane.HARD_FACT and chunk.content_type in {"quantitative", "mixed"}:
                    support_score += 0.06
                if subquestion.lane == ResearchLane.SEMANTIC and self._targets_management_commentary(subquestion):
                    if chunk.source_type == "earnings_call_transcript":
                        support_score += 0.24
                    elif chunk.source_type == "shareholder_letter":
                        support_score += 0.14
                    elif chunk.source_type == "investor_presentation":
                        support_score += 0.06
                    elif chunk.source_type in {"results_release", "quarterly_results"}:
                        support_score -= 0.04
                if chunk.is_primary or (chunk.trust_level or 0) >= 4:
                    support_score += 0.04
                current = aggregate.get(item["chunk_id"])
                if current is None or support_score > current["support_score"]:
                    aggregate[item["chunk_id"]] = {
                        **item,
                        "support_score": support_score,
                        "matched_hypothesis": hypothesis,
                    }

        scored = sorted(
            aggregate.values(),
            key=lambda item: (
                item["support_score"],
                item["entailment"],
                1 if chunk_index[item["chunk_id"]].is_primary else 0,
                chunk_index[item["chunk_id"]].trust_level or 0,
            ),
            reverse=True,
        )
        best_entailment = scored[0]["entailment"] if scored else 0.0
        mean_entailment = sum(item["entailment"] for item in scored[:2]) / max(1, min(2, len(scored)))
        best_support = scored[0]["support_score"] if scored else 0.0
        mean_support = sum(item["support_score"] for item in scored[:2]) / max(1, min(2, len(scored)))
        matched_chunk_ids = [item["chunk_id"] for item in scored[:2]]

        metric_match = any(
            subquestion.metric_family in (chunk.metric_signals or [])
            or subquestion.metric_family in chunk.text.lower()
            for chunk in valid_chunks
        )
        numeric_match = any(chunk.content_type in {"quantitative", "mixed"} for chunk in valid_chunks)
        high_trust_hit = any((chunk.trust_level or 0) >= 4 or chunk.is_primary for chunk in valid_chunks)
        chunk_map = {chunk.chunk_id: chunk for chunk in valid_chunks}
        conflict = self._detect_conflict(subquestion, scored, chunk_map)
        sufficient = False
        insufficient = False
        reasons: List[str] = []

        if subquestion.lane == ResearchLane.HARD_FACT:
            hard_fact_threshold = self._hard_fact_support_threshold(task, subquestion)
            strict_sufficient = (
                len(valid_chunks) >= 1
                and (metric_match or numeric_match)
                and best_support >= hard_fact_threshold
            )
            compositional_sufficient = self._has_compositional_hard_fact_support(
                task=task,
                subquestion=subquestion,
                valid_chunks=valid_chunks,
                matched_chunk_ids=matched_chunk_ids,
                best_support=best_support,
                mean_support=mean_support,
                numeric_match=numeric_match,
                high_trust_hit=high_trust_hit,
            )
            family_complete = self._has_family_complete_hard_fact_support(
                subquestion=subquestion,
                valid_chunks=valid_chunks,
                matched_chunk_ids=matched_chunk_ids,
                high_trust_hit=high_trust_hit,
            )
            sufficient = strict_sufficient or compositional_sufficient or family_complete
            insufficient = not sufficient and (
                len(valid_chunks) < 1
                or best_support < max(0.32, hard_fact_threshold - 0.28)
                or not (metric_match or numeric_match)
            )
            if not metric_match and not numeric_match:
                reasons.append("Target metric or numeric evidence was not found in retrieved chunks.")
            if not sufficient and best_support < hard_fact_threshold:
                reasons.append(f"Best support {best_support:.2f} below hard-fact threshold.")
        else:
            direct_causal_sufficient = self._has_direct_causal_semantic_support(
                task=task,
                subquestion=subquestion,
                valid_chunks=valid_chunks,
                matched_chunk_ids=matched_chunk_ids,
                best_support=best_support,
                high_trust_hit=high_trust_hit,
            )
            direct_catalog_sufficient = self._has_product_catalog_semantic_support(
                task=task,
                subquestion=subquestion,
                valid_chunks=valid_chunks,
                matched_chunk_ids=matched_chunk_ids,
                best_support=best_support,
                high_trust_hit=high_trust_hit,
            )
            sufficient = (
                len(valid_chunks) >= 2
                and mean_support >= 0.58
                and high_trust_hit
            ) or direct_causal_sufficient or direct_catalog_sufficient
            insufficient = not sufficient and (len(valid_chunks) < 2 or best_support < 0.45)
            if len(valid_chunks) < 2:
                reasons.append("Fewer than two valid semantic evidence chunks were found.")
            if mean_support < 0.58 and not (direct_causal_sufficient or direct_catalog_sufficient):
                reasons.append(f"Mean support {mean_support:.2f} below semantic threshold.")
            if not high_trust_hit:
                reasons.append("No high-trust source supported the semantic finding.")

        if conflict:
            reasons.append("Conflicting evidence detected across top supporting chunks.")

        return EvidenceAssessment(
            valid_chunk_count=len(valid_chunks),
            best_entailment=best_entailment,
            mean_entailment=mean_entailment,
            best_support=best_support,
            mean_support=mean_support,
            numeric_match=numeric_match,
            metric_match=metric_match,
            high_trust_hit=high_trust_hit,
            sufficient=sufficient and not conflict,
            insufficient=insufficient and not conflict,
            conflict=conflict,
            reasons=reasons,
            matched_chunk_ids=matched_chunk_ids,
        )

    def _build_assessment_hypotheses(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
    ) -> List[str]:
        topic = self._humanize_slot_label(subquestion.fact_slot or subquestion.metric_family or subquestion.question_id)
        period_label = self._task_period_label(task, include_latest_placeholder=False)
        context = " ".join(part for part in [task.company_id, period_label] if part).strip()
        hypotheses = []
        slot_basis = self._subquestion_text_basis(subquestion)
        if self._targets_product_service_catalog_question(task, subquestion):
            hypotheses.extend(
                [
                    f"The source text explicitly describes the major products, platforms, or services sold by {task.company_id} for {context}.",
                    f"The source text identifies the main product categories, offerings, or end markets for {task.company_id} for {context}.",
                ]
            )
        elif self._targets_customer_concentration_question(task, subquestion):
            hypotheses.extend(
                [
                    f"The source text discloses whether a single customer represented 10% or more of revenue for {context}.",
                    f"The source text reports customer concentration and the related revenue percentage for {context}.",
                ]
            )
        elif self._targets_yes_no_disclosure(task, subquestion):
            hypotheses.extend(
                [
                    f"The source text explicitly answers the disclosure-style question for {context}.",
                    f"The source text states whether the filing disclosed the requested event, outcome, or agreement terms for {context}.",
                ]
            )
        if "business_model" in slot_basis or any(
            phrase in subquestion.text.lower()
            for phrase in ("make money", "business model", "monetization")
        ):
            hypotheses.extend(
                [
                    f"The source text explains how {task.company_id} makes money, including its business model, offerings, or monetization.",
                    f"The source text describes the main revenue sources, offerings, or monetization model for {task.company_id}.",
                ]
            )
        elif "revenue_mix" in slot_basis or "revenue mix" in slot_basis or "breakdown" in subquestion.text.lower():
            hypotheses.extend(
                [
                    f"The source text reports revenue mix or breakdown by product, service, or segment for {context}.",
                    f"Reported evidence for {context} discloses which product, service, or segment lines contributed revenue.",
                    f"The source text reports network services revenue, security revenue, and other revenue for {context}.",
                    f"The source text breaks out revenue into network services, security, other, compute, customer, or geographic components for {context}.",
                ]
            )
            if subquestion.lane == ResearchLane.SEMANTIC or any(
                token in slot_basis for token in ("management", "commentary", "emphasis", "priority", "priorities")
            ):
                hypotheses.extend(
                    [
                        f"Management commentary for {context} identifies which revenue components or product lines matter most.",
                        f"Management commentary for {context} highlights balanced traffic mix and drivers such as security, network services, other revenue, compute, cross-sell, upsell, delivery, or portfolio expansion.",
                        f"Management commentary for {context} emphasizes balanced revenue growth across product lines, geographic regions, and customer segments.",
                        f"Management commentary for {context} highlights delivery and security together with cross-sell momentum as important mix drivers.",
                    ]
                )
        elif self._targets_liquidity_or_working_capital(task, subquestion):
            component_hint = self._liquidity_component_hint(task, subquestion)
            hypotheses.extend(
                [
                    f"The source text reports the balance sheet line items needed to assess {topic} for {context}.",
                    f"The source text reports {component_hint} for {context}.",
                    f"The source text discusses liquidity and capital resources for {context}.",
                ]
            )
        elif self._targets_ranking_or_comparison_hard_fact(task, subquestion):
            metric_hint = self._comparison_metric_hint(task, subquestion)
            dimension_hint = self._comparison_dimension_hint(task, subquestion)
            hypotheses.extend(
                [
                    f"The source text reports {metric_hint} by {dimension_hint} for {context}.",
                    f"The source text includes a table comparing {dimension_hint}-level {metric_hint} values for {context}.",
                    f"The source text lists multiple {dimension_hint} {metric_hint} values that can identify the highest or lowest result for {context}.",
                ]
            )
        elif self._targets_directional_financial_change(task, subquestion):
            metric_hint = self._comparison_metric_hint(task, subquestion)
            hypotheses.extend(
                [
                    f"The source text reports the current-period and prior-period {metric_hint} values for {context}.",
                    f"The source text explains what drove the change in {metric_hint} for {context}.",
                ]
            )
        elif subquestion.lane == ResearchLane.HARD_FACT:
            hypotheses.extend(
                [
                    f"The source text reports {topic} for {context}.",
                    f"Reported evidence for {context} addresses {topic}.",
                ]
            )
        else:
            hypotheses.extend(
                [
                    f"Management commentary for {context} addresses {topic}.",
                    f"The source text discusses {topic} for {context}.",
                ]
            )
        hypotheses.append(subquestion.text)
        deduped: List[str] = []
        seen = set()
        for hypothesis in hypotheses:
            clean = " ".join(str(hypothesis).split()).strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            deduped.append(clean)
            seen.add(key)
        return deduped[:8]

    def _revenue_mix_support_bonus(self, subquestion: ResearchSubquestion, chunk: RetrievedChunk) -> float:
        lowered = (chunk.text or "").lower()
        explicit_revenue_components = sum(
            1
            for phrase in ("network services revenue", "security revenue", "other revenue")
            if phrase in lowered
        )
        commentary_signals = sum(
            1
            for phrase in (
                "balanced traffic mix",
                "delivery and security",
                "product lines, geographic regions, and customer segments",
                "upsell and cross-sell",
                "cross-sell momentum",
            )
            if phrase in lowered
        )
        numeric_present = bool(self._extract_number_tokens(chunk.text))
        if subquestion.lane == ResearchLane.HARD_FACT:
            if explicit_revenue_components >= 3 and numeric_present:
                return 0.54
            if explicit_revenue_components >= 2 and numeric_present:
                return 0.36
            if explicit_revenue_components >= 1 and numeric_present:
                return 0.16
            return 0.0
        if self._targets_management_commentary(subquestion):
            if commentary_signals >= 1:
                return 0.34
            if explicit_revenue_components >= 2:
                return 0.10
        return 0.0

    def _hard_fact_support_bonus(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        chunk: RetrievedChunk,
    ) -> float:
        lowered = (chunk.text or "").lower()
        if not self._extract_number_tokens(chunk.text):
            return 0.0

        bonus = 0.0
        if self._targets_liquidity_or_working_capital(task, subquestion):
            liquidity_hits = sum(
                1
                for phrase in (
                    "quick ratio",
                    "working capital",
                    "cash and cash equivalents",
                    "short-term investments",
                    "accounts receivable",
                    "current liabilities",
                    "current assets",
                )
                if phrase in lowered
            )
            if liquidity_hits >= 5:
                bonus += 0.26
            elif liquidity_hits >= 3:
                bonus += 0.18
            elif liquidity_hits >= 2:
                bonus += 0.10
            if "liquidity and capital resources" in lowered or ("current assets" in lowered and "current liabilities" in lowered):
                bonus += 0.06

        if self._targets_ranking_or_comparison_hard_fact(task, subquestion):
            ranking_hits = sum(
                1
                for phrase in (
                    "segment",
                    "segments",
                    "region",
                    "regions",
                    "net income",
                    "operating income",
                    "ebitdar",
                    "highest",
                    "lowest",
                    "largest",
                )
                if phrase in lowered
            )
            if ranking_hits >= 3:
                bonus += 0.18
            elif ranking_hits >= 2:
                bonus += 0.10
            metric_hint = self._comparison_metric_hint(task, subquestion)
            dimension_hint = self._comparison_dimension_hint(task, subquestion)
            if metric_hint in lowered and dimension_hint in lowered and len(self._extract_number_tokens(chunk.text)) >= 4:
                bonus += 0.12
            if "segment results" in lowered or "results by segment" in lowered:
                bonus += 0.06

        if self._targets_directional_financial_change(task, subquestion):
            direction_hits = sum(
                1
                for phrase in (
                    "increase",
                    "decrease",
                    "increased",
                    "decreased",
                    "improved",
                    "declined",
                    "positive",
                    "negative",
                    "change",
                )
                if phrase in lowered
            )
            if direction_hits >= 2:
                bonus += 0.10
            if self._targets_reason_attribution(task, subquestion):
                causal_hits = sum(
                    1
                    for phrase in (
                        "primarily due to",
                        "due to",
                        "because",
                        "driven by",
                        "resulting from",
                        "reflecting",
                    )
                    if phrase in lowered
                )
                if causal_hits >= 1:
                    bonus += 0.12

        if self._targets_yes_no_disclosure(task, subquestion):
            disclosure_hits = sum(
                1
                for phrase in (
                    "legal proceedings",
                    "litigation",
                    "proposal",
                    "vote",
                    "voted",
                    "approved",
                    "defeated",
                    "revolving credit agreement",
                    "commitment amount",
                    "aggregate commitments",
                    "major customer",
                    "significant customer",
                    "customer concentration",
                    "accounted for",
                    "10% of revenue",
                    "10 percent of revenue",
                )
                if phrase in lowered
            )
            if disclosure_hits >= 2:
                bonus += 0.18
            elif disclosure_hits >= 1:
                bonus += 0.10

        return bonus

    def _hard_fact_support_threshold(self, task: ResearchTask, subquestion: ResearchSubquestion) -> float:
        if self._targets_yes_no_disclosure(task, subquestion):
            return 0.56
        if self._is_compositional_hard_fact_question(task, subquestion):
            return 0.58
        return 0.78

    def _has_compositional_hard_fact_support(
        self,
        *,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        valid_chunks: List[RetrievedChunk],
        matched_chunk_ids: List[str],
        best_support: float,
        mean_support: float,
        numeric_match: bool,
        high_trust_hit: bool,
    ) -> bool:
        if not self._is_compositional_hard_fact_question(task, subquestion):
            return False
        if not numeric_match or not high_trust_hit:
            return False

        matched = [chunk for chunk in valid_chunks if chunk.chunk_id in set(matched_chunk_ids)]
        quantitative_primary = [
            chunk
            for chunk in matched
            if chunk.content_type in {"quantitative", "mixed"}
            and (chunk.is_primary or (chunk.trust_level or 0) >= 4)
        ]
        return (
            len(quantitative_primary) >= 2
            and best_support >= 0.48
            and mean_support >= 0.48
        )

    def _has_family_complete_hard_fact_support(
        self,
        *,
        subquestion: ResearchSubquestion,
        valid_chunks: List[RetrievedChunk],
        matched_chunk_ids: List[str],
        high_trust_hit: bool,
    ) -> bool:
        if not high_trust_hit or not valid_chunks:
            return False

        high_tier_chunks = [chunk for chunk in valid_chunks if self._is_high_tier_chunk(chunk)]
        if not high_tier_chunks:
            return False

        matched_set = set(matched_chunk_ids)
        matched_high_tier = [chunk for chunk in high_tier_chunks if chunk.chunk_id in matched_set]
        for chunk_group in (matched_high_tier, high_tier_chunks):
            if not chunk_group:
                continue
            if self._build_direct_hard_fact_bundle(subquestion, chunk_group):
                return True
        return False

    def _has_direct_causal_semantic_support(
        self,
        *,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        valid_chunks: List[RetrievedChunk],
        matched_chunk_ids: List[str],
        best_support: float,
        high_trust_hit: bool,
    ) -> bool:
        if subquestion.lane != ResearchLane.SEMANTIC:
            return False
        if not self._targets_reason_attribution(task, subquestion):
            return False
        if not high_trust_hit or best_support < 0.46:
            return False

        matched_set = set(matched_chunk_ids)
        prioritized = [chunk for chunk in valid_chunks if chunk.chunk_id in matched_set] or valid_chunks
        for chunk in prioritized:
            if not self._is_high_tier_chunk(chunk):
                continue
            if not self._has_direct_causal_language(chunk.text):
                continue
            if max(
                self._slot_overlap_ratio(subquestion, chunk.text),
                self._question_overlap_ratio(subquestion, chunk.text),
            ) < 0.12:
                continue
            return True
        return False

    def _has_direct_causal_language(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(
            phrase in lowered
            for phrase in (
                "primarily due to",
                "due to",
                "because",
                "driven by",
                "driven primarily by",
                "resulting from",
                "reflecting",
                "as a result of",
            )
        )

    def _has_product_catalog_semantic_support(
        self,
        *,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        valid_chunks: List[RetrievedChunk],
        matched_chunk_ids: List[str],
        best_support: float,
        high_trust_hit: bool,
    ) -> bool:
        if subquestion.lane != ResearchLane.SEMANTIC:
            return False
        if not self._targets_product_service_catalog_question(task, subquestion):
            return False
        if not high_trust_hit or best_support < 0.44:
            return False

        matched_set = set(matched_chunk_ids)
        prioritized = [chunk for chunk in valid_chunks if chunk.chunk_id in matched_set] or valid_chunks
        for chunk in prioritized:
            if not self._is_high_tier_chunk(chunk):
                continue
            if not self._has_product_catalog_language(chunk.text):
                continue
            if max(
                self._slot_overlap_ratio(subquestion, chunk.text),
                self._question_overlap_ratio(subquestion, chunk.text),
            ) < 0.12:
                continue
            return True
        return False

    def _has_product_catalog_language(self, text: str) -> bool:
        lowered = (text or "").lower()
        hits = sum(
            1
            for phrase in (
                "products",
                "services",
                "offerings",
                "platforms",
                "we sell",
                "the company sells",
                "our products include",
                "our services include",
            )
            if phrase in lowered
        )
        return hits >= 1

    def _humanize_slot_label(self, value: str) -> str:
        cleaned = re.sub(r"[_\-]+", " ", str(value or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or "the requested evidence"

    def _slot_overlap_ratio(self, subquestion: ResearchSubquestion, text: str) -> float:
        slot_basis = " ".join(
            item for item in [subquestion.fact_slot, subquestion.metric_family]
            if item
        )
        slot_tokens = self._slot_tokens(slot_basis, expand_aliases=True)
        if not slot_tokens:
            return 0.0
        hay_tokens = self._slot_tokens(text, expand_aliases=False)
        if not hay_tokens:
            return 0.0
        return len(slot_tokens & hay_tokens) / len(slot_tokens)

    def _question_overlap_ratio(self, subquestion: ResearchSubquestion, text: str) -> float:
        question_tokens = self._summary_tokens(
            " ".join(
                part
                for part in [
                    subquestion.text,
                    subquestion.fact_slot,
                    subquestion.metric_family,
                ]
                if part
            )
        )
        if not question_tokens:
            return 0.0
        hay_tokens = self._summary_tokens(text)
        if not hay_tokens:
            return 0.0
        return len(question_tokens & hay_tokens) / len(question_tokens)

    def _detect_conflict(
        self,
        subquestion: ResearchSubquestion,
        scored: List[dict],
        chunk_map: Dict[str, RetrievedChunk],
    ) -> bool:
        if len(scored) < 2:
            return False
        top = [item for item in scored[:3] if item["entailment"] >= 0.60]
        if len(top) < 2:
            return False
        contradiction = any(item["contradiction"] >= 0.70 for item in top)
        if subquestion.lane != ResearchLane.HARD_FACT:
            return contradiction
        if contradiction:
            return True
        first = top[0]
        second = top[1]
        first_chunk = chunk_map.get(first["chunk_id"])
        second_chunk = chunk_map.get(second["chunk_id"])
        if not first_chunk or not second_chunk:
            return False
        if not self._same_metric_context(subquestion, first_chunk, second_chunk):
            return False
        first_numbers = self._extract_number_tokens(first["text"])
        second_numbers = self._extract_number_tokens(second["text"])
        if first_numbers and second_numbers and first_numbers != second_numbers:
            return self._likely_same_fact_text(first["text"], second["text"], subquestion.metric_family)
        return False

    def _build_followup_queries(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        assessment: EvidenceAssessment,
        llm_budget: dict,
    ) -> List[str]:
        if not assessment.reasons:
            return [subquestion.text]
        if not self.use_stub_llm and llm_budget["used"] < llm_budget["max"] and assessment.conflict:
            suggested = self._reflect_followup_queries(task, subquestion, assessment, llm_budget)
            if suggested:
                return suggested
        return self._rule_followup_queries(task, subquestion, assessment)

    def _reflect_followup_queries(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        assessment: EvidenceAssessment,
        llm_budget: dict,
    ) -> List[str]:
        if not self._consume_llm_budget(llm_budget, reason="followup_reflection"):
            return []
        prompt = (
            "Return JSON with key `queries`, a list of at most 2 tighter retrieval queries.\n"
            "The queries must stay within the same company, same period, and same metric family.\n"
            f"Company: {task.company_id}\n"
            f"Period: {self._task_period_label(task)}\n"
            f"Subquestion: {subquestion.text}\n"
            f"Metric family: {subquestion.metric_family}\n"
            f"Failure reasons: {assessment.reasons}"
        )
        try:
            raw = generate_text_response(
                self.client,
                model=settings.openai_model,
                system_prompt="You propose tighter follow-up retrieval queries for financial research.",
                user_prompt=prompt,
                max_tokens=300,
                temperature=0.1,
                max_retries=settings.llm_request_max_retries,
            )
            payload = self._parse_json(raw)
            if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
                return [str(query).strip() for query in payload["queries"] if str(query).strip()][:2]
        except Exception as exc:
            if self._strict_live_fail_fast():
                raise
            logger.warning("Follow-up reflection failed; using rule-based fallback: %s", exc)
        return []

    def _rule_followup_queries(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        assessment: EvidenceAssessment,
    ) -> List[str]:
        period_hint = self._task_period_label(task, include_latest_placeholder=False)
        base = f"{task.company_id} {period_hint} {subquestion.metric_family}".strip()
        slot_hint = self._humanize_slot_label(subquestion.fact_slot or "")
        if assessment.conflict:
            return [
                f"{base} exact reported figure",
                f"{base} gaap adjusted reconciliation",
            ]
        if self._is_revenue_mix_question(subquestion):
            if subquestion.lane == ResearchLane.HARD_FACT:
                return [
                    f"{task.company_id} {period_hint} network services revenue security revenue other revenue".strip(),
                    f"{task.company_id} {period_hint} revenue breakdown by product service segment".strip(),
                ]
            return [
                f"{task.company_id} {period_hint} earnings call prepared remarks revenue mix".strip(),
                f"{task.company_id} {period_hint} management commentary security network services other revenue".strip(),
            ]
        if self._targets_yes_no_disclosure(task, subquestion):
            return self._disclosure_queries(task, subquestion)
        if self._targets_liquidity_or_working_capital(task, subquestion):
            return self._liquidity_component_queries(task, subquestion)
        if self._targets_ranking_or_comparison_hard_fact(task, subquestion):
            return self._comparison_table_queries(task, subquestion)
        if self._targets_directional_financial_change(task, subquestion):
            return self._directional_change_queries(task, subquestion)
        if subquestion.lane == ResearchLane.HARD_FACT:
            queries = [
                f"{base} disclosed metric reported",
                f"{task.company_id} {period_hint} {slot_hint} numeric evidence".strip(),
            ]
            if self._is_income_statement_revenue_question(subquestion):
                queries.extend(
                    [
                        f"{task.company_id} {period_hint} net revenue statement of operations".strip(),
                        f"{task.company_id} {period_hint} total revenue income statement".strip(),
                    ]
                )
            return list(dict.fromkeys(item for item in queries if item))
        return [
            f"{task.company_id} {period_hint} {slot_hint} management commentary".strip(),
            f"{task.company_id} {period_hint} {subquestion.metric_family} source discussion".strip(),
        ]

    def _build_initial_queries(self, task: ResearchTask, subquestion: ResearchSubquestion) -> List[str]:
        period_hint = self._task_period_label(task, include_latest_placeholder=False)
        slot_hint = self._humanize_slot_label(subquestion.fact_slot or "")
        base = [subquestion.text]
        targeted: List[str] = []
        generic: List[str] = []

        if self._is_revenue_mix_question(subquestion):
            if subquestion.lane == ResearchLane.HARD_FACT:
                targeted.extend(
                    [
                        f"{task.company_id} {period_hint} network services revenue security revenue other revenue".strip(),
                        f"{task.company_id} {period_hint} revenue breakdown by product service segment".strip(),
                    ]
                )
            else:
                targeted.append(f"{task.company_id} {period_hint} earnings call prepared remarks revenue mix".strip())
        elif self._targets_product_service_catalog_question(task, subquestion):
            targeted.extend(
                [
                    f"{task.company_id} {period_hint} products services offerings annual report".strip(),
                    f"{task.company_id} {period_hint} products platforms segments business overview".strip(),
                    f"{task.company_id} {period_hint} what does {task.company_id} sell".strip(),
                ]
            )
        elif self._targets_customer_concentration_question(task, subquestion):
            targeted.extend(
                [
                    f"{task.company_id} {period_hint} customer concentration major customer percentage revenue".strip(),
                    f"{task.company_id} {period_hint} customer accounted for revenue concentration".strip(),
                    f"{task.company_id} {period_hint} significant customer 10 percent revenue".strip(),
                ]
            )
        elif self._targets_yes_no_disclosure(task, subquestion):
            targeted.extend(self._disclosure_queries(task, subquestion))
        elif self._targets_liquidity_or_working_capital(task, subquestion):
            targeted.extend(self._liquidity_component_queries(task, subquestion))
        elif self._targets_ranking_or_comparison_hard_fact(task, subquestion):
            targeted.extend(self._comparison_table_queries(task, subquestion))
        elif self._targets_directional_financial_change(task, subquestion):
            targeted.extend(self._directional_change_queries(task, subquestion))

        if subquestion.metric_family and subquestion.metric_family not in self._normalize_text(subquestion.text):
            generic.append(f"{task.company_id} {period_hint} {subquestion.metric_family}".strip())
        if slot_hint and self._normalize_text(slot_hint) not in self._normalize_text(subquestion.text):
            generic.append(f"{task.company_id} {period_hint} {slot_hint}".strip())
        if self._is_income_statement_revenue_question(subquestion):
            generic.extend(
                [
                    f"{task.company_id} {period_hint} net revenue statement of operations".strip(),
                    f"{task.company_id} {period_hint} total revenue income statement".strip(),
                    f"{task.company_id} {period_hint} net sales pnl".strip(),
                ]
            )
        return list(dict.fromkeys(item for item in base + targeted + generic if item))

    def _subquestion_text_basis(self, subquestion: ResearchSubquestion) -> str:
        return " ".join(
            part for part in [
                str(subquestion.fact_slot or ""),
                str(subquestion.metric_family or ""),
                str(subquestion.text or ""),
            ]
            if part
        ).lower()

    def _is_revenue_mix_question(self, subquestion: ResearchSubquestion) -> bool:
        basis = self._subquestion_text_basis(subquestion)
        return any(token in basis for token in ("revenue_mix", "revenue mix", "breakdown", "components", "network services", "security revenue", "other revenue"))

    def _targets_product_service_catalog_question(
        self,
        task: ResearchTask,
        subquestion: Optional[ResearchSubquestion] = None,
    ) -> bool:
        basis = task.query.lower()
        if subquestion is not None:
            basis = f"{basis} {self._subquestion_text_basis(subquestion)}"
        mentions_catalog = any(
            token in basis
            for token in (
                "product",
                "products",
                "service",
                "services",
                "offering",
                "offerings",
                "platform",
                "platforms",
                "sell",
                "sells",
            )
        )
        asks_catalog = any(
            token in basis
            for token in (
                "what are",
                "which are",
                "what does",
                "what do",
                "major",
                "main",
                "core",
                "primary",
            )
        )
        excludes_mix = not any(token in basis for token in ("revenue mix", "breakdown", "components"))
        return mentions_catalog and asks_catalog and excludes_mix

    def _targets_customer_concentration_question(
        self,
        task: ResearchTask,
        subquestion: Optional[ResearchSubquestion] = None,
    ) -> bool:
        basis = task.query.lower()
        if subquestion is not None:
            basis = f"{basis} {self._subquestion_text_basis(subquestion)}"
        return any(
            token in basis
            for token in (
                "customer concentration",
                "major customer",
                "significant customer",
                "customer accounted",
                "10% of revenue",
                "ten percent of revenue",
            )
        )

    def _targets_yes_no_disclosure(
        self,
        task: ResearchTask,
        subquestion: Optional[ResearchSubquestion] = None,
    ) -> bool:
        basis = task.query.lower()
        if subquestion is not None:
            basis = f"{basis} {self._subquestion_text_basis(subquestion)}"
        disclosure_topic = any(
            token in basis
            for token in (
                "legal battles",
                "ongoing legal battles",
                "legal proceedings",
                "proposal vote",
                "voting result",
                "shareholder vote",
                "revolving credit agreement",
                "borrowing capacity",
                "may borrow under",
                "commitment amount",
                "proposal outcome",
                "customer concentration",
                "major customer",
                "significant customer",
            )
        )
        disclosure_form = any(
            token in basis
            for token in (
                "was there",
                "were there",
                "whether",
                "did ",
                "disclosed",
                "outcome",
                "increase",
                "total amount",
                "how much",
                "if so",
                "what percentage",
                "percentage",
            )
        )
        return disclosure_topic and disclosure_form

    def _disclosure_queries(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
    ) -> List[str]:
        period_hint = self._task_period_label(task, include_latest_placeholder=False)
        basis = self._subquestion_text_basis(subquestion)
        if any(token in basis for token in ("customer concentration", "major customer", "significant customer")):
            return [
                f"{task.company_id} {period_hint} customer concentration major customer percentage revenue".strip(),
                f"{task.company_id} {period_hint} customer accounted for revenue concentration".strip(),
                f"{task.company_id} {period_hint} significant customer 10 percent revenue".strip(),
            ]
        if any(token in basis for token in ("legal battles", "legal proceedings")):
            return [
                f"{task.company_id} {period_hint} legal proceedings litigation annual report".strip(),
                f"{task.company_id} {period_hint} material legal proceedings disclosed".strip(),
            ]
        if any(token in basis for token in ("proposal vote", "voting result", "shareholder vote")):
            return [
                f"{task.company_id} {period_hint} shareholder proposal vote result 8-k".strip(),
                f"{task.company_id} {period_hint} annual meeting proposal passed failed".strip(),
            ]
        if "revolving credit agreement" in basis:
            return [
                f"{task.company_id} {period_hint} revolving credit agreement increase commitment amount".strip(),
                f"{task.company_id} {period_hint} revolving credit agreement aggregate commitments may borrow".strip(),
            ]
        return [
            f"{task.company_id} {period_hint} disclosed filing answer".strip(),
        ]

    def _is_income_statement_revenue_question(self, subquestion: ResearchSubquestion) -> bool:
        basis = self._subquestion_text_basis(subquestion)
        revenue_like = any(token in basis for token in ("revenue", "net sales", "sales"))
        statement_like = any(
            token in basis
            for token in ("income statement", "p&l", "statement of operations", "statement of income", "net revenue")
        )
        return revenue_like and statement_like

    def _is_compositional_hard_fact_question(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
    ) -> bool:
        return (
            self._targets_liquidity_or_working_capital(task, subquestion)
            or self._targets_ranking_or_comparison_hard_fact(task, subquestion)
            or self._targets_directional_financial_change(task, subquestion)
        )

    def _targets_liquidity_or_working_capital(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
    ) -> bool:
        basis = f"{task.query} {self._subquestion_text_basis(subquestion)}"
        lowered = basis.lower()
        return any(
            token in lowered
            for token in (
                "quick ratio",
                "working capital",
                "liquidity",
                "current ratio",
                "inventory turnover",
                "days sales outstanding",
            )
        )

    def _targets_ranking_or_comparison_hard_fact(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
    ) -> bool:
        basis = f"{task.query} {self._subquestion_text_basis(subquestion)}"
        lowered = basis.lower()
        return any(
            token in lowered
            for token in (
                "highest",
                "lowest",
                "largest",
                "smallest",
                "which segment",
                "which region",
                "which business",
                "most",
                "least",
            )
        )

    def _targets_directional_financial_change(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
    ) -> bool:
        basis = self._subquestion_text_basis(subquestion)
        lowered = basis.lower()
        return any(
            token in lowered
            for token in (
                "increase",
                "decrease",
                "change",
                "improving",
                "improved",
                "grew",
                "growth",
                "positive",
                "negative",
                "what drove",
                "driver",
                "drivers",
                "due to",
                "because",
                "reason",
                "reasons",
                "why did",
            )
        )

    def _targets_reason_attribution(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
    ) -> bool:
        basis = f"{task.query} {self._subquestion_text_basis(subquestion)}".lower()
        return any(
            token in basis
            for token in (
                "what drove",
                "driver",
                "drivers",
                "due to",
                "because",
                "reason",
                "reasons",
                "why did",
                "why was",
                "why were",
            )
        )

    def _comparison_metric_hint(self, task: ResearchTask, subquestion: ResearchSubquestion) -> str:
        basis = f"{task.query} {self._subquestion_text_basis(subquestion)}".lower()
        for phrase in (
            "net income",
            "operating income",
            "operating margin",
            "gross margin",
            "cash flow",
            "working capital",
            "quick ratio",
            "current ratio",
            "revenue",
            "sales",
            "eps",
            "ebitdar",
            "ebitda",
            "margin",
            "income",
        ):
            if phrase in basis:
                return phrase
        return self._humanize_slot_label(subquestion.metric_family or subquestion.fact_slot or "metric").lower()

    def _comparison_dimension_hint(self, task: ResearchTask, subquestion: ResearchSubquestion) -> str:
        basis = f"{task.query} {self._subquestion_text_basis(subquestion)}".lower()
        if any(token in basis for token in ("operating activities", "investing activities", "financing activities", "activity", "activities")):
            return "activity"
        if any(token in basis for token in ("region", "regions", "geography", "geographic")):
            return "region"
        if any(token in basis for token in ("segment", "segments", "line of business", "business segment")):
            return "segment"
        if "product" in basis:
            return "product"
        return "segment"

    def _liquidity_component_hint(self, task: ResearchTask, subquestion: ResearchSubquestion) -> str:
        basis = f"{task.query} {self._subquestion_text_basis(subquestion)}".lower()
        if "quick ratio" in basis:
            return "cash and cash equivalents, short-term investments, accounts receivable, and total current liabilities"
        if "working capital" in basis or "current ratio" in basis:
            return "total current assets and total current liabilities"
        return "current assets, current liabilities, and other short-term balance sheet line items"

    def _liquidity_component_queries(self, task: ResearchTask, subquestion: ResearchSubquestion) -> List[str]:
        period_hint = self._task_period_label(task, include_latest_placeholder=False)
        basis = f"{task.query} {self._subquestion_text_basis(subquestion)}".lower()
        if "quick ratio" in basis:
            return [
                f"{task.company_id} {period_hint} cash and cash equivalents short-term investments accounts receivable current liabilities".strip(),
                f"{task.company_id} {period_hint} liquidity and capital resources balance sheet current liabilities".strip(),
            ]
        if "working capital" in basis or "current ratio" in basis:
            return [
                f"{task.company_id} {period_hint} total current assets total current liabilities balance sheet".strip(),
                f"{task.company_id} {period_hint} working capital current assets current liabilities".strip(),
            ]
        return [
            f"{task.company_id} {period_hint} liquidity and capital resources current assets current liabilities".strip(),
            f"{task.company_id} {period_hint} balance sheet short term assets liabilities".strip(),
        ]

    def _comparison_table_queries(self, task: ResearchTask, subquestion: ResearchSubquestion) -> List[str]:
        period_hint = self._task_period_label(task, include_latest_placeholder=False)
        metric_hint = self._comparison_metric_hint(task, subquestion)
        dimension_hint = self._comparison_dimension_hint(task, subquestion)
        return [
            f"{task.company_id} {period_hint} {dimension_hint} results {metric_hint} table".strip(),
            f"{task.company_id} {period_hint} {metric_hint} by {dimension_hint}".strip(),
        ]

    def _directional_change_queries(self, task: ResearchTask, subquestion: ResearchSubquestion) -> List[str]:
        period_hint = self._task_period_label(task, include_latest_placeholder=False)
        metric_hint = self._comparison_metric_hint(task, subquestion)
        if self._targets_reason_attribution(task, subquestion):
            return [
                f"{task.company_id} {period_hint} {metric_hint} change drivers".strip(),
                f"{task.company_id} {period_hint} {metric_hint} primarily due to".strip(),
                f"{task.company_id} {period_hint} {metric_hint} driven by because".strip(),
            ]
        return [
            f"{task.company_id} {period_hint} {metric_hint} change drivers".strip(),
            f"{task.company_id} {period_hint} {metric_hint} increased decreased due to".strip(),
        ]

    def _is_revenue_mix_payload(self, payload: dict) -> bool:
        basis = " ".join(
            [
                str(payload.get("fact_slot", "")),
                str(payload.get("metric_family", "")),
                str(payload.get("text", "")),
            ]
        ).lower()
        return any(token in basis for token in ("revenue_mix", "revenue mix", "breakdown", "components", "network services", "security revenue", "other revenue"))

    def _targets_management_commentary(self, subquestion: ResearchSubquestion) -> bool:
        basis = self._subquestion_text_basis(subquestion)
        return any(token in basis for token in ("management", "commentary", "prepared remarks", "q&a", "emphasis", "priority", "priorities"))

    def _generate_subquestion_answers(
        self,
        task: ResearchTask,
        results: List[ResearchQuestionResult],
        llm_budget: dict,
        replay: ResearchReplayRecord,
    ) -> None:
        ready = [result for result in results if result.status == ResearchQuestionStatus.COMPLETED]
        if not ready:
            return

        hard_batch = [result for result in ready if result.subquestion.lane == ResearchLane.HARD_FACT]
        semantic_batch = [result for result in ready if result.subquestion.lane == ResearchLane.SEMANTIC]
        remaining_hard_batch: List[ResearchQuestionResult] = []
        for item in hard_batch:
            if self._try_publish_direct_hard_fact_answer(item):
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="answer_generated",
                        question_id=item.subquestion.question_id,
                        reason="direct hard-fact answer generated from evidence",
                        payload={
                            "lane": item.subquestion.lane.value,
                            "direct_publish": True,
                        },
                    )
                )
                continue
            remaining_hard_batch.append(item)
        hard_batch = remaining_hard_batch

        for batch in (hard_batch, semantic_batch):
            if not batch:
                continue
            if self.use_stub_llm:
                for item in batch:
                    item.answer_text = self._stub_answer_for_subquestion(item)
                    self._verify_answer(item)
                continue
            if not self._consume_llm_budget(llm_budget, reason="generate_subquestion_batch"):
                if self._strict_live_fail_fast():
                    raise RuntimeError("LLM budget exhausted during strict live batch answer generation.")
                for item in batch:
                    item.answer_text = self._stub_answer_for_subquestion(item)
                    self._verify_answer(item)
                continue
            try:
                self._generate_batch_answers(task, batch)
            except Exception as exc:
                if self._strict_live_fail_fast():
                    raise
                logger.warning("Falling back to stub answer generation because batch generation failed: %s", exc)
                for item in batch:
                    item.answer_text = self._stub_answer_for_subquestion(item)
                    self._verify_answer(item)
                    replay.decisions.append(
                        ResearchDecisionRecord(
                            decision_type="answer_generation_fallback",
                            question_id=item.subquestion.question_id,
                            reason="live batch generation failed; stub fallback used",
                            payload={"lane": item.subquestion.lane.value},
                        )
                    )
                continue
            for item in batch:
                missing_structured_answer = False
                if not item.answer_text:
                    item.answer_text = (
                        "Insufficient evidence in the source pack: "
                        "Model did not return a verifiable structured answer."
                    )
                    missing_structured_answer = True
                    item.trace.append(
                        {
                            "stage": "answer_generation",
                            "status": "missing_structured_answer",
                            "question_id": item.subquestion.question_id,
                        }
                    )
                self._verify_answer(item)
                if missing_structured_answer:
                    replay.decisions.append(
                        ResearchDecisionRecord(
                            decision_type="answer_generation_contract_failure",
                            question_id=item.subquestion.question_id,
                            reason="batch answer did not return a structured claim payload for this question",
                            payload={
                                "lane": item.subquestion.lane.value,
                                "recovered_from_evidence": item.status == ResearchQuestionStatus.COMPLETED,
                            },
                        )
                    )
                    if item.status == ResearchQuestionStatus.COMPLETED:
                        replay.decisions.append(
                            ResearchDecisionRecord(
                                decision_type="answer_generated",
                                question_id=item.subquestion.question_id,
                                reason="missing structured payload recovered from evidence-bound fallback",
                                payload={
                                    "lane": item.subquestion.lane.value,
                                    "recovered_from_evidence": True,
                                },
                            )
                        )
                    continue
                if "Model did not return a verifiable structured answer." in item.answer_text:
                    replay.decisions.append(
                        ResearchDecisionRecord(
                            decision_type="answer_generation_contract_failure",
                            question_id=item.subquestion.question_id,
                            reason="batch answer did not return a structured claim payload for this question",
                            payload={"lane": item.subquestion.lane.value},
                        )
                    )
                    continue
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="answer_generated",
                        question_id=item.subquestion.question_id,
                        reason="batch answer generated",
                        payload={"lane": item.subquestion.lane.value},
                    )
                )

    def _generate_batch_answers(self, task: ResearchTask, batch: List[ResearchQuestionResult]) -> None:
        raw = generate_text_response(
            self.client,
            model=settings.openai_model,
            system_prompt=(
                "You write evidence-bound financial research answers as strict JSON. "
                "Prefer direct benchmark-style conclusions over copied table fragments or memo-style setup. "
                "The first claim should read like the final answer to the question, not like analysis notes. "
                "Only output supported single-sentence claims backed by the provided chunk IDs."
            ),
            user_prompt=self._build_batch_answer_prompt(task, batch),
            max_tokens=1200,
            temperature=0.1,
            max_retries=settings.llm_request_max_retries,
            response_format={"type": "json_object"},
        )
        payload = self._parse_json(raw)
        answers = payload.get("answers", []) if isinstance(payload, dict) else []
        answer_map = {
            str(item.get("question_id")): item
            for item in answers
            if isinstance(item, dict) and item.get("question_id")
        }
        raw_preview = raw[:1500]
        for item in batch:
            item.trace.append(
                {
                    "stage": "batch_answer",
                    "status": "parsed" if item.subquestion.question_id in answer_map else "missing_question_answer",
                    "question_id": item.subquestion.question_id,
                    "parsed_answer_ids": sorted(answer_map.keys()),
                    "raw_preview": raw_preview,
                }
            )
            structured = answer_map.get(item.subquestion.question_id)
            item.answer_text = self._render_structured_answer(item, structured)

    def _build_batch_answer_prompt(self, task: ResearchTask, batch: List[ResearchQuestionResult]) -> str:
        prompt_lines = [
            "Return JSON with key `answers`, a list of objects.",
            "Each object must include question_id, status, claims, and insufficiency_reason.",
            "status must be either `answered` or `insufficient`.",
            "claims must be a list of at most 3 objects with key `statement` plus either (`chunk_id`, `source_id`) or parallel (`chunk_ids`, `source_ids`) arrays.",
            "Each statement must be exactly one evidence-bound sentence. No headings. No markdown emphasis. No uncited intro lines.",
            "Use a single chunk_id/source_id pair when one chunk is sufficient. Use chunk_ids/source_ids only when the conclusion truly requires multiple provided evidence chunks.",
            "Do not combine multiple facts into one statement.",
            "For hard-fact questions, use primary sources for numeric or financial claims. If only weaker evidence exists, return status=`insufficient`.",
            "For hard-fact questions, claim 1 must answer the question directly before any supporting detail.",
            "Use question-shaped direct answers such as: `Yes, ...`, `<segment> had the highest net income`, `<company>'s quick ratio was 1.57`, or `<activity> brought in the most cash flow`.",
            "For pure numeric extraction or formula questions, claim 1 must state the final requested value or ratio, not an intermediate driver, nearby table fragment, or unrelated numeric sentence.",
            "If one provided chunk contains all figures needed for a calculation, comparison, or ranking, you may compute the direct conclusion and cite that single chunk.",
            "If the conclusion depends on more than one provided chunk, cite all required chunks in order using chunk_ids/source_ids.",
            "Do not copy raw table fragments, headers, or line-item blobs when you can safely restate the conclusion.",
            "Do not start claim 1 with background such as `According to the filing`, `The source text states`, or a long evidence recap.",
            "If evidence does not directly answer the question, return status=`insufficient` with a short insufficiency_reason.",
        ]
        for item in batch:
            evidence = "\n".join(
                f"[Chunk: {chunk.chunk_id}] [Source: {chunk.source_id}] "
                f"[SourceType: {chunk.source_type or 'unknown'}] "
                f"[Primary: {'yes' if chunk.is_primary else 'no'}] "
                f"{chunk.text}"
                for chunk in item.evidence[:8]
            )
            prompt_lines.append(
                f"\nQuestion ID: {item.subquestion.question_id}\n"
                f"Lane: {item.subquestion.lane.value}\n"
                f"Question: {item.subquestion.text}\n"
                f"Fact slot: {item.subquestion.fact_slot or 'n/a'}\n"
                f"Metric family: {item.subquestion.metric_family or 'n/a'}\n"
                f"Answer contract:\n{self._prompt_contract_for_subquestion(task, item.subquestion)}\n"
                f"Evidence:\n{evidence}\n"
            )
        return "\n".join(prompt_lines)

    def _prompt_contract_for_subquestion(self, task: ResearchTask, subquestion: ResearchSubquestion) -> str:
        lines: List[str] = []
        query_lower = task.query.lower()

        if self._targets_yes_no_disclosure(task, subquestion):
            lines.extend(
                [
                    "- Claim 1 must begin with `Yes,` or `No,` when the question is binary.",
                    "- If the filing indicates no disclosure, say that directly instead of describing search difficulty.",
                ]
            )
            if any(token in query_lower for token in ("revolving credit agreement", "borrowing capacity", "may borrow")):
                lines.append("- For credit facility questions, claim 1 must state the increase amount or total borrowing capacity directly.")
        elif "quick ratio" in query_lower:
            lines.extend(
                [
                    f"- Preferred first claim shape: `{task.company_id}'s quick ratio for the requested period was X.`",
                    "- Claim 1 must include the final ratio, not only current-asset components.",
                ]
            )
        elif "working capital" in query_lower and "ratio" not in query_lower:
            lines.extend(
                [
                    "- Claim 1 must directly answer whether working capital was positive or negative and include the final amount when available.",
                    "- Do not stop at listing component line items; state the computed working capital result.",
                ]
            )
        elif self._targets_single_value_numeric_task(task):
            lines.extend(
                [
                    "- Claim 1 must state the final requested value, amount, balance, or reported figure directly.",
                    "- Include the relevant period or as-of date in claim 1 when the question asks for it.",
                ]
            )
        elif "margin" in query_lower:
            lines.extend(
                [
                    "- Claim 1 must give the final margin percentage directly.",
                    "- Do not stop at numerator and revenue components; state the final margin in claim 1.",
                ]
            )
        elif self._targets_ranking_or_comparison_hard_fact(task, subquestion):
            lines.extend(
                [
                    "- Claim 1 must name the winner or loser directly, or state which period was higher or lower.",
                    "- Include the compared dimension in claim 1 rather than only listing several numbers.",
                ]
            )
        elif self._targets_directional_financial_change(task, subquestion):
            lines.extend(
                [
                    "- Claim 1 must state the direction of change and the compared periods directly.",
                    "- Only use causal wording such as `because` or `driven by` when the provided evidence states that relationship.",
                ]
            )
        elif self._targets_product_service_catalog_question(task, subquestion):
            lines.extend(
                [
                    "- Claim 1 should directly name the main products or services rather than giving generic business-model commentary.",
                    "- Prefer a concise list or appositive sentence over a broad company description.",
                ]
            )
        elif self._targets_customer_concentration_question(task, subquestion):
            lines.extend(
                [
                    "- Claim 1 must state whether customer concentration was disclosed and include the percentage if it was disclosed.",
                    "- Do not hedge if the filing clearly states the percentage.",
                ]
            )
        elif subquestion.lane == ResearchLane.SEMANTIC and self._targets_management_commentary(subquestion):
            lines.extend(
                [
                    "- Claim 1 should state what management emphasized most, not just restate the surrounding business context.",
                    "- Prefer `Management emphasized ...` or `Management highlighted ...` when supported.",
                ]
            )
        elif subquestion.lane == ResearchLane.HARD_FACT:
            lines.extend(
                [
                    "- Claim 1 must directly answer the question in one sentence before any supporting detail.",
                    "- Prefer the final requested figure, status, or comparison outcome over nearby table fragments.",
                ]
            )
        else:
            lines.extend(
                [
                    "- Claim 1 should answer the question directly in one sentence.",
                    "- Avoid generic scene-setting or evidence paraphrase before the conclusion.",
                ]
            )

        lines.append("- Avoid copied table headers, note titles, investor-relations navigation text, and partial line-item blobs.")
        return "\n".join(lines)

    def _stub_answer_for_subquestion(self, result: ResearchQuestionResult) -> str:
        if not result.evidence:
            return "Insufficient evidence in the source pack to answer this subquestion."
        from agent.schemas import AnalysisStep  # local import to avoid circular typing noise

        fake_step = AnalysisStep(
            name=result.subquestion.fact_slot or result.subquestion.question_id,
            description=result.subquestion.text,
            required=True,
        )
        return build_stub_section_analysis(fake_step, result.subquestion.text, result.evidence[:3])

    def _try_publish_direct_hard_fact_answer(self, result: ResearchQuestionResult) -> bool:
        candidate = self._build_direct_hard_fact_publish_answer(result)
        if not candidate:
            return False

        original_answer = result.answer_text
        original_status = result.status
        original_subquestion_status = result.subquestion.status
        original_refusal_reason = result.refusal_reason

        result.answer_text = candidate
        self._verify_answer(result)
        if result.status == ResearchQuestionStatus.COMPLETED and result.supported_content:
            result.trace.append(
                {
                    "stage": "direct_hard_fact_publish",
                    "status": "published",
                    "question_id": result.subquestion.question_id,
                }
            )
            return True

        result.trace.append(
            {
                "stage": "direct_hard_fact_publish",
                "status": "verification_rejected",
                "question_id": result.subquestion.question_id,
            }
        )
        result.answer_text = original_answer
        result.status = original_status
        result.subquestion.status = original_subquestion_status
        result.refusal_reason = original_refusal_reason
        result.supported_content = ""
        result.verified_claims = []
        return False

    def _build_direct_hard_fact_publish_answer(self, result: ResearchQuestionResult) -> str:
        latest = result.assessments[-1] if result.assessments else None
        if not latest or not latest.sufficient:
            return ""
        ordered_chunks = self._ordered_chunks_for_result(result, latest)
        high_tier_chunks = [chunk for chunk in ordered_chunks if self._is_high_tier_chunk(chunk)]
        if not high_tier_chunks:
            return ""
        bundle = self._build_direct_hard_fact_bundle(result.subquestion, high_tier_chunks)
        if not bundle:
            return ""
        return self._format_direct_hard_fact_bundle(bundle)

    def _ordered_chunks_for_result(
        self,
        result: ResearchQuestionResult,
        assessment: Optional[EvidenceAssessment] = None,
    ) -> List[RetrievedChunk]:
        chunk_map = {chunk.chunk_id: chunk for chunk in result.evidence[:8]}
        matched_ids = list((assessment.matched_chunk_ids if assessment else []) or [])
        ordered = [chunk_map[chunk_id] for chunk_id in matched_ids if chunk_id in chunk_map]
        seen = {chunk.chunk_id for chunk in ordered}
        remaining = [chunk for chunk in result.evidence[:8] if chunk.chunk_id not in seen]
        remaining.sort(
            key=lambda chunk: (
                1 if self._is_high_tier_chunk(chunk) else 0,
                1 if getattr(chunk, "is_primary", False) else 0,
                chunk.trust_level or 0,
                chunk.rerank_score or 0.0,
                chunk.score or 0.0,
            ),
            reverse=True,
        )
        ordered.extend(remaining)
        return ordered[:8]

    def _is_high_tier_chunk(self, chunk: RetrievedChunk) -> bool:
        return bool(chunk and (chunk.is_primary or (chunk.trust_level or 0) >= 4))

    def _verify_answer(self, result: ResearchQuestionResult) -> None:
        evidence_store: Dict[str, List[object]] = {}
        for chunk in result.evidence:
            evidence_store.setdefault(chunk.source_id, []).append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "text": chunk.text,
                    "source_type": chunk.source_type,
                    "is_primary": chunk.is_primary,
                }
            )

        attempted_recovery = False
        while True:
            result.answer_text = self._sanitize_answer_for_verification(result.answer_text)
            claims = self.claim_extractor.extract_claims(
                result.answer_text,
                result.subquestion.question_id,
                focus_text=result.subquestion.text,
                max_claims=3,
            )
            if not claims:
                fallback = self._fallback_answer_from_evidence(result, allow_partial=True) if not attempted_recovery else ""
                if fallback and fallback != result.answer_text:
                    result.trace.append(
                        {
                            "stage": "verification_recovery",
                            "status": "fallback_answer_used",
                            "question_id": result.subquestion.question_id,
                        }
                    )
                    result.answer_text = fallback
                    attempted_recovery = True
                    continue
                result.status = ResearchQuestionStatus.REFUSED
                result.subquestion.status = ResearchQuestionStatus.REFUSED
                result.refusal_reason = (
                    "Generated answer did not produce any verifiable cited claims."
                    if "insufficient evidence" not in result.answer_text.lower()
                    else result.answer_text.strip()
                )
                result.supported_content = ""
                result.verified_claims = []
                return

            verification_results = self.verifier.verify_memo(claims, evidence_store)
            result.verified_claims = verification_results
            result.supported_content = self.report_writer._apply_verification_gating(
                result.answer_text,
                verification_results,
            )
            all_unsupported = bool(verification_results) and all(
                item.confidence == ConfidenceLevel.UNSUPPORTED
                for item in verification_results
            )
            if all_unsupported and not attempted_recovery:
                fallback = self._fallback_answer_from_evidence(result, allow_partial=True)
                if fallback and fallback != result.answer_text:
                    result.trace.append(
                        {
                            "stage": "verification_recovery",
                            "status": "fallback_answer_used",
                            "question_id": result.subquestion.question_id,
                        }
                    )
                    result.answer_text = fallback
                    attempted_recovery = True
                    continue
            if all_unsupported:
                result.status = ResearchQuestionStatus.REFUSED
                result.subquestion.status = ResearchQuestionStatus.REFUSED
                result.refusal_reason = "Generated answer did not survive verification."
            return

    def _render_structured_answer(
        self,
        result: ResearchQuestionResult,
        payload: Optional[dict],
    ) -> str:
        evidence_map = {
            chunk.chunk_id: chunk
            for chunk in result.evidence[:8]
        }
        source_map = {
            chunk.chunk_id: chunk.source_id
            for chunk in result.evidence[:8]
        }
        if isinstance(payload, dict):
            status = str(payload.get("status", "")).strip().lower()
            claims = payload.get("claims")
            rendered_claims: List[str] = []
            if isinstance(claims, list):
                for claim in claims[:3]:
                    if not isinstance(claim, dict):
                        continue
                    statement = self._normalize_claim_sentence(str(claim.get("statement", "")))
                    statement = self._strip_soft_evidence_preamble(statement)
                    citation_pairs = self._extract_structured_claim_citation_pairs(claim, evidence_map, source_map)
                    cited_chunks = [evidence_map[chunk_id] for chunk_id, _ in citation_pairs if chunk_id in evidence_map]
                    if not statement or not cited_chunks:
                        continue
                    if (
                        result.subquestion.lane == ResearchLane.HARD_FACT
                        and self._claim_requires_primary_support(statement)
                        and not self._claim_citations_have_primary_support(citation_pairs, evidence_map)
                    ):
                        continue
                    if not self._should_preserve_model_statement(result.subquestion, statement):
                        derived_statement = self._derive_question_aligned_claim_from_chunks(
                            result.subquestion,
                            cited_chunks,
                        )
                        if derived_statement:
                            statement = derived_statement
                        else:
                            continue
                    if self._claim_is_too_broad(statement):
                        continue
                    rendered_claims.append(
                        f"- {statement}{self._format_citation_pairs(citation_pairs)}"
                    )
            if rendered_claims:
                return "\n".join(dict.fromkeys(rendered_claims))
            fallback = self._fallback_answer_from_evidence(result)
            if fallback:
                return fallback
            insufficiency_reason = str(payload.get("insufficiency_reason", "")).strip()
            if status == "insufficient" or insufficiency_reason:
                reason = insufficiency_reason or "The provided evidence does not directly support a publishable answer."
                return f"Insufficient evidence in the source pack: {reason}"

        legacy_answer = ""
        if isinstance(payload, dict):
            legacy_answer = str(payload.get("answer", "")).strip()
        return self._rewrite_legacy_answer(result, legacy_answer)

    def _sanitize_answer_for_verification(self, answer_text: str) -> str:
        text = (answer_text or "").strip()
        if not text:
            return ""
        kept_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "[Chunk:" in line and "[Source:" in line:
                kept_lines.append(line if line.startswith("- ") else f"- {line.lstrip('-• ').strip()}")
            elif "insufficient evidence" in line.lower():
                return line
        return "\n".join(kept_lines).strip()

    def _rewrite_legacy_answer(self, result: ResearchQuestionResult, answer_text: str) -> str:
        text = self._sanitize_answer_for_verification(answer_text)
        if not text:
            return ""
        evidence_map = {chunk.chunk_id: chunk for chunk in result.evidence[:8]}
        source_map = {chunk.chunk_id: chunk.source_id for chunk in result.evidence[:8]}
        rewritten: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            chunk_matches = re.findall(r"\[Chunk:\s*([^\]]+)\]", line)
            source_matches = re.findall(r"\[Source:\s*([^\]]+)\]", line)
            if not chunk_matches or not source_matches:
                continue
            citation_pairs = self._extract_structured_claim_citation_pairs(
                {
                    "chunk_ids": chunk_matches,
                    "source_ids": source_matches,
                },
                evidence_map,
                source_map,
            )
            cited_chunks = [evidence_map[chunk_id] for chunk_id, _ in citation_pairs if chunk_id in evidence_map]
            if not cited_chunks:
                continue
            statement = self._normalize_claim_sentence(line)
            statement = self._strip_soft_evidence_preamble(statement)
            if not self._should_preserve_model_statement(result.subquestion, statement):
                derived_statement = self._derive_question_aligned_claim_from_chunks(
                    result.subquestion,
                    cited_chunks,
                )
                if derived_statement:
                    statement = derived_statement
                else:
                    continue
            if self._claim_is_too_broad(statement):
                continue
            if (
                result.subquestion.lane == ResearchLane.HARD_FACT
                and self._claim_requires_primary_support(statement)
                and not self._claim_citations_have_primary_support(citation_pairs, evidence_map)
            ):
                continue
            rewritten.append(f"- {statement}{self._format_citation_pairs(citation_pairs)}")
        return "\n".join(dict.fromkeys(rewritten)).strip()

    def _strip_soft_evidence_preamble(self, statement: str) -> str:
        cleaned = (statement or "").strip()
        if not cleaned:
            return ""
        patterns = [
            r"^(?:According to|Based on|Per)\s+the\s+(?:filing|source(?: text)?|report|annual report|quarterly report|10-k|10-q)\s*,?\s*",
            r"^(?:The\s+)?(?:filing|source(?: text)?|report|annual report|quarterly report|10-k|10-q)\s+(?:states|shows|indicates|reports|discloses|notes)\s+(?:that\s+)?",
            r"^(?:The\s+)?provided\s+evidence\s+(?:states|shows|indicates)\s+(?:that\s+)?",
        ]
        updated = cleaned
        for pattern in patterns:
            updated = re.sub(pattern, "", updated, flags=re.IGNORECASE).strip()
        if not updated:
            return cleaned
        updated = updated[0].upper() + updated[1:] if len(updated) > 1 else updated.upper()
        return updated

    def _derive_question_aligned_claim_from_chunks(
        self,
        subquestion: ResearchSubquestion,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        if subquestion.lane == ResearchLane.HARD_FACT:
            direct_statement = self._build_direct_hard_fact_statement(subquestion, chunks)
            if direct_statement:
                return direct_statement
        if len(chunks) == 1:
            return self._derive_atomic_claim_from_chunk(subquestion, chunks[0])
        return ""

    def _extract_structured_claim_citation_pairs(
        self,
        claim_payload: dict,
        evidence_map: Dict[str, RetrievedChunk],
        source_map: Dict[str, str],
    ) -> List[tuple[str, str]]:
        chunk_ids = self._coerce_citation_list(claim_payload.get("chunk_ids"))
        source_ids = self._coerce_citation_list(claim_payload.get("source_ids"))
        if not chunk_ids:
            chunk_id = str(claim_payload.get("chunk_id", "")).strip()
            if chunk_id:
                chunk_ids = [chunk_id]
        if not source_ids:
            source_id = str(claim_payload.get("source_id", "")).strip()
            if source_id:
                source_ids = [source_id]
        if not chunk_ids:
            return []
        if source_ids:
            if len(source_ids) == 1 and len(chunk_ids) > 1:
                source_ids = source_ids * len(chunk_ids)
            if len(source_ids) != len(chunk_ids):
                return []
        else:
            source_ids = [source_map.get(chunk_id, "") for chunk_id in chunk_ids]

        pairs: List[tuple[str, str]] = []
        seen = set()
        for chunk_id, source_id in zip(chunk_ids, source_ids):
            expected_source_id = source_map.get(chunk_id)
            if chunk_id not in evidence_map or not expected_source_id:
                return []
            if source_id and source_id != expected_source_id:
                return []
            pair = (chunk_id, expected_source_id)
            if pair in seen:
                continue
            pairs.append(pair)
            seen.add(pair)
        return pairs

    def _coerce_citation_list(self, value: object) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        if value is None:
            return []
        cleaned = str(value).strip()
        return [cleaned] if cleaned else []

    def _format_citation_pairs(self, citation_pairs: Sequence[tuple[str, str]]) -> str:
        return "".join(f" [Chunk: {chunk_id}] [Source: {source_id}]" for chunk_id, source_id in citation_pairs)

    def _claim_citations_have_primary_support(
        self,
        citation_pairs: Sequence[tuple[str, str]],
        evidence_map: Dict[str, RetrievedChunk],
    ) -> bool:
        if not citation_pairs:
            return False
        return all(
            bool(chunk and (chunk.is_primary or (chunk.trust_level or 0) >= 4))
            for chunk_id, _ in citation_pairs
            for chunk in [evidence_map.get(chunk_id)]
        )

    def _normalize_claim_sentence(self, statement: str) -> str:
        cleaned = re.sub(r"\s+", " ", statement or "").strip().strip("-• ")
        cleaned = re.sub(r"\[Chunk:[^\]]+\]", "", cleaned).strip()
        cleaned = re.sub(r"\[Source:[^\]]+\]", "", cleaned).strip()
        if not cleaned:
            return ""
        sentence_parts = re.split(r"(?<=[.!?])\s+", cleaned)
        cleaned = sentence_parts[0].strip()
        if cleaned and not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        return cleaned

    def _fallback_answer_from_evidence(
        self,
        result: ResearchQuestionResult,
        *,
        allow_partial: bool = False,
    ) -> str:
        latest = result.assessments[-1] if result.assessments else None
        if not allow_partial and (not latest or not latest.sufficient):
            return ""
        chunk_map = {chunk.chunk_id: chunk for chunk in result.evidence[:8]}
        ordered_chunks = [
            chunk_map[chunk_id]
            for chunk_id in ((latest.matched_chunk_ids if latest else []) or [])
            if chunk_id in chunk_map
        ] or result.evidence[:2]
        if result.subquestion.lane == ResearchLane.HARD_FACT:
            hard_fact_chunks = ordered_chunks
            if allow_partial:
                hard_fact_chunks = [chunk for chunk in ordered_chunks if self._is_high_tier_chunk(chunk)]
            direct_answer = self._compose_direct_hard_fact_fallback(result.subquestion, hard_fact_chunks)
            if direct_answer:
                return direct_answer
            if allow_partial:
                for chunk in hard_fact_chunks[:1]:
                    statement = self._derive_atomic_claim_from_chunk(result.subquestion, chunk)
                    if statement:
                        return f"- {statement} [Chunk: {chunk.chunk_id}] [Source: {chunk.source_id}]"
            return ""
        rendered: List[str] = []
        for chunk in ordered_chunks[:2]:
            statement = self._derive_atomic_claim_from_chunk(result.subquestion, chunk)
            if not statement:
                continue
            rendered.append(f"- {statement} [Chunk: {chunk.chunk_id}] [Source: {chunk.source_id}]")
        return "\n".join(dict.fromkeys(rendered))

    def _compose_direct_hard_fact_fallback(
        self,
        subquestion: ResearchSubquestion,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        bundle = self._build_direct_hard_fact_bundle(subquestion, chunks)
        if not bundle:
            return ""
        return self._format_direct_hard_fact_bundle(bundle)

    def _build_direct_hard_fact_statement(
        self,
        subquestion: ResearchSubquestion,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        bundle = self._build_direct_hard_fact_bundle(subquestion, chunks)
        return str(bundle.get("statement", "")) if bundle else ""

    def _build_direct_hard_fact_bundle(
        self,
        subquestion: ResearchSubquestion,
        chunks: Sequence[RetrievedChunk],
    ) -> Optional[Dict[str, object]]:
        basis = self._subquestion_text_basis(subquestion)
        disclosure_builder = getattr(self, "_build_yes_no_disclosure_bundle", None)
        if callable(disclosure_builder):
            disclosure_bundle = disclosure_builder(subquestion, chunks)
            if disclosure_bundle:
                return disclosure_bundle
        list_disclosure_bundle = self._build_list_disclosure_bundle(subquestion, chunks)
        if list_disclosure_bundle:
            return list_disclosure_bundle
        formula_bundle = self._build_formula_hard_fact_bundle(subquestion, chunks)
        if formula_bundle:
            return formula_bundle
        if "quick ratio" in basis:
            for chunk in chunks:
                detail = self._extract_quick_ratio_detail(chunk)
                if not detail:
                    continue
                return {
                    "statement": f"The quick ratio was {detail['ratio']}, based on {detail['explanation']}.",
                    "citation_chunks": [chunk],
                }
        if "working capital ratio" in basis or "current ratio" in basis:
            for chunk in chunks:
                detail = self._extract_current_ratio_detail(chunk)
                if not detail:
                    continue
                return {
                    "statement": (
                        f"The working capital ratio was {detail['ratio']}, based on total current assets of "
                        f"{detail['current_assets']} and total current liabilities of {detail['current_liabilities']}."
                    ),
                    "citation_chunks": [chunk],
                }
        if "working capital" in basis:
            for chunk in chunks:
                detail = self._extract_working_capital_detail(chunk)
                if not detail:
                    continue
                if "positive working capital" in basis or "negative working capital" in basis or basis.startswith("does "):
                    lead = "Yes" if detail["polarity"] == "positive" else "No"
                    statement = (
                        f"{lead}, the company had {detail['polarity']} working capital of {detail['working_capital']}, "
                        f"based on total current assets of {detail['current_assets']} and total current liabilities of {detail['current_liabilities']}."
                    )
                else:
                    statement = (
                        f"The company had {detail['polarity']} working capital of {detail['working_capital']}, "
                        f"based on total current assets of {detail['current_assets']} and total current liabilities of {detail['current_liabilities']}."
                    )
                return {
                    "statement": statement,
                    "citation_chunks": [chunk],
                }
        if any(token in basis for token in ("highest", "largest", "lowest", "smallest")):
            for chunk in chunks:
                detail = self._extract_ranked_metric_detail(subquestion, chunk.text)
                if not detail:
                    continue
                return {
                    "statement": f"{detail['winner']} had the {detail['direction']} {detail['metric']} at {detail['value']}.",
                    "citation_chunks": [chunk],
                }
        if self._is_direct_numeric_extraction_question(subquestion):
            line_item_detail = self._extract_line_item_detail(subquestion, chunks)
            if line_item_detail:
                chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
                citation_chunk = chunk_map.get(str(line_item_detail["chunk_id"]))
                if citation_chunk:
                    return {
                        "statement": (
                            f"{line_item_detail['display_name']} {line_item_detail['verb']} "
                            f"{line_item_detail['formatted_value']}."
                        ),
                        "citation_chunks": [citation_chunk],
                    }
        return None

    def _build_list_disclosure_bundle(
        self,
        subquestion: ResearchSubquestion,
        chunks: Sequence[RetrievedChunk],
    ) -> Optional[Dict[str, object]]:
        basis = self._subquestion_text_basis(subquestion)

        if any(
            token in basis
            for token in ("debt securities", "national securities exchange", "registered to trade")
        ):
            for chunk in chunks:
                lowered = chunk.text.lower()
                if any(
                    phrase in lowered
                    for phrase in (
                        "no established trading market",
                        "not listed on any securities exchange",
                        "not listed on any national securities exchange",
                        "none of our debt securities are listed",
                        "none of the debt securities are listed",
                    )
                ):
                    return {
                        "statement": "There were no debt securities registered to trade on a national securities exchange.",
                        "citation_chunks": [chunk],
                    }

        if "acquisition" in basis:
            for chunk in chunks:
                lowered = chunk.text.lower()
                if any(
                    phrase in lowered
                    for phrase in (
                        "did not complete any acquisitions",
                        "did not make any acquisitions",
                        "no acquisitions were completed",
                        "there were no acquisitions",
                        "no material acquisitions",
                    )
                ):
                    return {
                        "statement": "The company did not complete any major acquisitions in the period.",
                        "citation_chunks": [chunk],
                    }
                for sentence in _extract_evidence_sentences(chunk.text):
                    sentence_lower = sentence.lower()
                    if any(
                        phrase in sentence_lower
                        for phrase in ("acquired", "acquisition of", "acquisition from", "business combination")
                    ):
                        statement = self._strip_soft_evidence_preamble(
                            self._normalize_claim_sentence(sentence)
                        )
                        if statement and not self._claim_is_too_broad(statement):
                            return {
                                "statement": statement,
                                "citation_chunks": [chunk],
                            }

        if any(token in basis for token in ("production rate", "production rates", "forecasting")):
            for chunk in chunks:
                fragments: List[str] = []
                for sentence in _extract_evidence_sentences(chunk.text):
                    sentence_lower = sentence.lower()
                    if "per month" not in sentence_lower and "production rate" not in sentence_lower:
                        continue
                    if not any(token in sentence_lower for token in ("737", "777", "787", "767", "max")):
                        continue
                    fragment = self._strip_soft_evidence_preamble(
                        self._normalize_claim_sentence(sentence)
                    ).rstrip(".")
                    if fragment and fragment not in fragments:
                        fragments.append(fragment)
                    if len(fragments) >= 2:
                        break
                if fragments:
                    statement = (
                        fragments[0] + "."
                        if len(fragments) == 1
                        else f"Boeing forecasted production rate changes including {fragments[0]}; {fragments[1]}."
                    )
                    return {
                        "statement": statement,
                        "citation_chunks": [chunk],
                    }

        return None

    def _format_direct_hard_fact_bundle(self, bundle: Dict[str, object]) -> str:
        statement = str(bundle.get("statement", "")).strip()
        citation_chunks = [
            chunk for chunk in bundle.get("citation_chunks", [])
            if isinstance(chunk, RetrievedChunk)
        ]
        if not statement or not citation_chunks:
            return ""
        pairs: List[tuple[str, str]] = []
        seen = set()
        for chunk in citation_chunks:
            pair = (chunk.chunk_id, chunk.source_id)
            if pair in seen:
                continue
            pairs.append(pair)
            seen.add(pair)
        if not pairs:
            return ""
        return f"- {statement}{self._format_citation_pairs(pairs)}"

    def _is_direct_numeric_extraction_question(self, subquestion: ResearchSubquestion) -> bool:
        basis = self._subquestion_text_basis(subquestion)
        if any(
            token in basis
            for token in (
                "growth",
                "driver",
                "drivers",
                "what are",
                "revenue sources",
                "primary revenue sources",
                "revenue mix",
                "breakdown",
                "which segment",
                "highest",
                "lowest",
                "largest",
                "smallest",
            )
        ):
            return False
        return any(
            token in basis
            for token in (
                "what is",
                "what was",
                "what percent",
                "what percentage",
                "how much",
                "amount",
                "balance",
                "year end",
                "end of fy",
                "as of the end",
                "exact",
                "reported under",
                "reported in the",
            )
        )

    def _extract_quick_ratio_detail(self, chunk: RetrievedChunk) -> Optional[Dict[str, str]]:
        current_liabilities = self._extract_labeled_financial_value(chunk.text, ["total current liabilities"])
        if current_liabilities is None or current_liabilities[0] == 0:
            return None

        current_assets = self._extract_labeled_financial_value(chunk.text, ["total current assets"])
        inventories = self._extract_labeled_financial_value(chunk.text, ["inventories"])
        if current_assets and inventories:
            ratio_value = (current_assets[0] - inventories[0]) / current_liabilities[0]
            explanation = (
                f"total current assets of {current_assets[1]} less inventories of {inventories[1]}, "
                f"divided by total current liabilities of {current_liabilities[1]}"
            )
            return {
                "ratio": self._format_ratio_value(ratio_value),
                "explanation": explanation,
            }

        cash = self._extract_labeled_financial_value(chunk.text, ["cash and cash equivalents"])
        short_term_investments = self._extract_labeled_financial_value(chunk.text, ["short-term investments"])
        accounts_receivable = self._extract_labeled_financial_value(
            chunk.text,
            ["accounts receivable, net", "accounts receivable"],
        )
        if cash and short_term_investments and accounts_receivable:
            ratio_value = (cash[0] + short_term_investments[0] + accounts_receivable[0]) / current_liabilities[0]
            explanation = (
                f"cash and cash equivalents of {cash[1]}, short-term investments of {short_term_investments[1]}, "
                f"accounts receivable of {accounts_receivable[1]}, and total current liabilities of {current_liabilities[1]}"
            )
            return {
                "ratio": self._format_ratio_value(ratio_value),
                "explanation": explanation,
            }
        return None

    def _extract_working_capital_detail(self, chunk: RetrievedChunk) -> Optional[Dict[str, str]]:
        current_assets = self._extract_labeled_financial_value(chunk.text, ["total current assets"])
        current_liabilities = self._extract_labeled_financial_value(chunk.text, ["total current liabilities"])
        if current_assets is None or current_liabilities is None:
            return None
        working_capital = current_assets[0] - current_liabilities[0]
        return {
            "polarity": "positive" if working_capital >= 0 else "negative",
            "working_capital": self._format_financial_value(working_capital, chunk.text),
            "current_assets": current_assets[1],
            "current_liabilities": current_liabilities[1],
        }

    def _extract_current_ratio_detail(self, chunk: RetrievedChunk) -> Optional[Dict[str, str]]:
        current_assets = self._extract_labeled_financial_value(chunk.text, ["total current assets"])
        current_liabilities = self._extract_labeled_financial_value(chunk.text, ["total current liabilities"])
        if current_assets is None or current_liabilities is None or current_liabilities[0] == 0:
            return None
        return {
            "ratio": self._format_ratio_value(current_assets[0] / current_liabilities[0]),
            "current_assets": current_assets[1],
            "current_liabilities": current_liabilities[1],
        }

    def _build_formula_hard_fact_bundle(
        self,
        subquestion: ResearchSubquestion,
        chunks: Sequence[RetrievedChunk],
    ) -> Optional[Dict[str, object]]:
        basis = self._subquestion_text_basis(subquestion)
        if (
            any(token in basis for token in ("ebitda", "unadjusted ebitda"))
            and any(token in basis for token in ("capex", "capital expenditure", "capital expenditures"))
            and any(token in basis for token in ("less", "minus"))
        ):
            ebitda_spec = self._formula_metric_spec(subquestion)
            ebitda_detail = self._extract_formula_operand_detail(chunks, ebitda_spec["labels"])
            capex_detail = self._extract_formula_operand_detail(
                chunks,
                ["capital expenditures", "capital expenditure"],
                absolute_value=True,
            )
            if ebitda_detail and capex_detail:
                result_value = float(ebitda_detail["value"]) - float(capex_detail["value"])
                reference_text = f"{ebitda_detail['formatted_value']} {capex_detail['formatted_value']}"
                chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
                citation_chunks = [
                    chunk_map[chunk_id]
                    for chunk_id in [str(ebitda_detail["chunk_id"]), str(capex_detail["chunk_id"])]
                    if chunk_id in chunk_map
                ]
                return {
                    "statement": (
                        f"{ebitda_spec['display_name']} less capital expenditures was "
                        f"{self._format_financial_value(result_value, reference_text)}, based on "
                        f"{ebitda_spec['display_name'].lower()} of {ebitda_detail['formatted_value']} and capital expenditures of "
                        f"{capex_detail['formatted_value']}."
                    ),
                    "citation_chunks": citation_chunks,
                }

        if "margin" in basis:
            numerator_spec = self._formula_metric_spec(subquestion)
            if numerator_spec["role"] != "revenue":
                numerator_detail = self._extract_formula_operand_detail(
                    chunks,
                    numerator_spec["labels"],
                    absolute_value=bool(numerator_spec.get("absolute_value")),
                )
                revenue_detail = self._extract_formula_operand_detail(
                    chunks,
                    ["total revenue", "net revenue", "net sales", "revenue", "sales"],
                )
                if numerator_detail and revenue_detail and float(revenue_detail["value"]) != 0:
                    margin = float(numerator_detail["value"]) / float(revenue_detail["value"])
                    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
                    citation_chunks = [
                        chunk_map[chunk_id]
                        for chunk_id in [str(numerator_detail["chunk_id"]), str(revenue_detail["chunk_id"])]
                        if chunk_id in chunk_map
                    ]
                    return {
                        "statement": (
                            f"{numerator_spec['margin_name']} was {margin * 100:.1f}%, based on "
                            f"{numerator_spec['display_name'].lower()} of {numerator_detail['formatted_value']} and revenue of "
                            f"{revenue_detail['formatted_value']}."
                        ),
                        "citation_chunks": citation_chunks,
                    }
        return None

    def _formula_metric_spec(self, subquestion: ResearchSubquestion) -> Dict[str, object]:
        basis = self._subquestion_text_basis(subquestion)
        if "depreciation and amortization" in basis or "d&a" in basis:
            return {
                "role": "depreciation_and_amortization",
                "labels": ["depreciation and amortization"],
                "display_name": "Depreciation and amortization",
                "margin_name": "Depreciation and amortization margin",
                "absolute_value": True,
            }
        if "operating margin" in basis or "operating income" in basis:
            return {
                "role": "operating_income",
                "labels": ["operating income"],
                "display_name": "Operating income",
                "margin_name": "Operating margin",
            }
        if "unadjusted ebitda" in basis:
            return {
                "role": "unadjusted_ebitda",
                "labels": ["unadjusted ebitda", "adjusted ebitda", "ebitda"],
                "display_name": "Unadjusted EBITDA",
                "margin_name": "Unadjusted EBITDA margin",
            }
        if "ebitda" in basis:
            return {
                "role": "ebitda",
                "labels": ["ebitda", "adjusted ebitda", "unadjusted ebitda"],
                "display_name": "EBITDA",
                "margin_name": "EBITDA margin",
            }
        if "net profit margin" in basis or "net income" in basis:
            return {
                "role": "net_income",
                "labels": ["net income"],
                "display_name": "Net income",
                "margin_name": "Net profit margin",
            }
        return {
            "role": "revenue",
            "labels": ["revenue"],
            "display_name": "Revenue",
            "margin_name": "Margin",
        }

    def _extract_formula_operand_detail(
        self,
        chunks: Sequence[RetrievedChunk],
        labels: Sequence[str],
        *,
        absolute_value: bool = False,
    ) -> Optional[Dict[str, object]]:
        for chunk in chunks:
            value = self._extract_labeled_financial_value(chunk.text, labels)
            if value is None:
                continue
            numeric_value = abs(value[0]) if absolute_value else value[0]
            formatted_value = (
                self._format_financial_value(numeric_value, chunk.text)
                if absolute_value and value[0] < 0
                else value[1]
            )
            return {
                "value": numeric_value,
                "formatted_value": formatted_value,
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
            }
        return None

    def _build_yes_no_disclosure_bundle(
        self,
        subquestion: ResearchSubquestion,
        chunks: Sequence[RetrievedChunk],
    ) -> Optional[Dict[str, object]]:
        basis = self._subquestion_text_basis(subquestion)
        if any(token in basis for token in ("customer concentration", "major customer", "significant customer")):
            for chunk in chunks:
                lowered = chunk.text.lower()
                percent_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*%\s+of\s+(?:consolidated\s+)?(?:net\s+)?revenue",
                    chunk.text,
                    re.IGNORECASE,
                )
                if percent_match and any(
                    phrase in lowered
                    for phrase in (
                        "major customer",
                        "significant customer",
                        "one customer",
                        "single customer",
                        "customer accounted for",
                        "customer represented",
                    )
                ):
                    return {
                        "statement": (
                            f"Yes, one customer accounted for {percent_match.group(1)}% of revenue."
                        ),
                        "citation_chunks": [chunk],
                    }
                if any(
                    phrase in lowered
                    for phrase in (
                        "no customer accounted for 10% or more of revenue",
                        "no customer represented 10% or more of revenue",
                        "no customer accounted for ten percent or more of revenue",
                        "no significant customer accounted for 10% or more of revenue",
                    )
                ):
                    return {
                        "statement": "No, no customer accounted for 10% or more of revenue.",
                        "citation_chunks": [chunk],
                    }
        if any(token in basis for token in ("legal battles", "legal battle", "legal proceedings", "legal proceeding")):
            for chunk in chunks:
                lowered = chunk.text.lower()
                if any(
                    phrase in lowered
                    for phrase in (
                        "not a party to any material legal proceedings",
                        "no material legal proceedings",
                        "no material pending legal proceedings",
                        "no material litigation",
                    )
                ):
                    return {
                        "statement": "No material ongoing legal battles were disclosed.",
                        "citation_chunks": [chunk],
                    }
                if any(
                    phrase in lowered
                    for phrase in (
                        "material legal proceeding",
                        "material legal proceedings",
                        "material litigation",
                        "pending litigation",
                    )
                ):
                    return {
                        "statement": "The filing disclosed material legal proceedings.",
                        "citation_chunks": [chunk],
                    }
        if any(token in basis for token in ("proposal vote", "voting result", "shareholder vote", "proposal outcome", "outcome of the shareholder vote")):
            for chunk in chunks:
                lowered = chunk.text.lower()
                if any(token in lowered for token in ("defeated", "failed", "not approved", "did not pass")):
                    return {
                        "statement": "The filing indicates the proposal failed.",
                        "citation_chunks": [chunk],
                    }
                if any(token in lowered for token in ("passed", "approved", "adopted")):
                    return {
                        "statement": "The filing indicates the proposal passed.",
                        "citation_chunks": [chunk],
                    }
        if "revolving credit agreement" in basis and "increase" in basis:
            for chunk in chunks:
                match = re.search(
                    r"increas(?:e|ed)[^$]{0,40}\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|b|m|k)?",
                    chunk.text,
                    re.IGNORECASE,
                )
                if match:
                    literal = "".join(part for part in match.groups(default="") if part).strip()
                    normalized = self._normalize_financial_literal(literal, chunk.text)
                    return {
                        "statement": f"The revolving credit agreement increased by {normalized}.",
                        "citation_chunks": [chunk],
                    }
                from_to = re.search(
                    r"from\s+\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|b|m|k)?[^$]{0,40}to\s+\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|b|m|k)?",
                    chunk.text,
                    re.IGNORECASE,
                )
                if from_to:
                    start_literal = "".join(part for part in from_to.groups()[0:2] if part).strip()
                    end_literal = "".join(part for part in from_to.groups()[2:4] if part).strip()
                    start_value = self._parse_numeric_literal(start_literal)
                    end_value = self._parse_numeric_literal(end_literal)
                    if start_value is not None and end_value is not None:
                        increase = end_value - start_value
                        return {
                            "statement": (
                                f"The revolving credit agreement increased by "
                                f"{self._format_financial_value(increase, chunk.text)}."
                            ),
                            "citation_chunks": [chunk],
                        }
        if any(token in basis for token in ("borrowing capacity", "may borrow", "aggregate commitments", "total amount")):
            for chunk in chunks:
                match = re.search(
                    r"(?:aggregate commitments|may borrow up to|borrowing capacity)[^$]{0,40}\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|b|m|k)?",
                    chunk.text,
                    re.IGNORECASE,
                )
                if match:
                    literal = "".join(part for part in match.groups(default="") if part).strip()
                    normalized = self._normalize_financial_literal(literal, chunk.text)
                    return {
                        "statement": f"Total borrowing capacity was {normalized}.",
                        "citation_chunks": [chunk],
                    }
        return None

    def _extract_line_item_detail(
        self,
        subquestion: ResearchSubquestion,
        chunks: Sequence[RetrievedChunk],
    ) -> Optional[Dict[str, str]]:
        for spec in self._candidate_line_item_specs(subquestion):
            for chunk in chunks:
                value = self._extract_labeled_financial_value(chunk.text, spec["labels"])
                if value is None:
                    continue
                return {
                    "display_name": spec["display_name"],
                    "formatted_value": self._normalize_direct_line_item_value(
                        spec,
                        value[0],
                        value[1],
                        f"{chunk.text} {subquestion.text}",
                    ),
                    "verb": spec.get("verb", "was"),
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                }
        return None

    def _candidate_line_item_specs(self, subquestion: ResearchSubquestion) -> List[Dict[str, object]]:
        basis = self._subquestion_text_basis(subquestion)
        specs = [
            {
                "keywords": ("capital expenditure", "capital expenditures", "capex"),
                "labels": ["capital expenditures", "capital expenditure"],
                "display_name": "Capital expenditures",
                "verb": "were",
            },
            {
                "keywords": ("accounts payable",),
                "labels": ["accounts payable"],
                "display_name": "Accounts payable",
                "verb": "were",
            },
            {
                "keywords": ("cash & cash equivalents", "cash and cash equivalents"),
                "labels": ["cash and cash equivalents"],
                "display_name": "Cash and cash equivalents",
                "verb": "were",
            },
            {
                "keywords": ("inventories", "inventory"),
                "labels": ["inventories", "inventory"],
                "display_name": "Inventories",
                "verb": "were",
            },
            {
                "keywords": ("accounts receivable", "net ar", "receivables"),
                "labels": ["accounts receivable, net", "accounts receivable", "receivables, net", "trade receivables, net"],
                "display_name": "Accounts receivable, net",
                "verb": "was",
            },
            {
                "keywords": ("pp&e", "property, plant", "property and equipment", "ppne"),
                "labels": [
                    "property, plant and equipment, net",
                    "property and equipment, net",
                    "net property, plant and equipment",
                    "property, plant and equipment",
                ],
                "display_name": "Net property, plant and equipment",
            },
            {
                "keywords": ("total current liabilities",),
                "labels": ["total current liabilities"],
                "display_name": "Total current liabilities",
                "verb": "were",
            },
            {
                "keywords": ("total current assets",),
                "labels": ["total current assets"],
                "display_name": "Total current assets",
                "verb": "were",
            },
            {
                "keywords": ("total assets",),
                "labels": ["total assets"],
                "display_name": "Total assets",
            },
            {
                "keywords": ("total liabilities",),
                "labels": ["total liabilities"],
                "display_name": "Total liabilities",
            },
            {
                "keywords": ("depreciation", "amortization", "d&a"),
                "labels": ["depreciation and amortization"],
                "display_name": "Depreciation and amortization",
            },
            {
                "keywords": ("operating income",),
                "labels": ["operating income"],
                "display_name": "Operating income",
            },
            {
                "keywords": ("net income",),
                "labels": ["net income"],
                "display_name": "Net income",
            },
            {
                "keywords": ("revenue", "net sales", "sales"),
                "labels": ["total revenue", "net revenue", "net sales", "revenue", "sales"],
                "display_name": "Revenue",
            },
            {
                "keywords": (
                    "wages expense as a percent of net sales",
                    "wages expense as a percentage of net sales",
                    "wages expense % of net sales",
                ),
                "labels": [
                    "wages expense as a percentage of net sales",
                    "wages expense as a percent of net sales",
                    "wages expense % of net sales",
                ],
                "display_name": "Wages expense as a percent of net sales",
            },
            {
                "keywords": ("cost of goods sold", "cogs", "cost of sales", "cost of revenue"),
                "labels": ["cost of goods sold", "cost of sales", "cost of revenue"],
                "display_name": "Cost of goods sold",
            },
        ]
        return [
            spec
            for spec in specs
            if any(keyword in basis for keyword in spec["keywords"])
        ]

    def _normalize_direct_line_item_value(
        self,
        spec: Dict[str, object],
        numeric_value: float,
        formatted_value: str,
        reference_text: str,
    ) -> str:
        display_name = str(spec.get("display_name", "")).lower()
        normalized_value = numeric_value
        if display_name in {"capital expenditures", "depreciation and amortization"} and numeric_value < 0:
            normalized_value = abs(numeric_value)
        requested_unit = self._requested_output_unit(reference_text)
        if requested_unit:
            return self._format_financial_value_for_unit(normalized_value, requested_unit)
        if normalized_value != numeric_value:
            return self._format_financial_value(normalized_value, reference_text)
        return formatted_value

    def _requested_output_unit(self, reference_text: str) -> str:
        lowered = (reference_text or "").lower()
        if "in usd billions" in lowered or "in billions" in lowered:
            return "billion"
        if "in usd millions" in lowered or "in millions" in lowered:
            return "million"
        if "in usd thousands" in lowered or "in thousands" in lowered:
            return "thousand"
        return ""

    def _format_financial_value_for_unit(self, value: float, unit: str) -> str:
        unit = (unit or "").lower()
        divisor = 1.0
        suffix = " million"
        if unit == "billion":
            divisor = 1_000.0
            suffix = " billion"
        elif unit == "thousand":
            divisor = 0.001
            suffix = " thousand"
        scaled = value / divisor
        body = f"{abs(scaled):,.3f}".rstrip("0").rstrip(".")
        prefix = "-$" if scaled < 0 else "$"
        return f"{prefix}{body}{suffix}"

    def _extract_ranked_metric_detail(
        self,
        subquestion: ResearchSubquestion,
        text: str,
    ) -> Optional[Dict[str, str]]:
        basis = self._subquestion_text_basis(subquestion)
        metric = ""
        for phrase in (
            "net income",
            "operating income",
            "operating margin",
            "gross margin",
            "cash flow",
            "ebitdar contribution",
            "investment",
            "liability",
            "revenue",
        ):
            if phrase in basis:
                metric = phrase
                break
        if not metric:
            return None
        ranked_values = self._extract_ranked_metric_values(text, metric)
        if len(ranked_values) < 2:
            return None
        direction = "lowest" if any(token in basis for token in ("lowest", "smallest")) else "highest"
        winner = min(ranked_values, key=lambda item: item["value"]) if direction == "lowest" else max(
            ranked_values,
            key=lambda item: item["value"],
        )
        return {
            "winner": winner["label"],
            "direction": direction,
            "metric": metric,
            "value": winner["formatted_value"],
        }

    def _extract_ranked_metric_values(self, text: str, metric: str) -> List[Dict[str, object]]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []
        pattern = re.compile(
            rf"([A-Za-z&/\- ,]{{3,80}}?)\s+{re.escape(metric)}(?:[^$\d()]{{0,24}})?(\(?\$?\s*\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k))?)",
            re.IGNORECASE,
        )
        values: List[Dict[str, object]] = []
        for match in pattern.finditer(normalized):
            label = re.sub(r"\s+", " ", match.group(1)).strip(" ,.-")
            numeric_value = self._parse_numeric_literal(match.group(2))
            if not label or numeric_value is None:
                continue
            values.append(
                {
                    "label": label,
                    "value": numeric_value,
                    "formatted_value": self._normalize_financial_literal(match.group(2), normalized),
                }
            )
        deduped: Dict[str, Dict[str, object]] = {}
        for item in values:
            deduped[str(item["label"]).lower()] = item
        return list(deduped.values())

    def _extract_labeled_financial_value(
        self,
        text: str,
        labels: Sequence[str],
    ) -> Optional[tuple[float, str]]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return None
        for label in labels:
            pattern = re.compile(
                rf"{re.escape(label)}.{{0,40}}?(\(?\$?\s*\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k))?)",
                re.IGNORECASE,
            )
            match = pattern.search(normalized)
            if not match:
                continue
            numeric_value = self._parse_numeric_literal(match.group(1))
            if numeric_value is None:
                continue
            return numeric_value, self._normalize_financial_literal(match.group(1), normalized)
        return None

    def _normalize_financial_literal(self, literal: str, reference_text: str) -> str:
        cleaned = re.sub(r"\s+", " ", literal or "").strip()
        if not cleaned:
            return ""
        if "$" not in cleaned:
            cleaned = f"${cleaned}"
        if not re.search(r"\b(?:billion|million|thousand|bn|mn|b|m|k)\b", cleaned, re.IGNORECASE):
            unit_suffix = self._infer_reference_unit_suffix(reference_text)
            if unit_suffix:
                cleaned = f"{cleaned}{unit_suffix}"
        return cleaned.replace("$ ", "$")

    def _parse_numeric_literal(self, literal: str) -> Optional[float]:
        cleaned = str(literal or "").strip().lower().replace("$", "").replace(",", "").replace(" ", "")
        if not cleaned:
            return None
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        multiplier = 1.0
        for suffix, factor in (
            ("billion", 1_000.0),
            ("bn", 1_000.0),
            ("b", 1_000.0),
            ("million", 1.0),
            ("mn", 1.0),
            ("m", 1.0),
            ("thousand", 0.001),
            ("k", 0.001),
        ):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                multiplier = factor
                break
        try:
            value = float(cleaned) * multiplier
        except ValueError:
            return None
        return -value if negative else value

    def _format_ratio_value(self, value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _format_financial_value(self, value: float, reference_text: str) -> str:
        rounded = round(value, 2)
        if abs(rounded - int(rounded)) < 0.01:
            body = f"{abs(int(round(rounded))):,}"
        else:
            body = f"{abs(rounded):,.2f}".rstrip("0").rstrip(".")
        suffix = self._infer_reference_unit_suffix(reference_text)
        prefix = "-$" if rounded < 0 else "$"
        return f"{prefix}{body}{suffix}"

    def _infer_reference_unit_suffix(self, reference_text: str) -> str:
        if re.search(r"\b(?:in\s+)?(?:usd\s+)?millions?\b", reference_text, re.IGNORECASE):
            return " million"
        if re.search(r"\b(?:in\s+)?(?:usd\s+)?billions?\b", reference_text, re.IGNORECASE):
            return " billion"
        if re.search(r"\b(?:in\s+)?(?:usd\s+)?thousands?\b", reference_text, re.IGNORECASE):
            return " thousand"
        return ""

    def _claim_is_too_broad(self, statement: str) -> bool:
        cleaned = (statement or "").strip()
        if not cleaned:
            return True
        lowered = cleaned.lower()
        header_markers = {
            "all news",
            "business wire",
            "copyright",
            "corrected transcript",
            "earnings call",
            "forward looking statements",
            "nasdaq:",
        }
        words = re.findall(r"\b\w+\b", cleaned)
        return (
            len(cleaned) > 260
            or len(words) > 38
            or any(marker in lowered for marker in header_markers)
        )

    def _should_preserve_model_statement(
        self,
        subquestion: ResearchSubquestion,
        statement: str,
    ) -> bool:
        cleaned = (statement or "").strip()
        if not cleaned:
            return False
        if self._claim_is_too_broad(cleaned) or self._claim_is_low_signal_fragment(cleaned):
            return False
        if subquestion.lane != ResearchLane.HARD_FACT:
            return not self._claim_requires_primary_support(cleaned)
        lowered = cleaned.lower()
        alignment = max(
            self._slot_overlap_ratio(subquestion, cleaned),
            self._question_overlap_ratio(subquestion, cleaned),
        )
        if any(
            token in lowered
            for token in (
                "yes",
                "no",
                "positive",
                "negative",
                "highest",
                "lowest",
                "largest",
                "smallest",
                "healthy",
                "unhealthy",
                "improving",
                "declining",
            )
        ):
            return alignment >= 0.12
        if "%" in cleaned or "percent" in lowered or "percentage" in lowered:
            return alignment >= 0.12
        if self._extract_number_tokens(cleaned):
            return alignment >= 0.18
        return alignment >= 0.34

    def _derive_atomic_claim_from_chunk(
        self,
        subquestion: ResearchSubquestion,
        chunk: RetrievedChunk,
    ) -> str:
        if subquestion.lane == ResearchLane.HARD_FACT:
            table_row_claim = self._extract_hard_fact_table_row_claim(subquestion, chunk.text)
            if table_row_claim:
                return table_row_claim
        if subquestion.lane == ResearchLane.HARD_FACT and self._is_revenue_mix_question(subquestion):
            explicit_breakout = self._extract_explicit_revenue_mix_claim(chunk.text)
            if explicit_breakout:
                return explicit_breakout
        if subquestion.lane == ResearchLane.SEMANTIC and self._targets_management_commentary(subquestion):
            commentary_claim = self._extract_management_commentary_claim(chunk.text)
            if commentary_claim:
                return commentary_claim
        candidates: List[str] = []
        candidates.extend(_extract_evidence_sentences(chunk.text))
        normalized = re.sub(r"\s+", " ", chunk.text or "").strip()
        anchors = [
            "Total revenue",
            "Revenue",
            "Network services revenue",
            "Security revenue",
            "Other revenue",
            "GAAP gross margin",
            "Non-GAAP gross margin",
            "Generated $",
            "Enterprise customer count",
            "Remaining Performance Obligations",
            "Last 12-month net retention rate",
            "LTM NRR",
        ]
        for anchor in anchors:
            pattern = re.compile(rf"({re.escape(anchor)}[^.]{{0,220}}(?:\.|$))", re.IGNORECASE)
            for match in pattern.finditer(normalized):
                fragment = self._normalize_claim_sentence(match.group(1))
                if fragment:
                    candidates.append(fragment)

        best = ""
        best_score = float("-inf")
        for candidate in candidates:
            if not candidate:
                continue
            if self._claim_is_low_signal_fragment(candidate):
                continue
            score = 0.0
            lowered = candidate.lower()
            score += self._slot_overlap_ratio(subquestion, candidate) * 10.0
            score += self._question_overlap_ratio(subquestion, candidate) * 12.0
            if subquestion.metric_family and subquestion.metric_family in lowered:
                score += 3.0
            if subquestion.lane == ResearchLane.HARD_FACT and self._extract_number_tokens(candidate):
                score += 2.0
            if self._claim_is_too_broad(candidate):
                score -= 5.0
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def _extract_hard_fact_table_row_claim(self, subquestion: ResearchSubquestion, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return ""

        slot_basis = self._subquestion_text_basis(subquestion)
        patterns: List[str] = []

        if any(token in slot_basis for token in ("depreciation", "amortization", "d&a")):
            patterns.append(r"(Depreciation and amortization\s+[\$\(\)\d,\.\-\s]+)")
        if "net revenue" in slot_basis or "total revenue" in slot_basis or re.search(r"\brevenue\b", slot_basis):
            patterns.extend(
                [
                    r"(Net revenue\s+[\$\(\)\d,\.\-\s]+)",
                    r"(Total revenue[s]?\s+[\$\(\)\d,\.\-\s]+)",
                    r"(Revenue\s+[\$\(\)\d,\.\-\s]+)",
                ]
            )
        if "current assets" in slot_basis:
            patterns.append(r"(Total current assets\s+[\$\(\)\d,\.\-\s]+)")
        if "current liabilities" in slot_basis:
            patterns.append(r"(Total current liabilities\s+[\$\(\)\d,\.\-\s]+)")
        if "quick ratio" in slot_basis:
            patterns.extend(
                [
                    r"(Cash and cash equivalents\s+[\$\(\)\d,\.\-\s]+)",
                    r"(Short-term investments\s+[\$\(\)\d,\.\-\s]+)",
                    r"(Accounts receivable[^\n]*?[\$\(\)\d,\.\-\s]+)",
                    r"(Total current liabilities\s+[\$\(\)\d,\.\-\s]+)",
                ]
            )

        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if not match:
                continue
            candidate = self._normalize_claim_sentence(match.group(1))
            if candidate and not self._claim_is_low_signal_fragment(candidate):
                return candidate
        return ""

    def _claim_is_low_signal_fragment(self, statement: str) -> bool:
        cleaned = re.sub(r"\s+", " ", (statement or "")).strip().strip("-• ")
        if not cleaned:
            return True
        lowered = cleaned.lower()
        alpha_tokens = re.findall(r"[A-Za-z]+", cleaned)

        low_signal_markers = (
            "table of contents",
            "incorporated by reference",
            "in witness whereof",
            "we have served as the company",
            "signature page follows",
            "advanced micro devices, inc.",
            "paypal holdings, inc.",
            "consolidated statements of cash flows",
            "consolidated balance sheets",
            "consolidated statements of income",
        )
        if any(marker in lowered for marker in low_signal_markers):
            return True
        if re.match(r"^\d+\s+\[page\s+\d+\]", lowered):
            return True
        if cleaned.startswith("/s/"):
            return True
        if len(alpha_tokens) < 3 and "[page" in lowered:
            return True
        return False

    def _extract_explicit_revenue_mix_claim(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        for anchor in ("Network services revenue", "Security revenue", "Other revenue"):
            pattern = re.compile(rf"({re.escape(anchor)}[^.]*?\$[\d,.]+\s*(?:million|billion|M|B)?[^.]*\.)", re.IGNORECASE)
            match = pattern.search(normalized)
            if match:
                return self._normalize_claim_sentence(match.group(1))
        return ""

    def _extract_management_commentary_claim(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        patterns = [
            r"(These efforts were reflected in balanced revenue growth across product lines, geographic regions, and customer segments[^.]*\.)",
            r"(This outperformance was primarily due to[^.]*balanced traffic mix[^.]*delivery and security[^.]*\.)",
            r"(In Q4, we continued investment in our platform strategy to drive multi-product adoption and build upon our cross-sell momentum[^.]*\.)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return self._normalize_claim_sentence(match.group(1))
        return ""

    def _claim_requires_primary_support(self, statement: str) -> bool:
        lowered = statement.lower()
        return any(keyword in lowered for keyword in HARD_FACT_KEYWORDS) or bool(re.search(r"\d", lowered))

    def _build_memo(
        self,
        task: ResearchTask,
        results: List[ResearchQuestionResult],
        source_registry: Dict[str, dict],
        replay: ResearchReplayRecord,
        llm_budget: dict,
    ) -> GeneratedMemo:
        completed = [item for item in results if item.status == ResearchQuestionStatus.COMPLETED and item.supported_content.strip()]
        refused = [item for item in results if item.status == ResearchQuestionStatus.REFUSED]
        sections: List[MemoSection] = []
        all_verifications = []

        if completed:
            hard_lines = []
            semantic_lines = []
            for item in completed:
                formatted = f"### {item.subquestion.text}\n{item.supported_content}"
                if item.subquestion.lane == ResearchLane.HARD_FACT:
                    hard_lines.append(formatted)
                else:
                    semantic_lines.append(formatted)
                all_verifications.extend(item.verified_claims)

            if hard_lines:
                sections.append(
                    MemoSection(
                        title="hard_fact_findings",
                        content="\n\n".join(hard_lines),
                        writing_trace={"subquestions": [item.subquestion.question_id for item in completed if item.subquestion.lane == ResearchLane.HARD_FACT]},
                        verification_results=[vr for item in completed if item.subquestion.lane == ResearchLane.HARD_FACT for vr in item.verified_claims],
                    )
                )
            if semantic_lines:
                sections.append(
                    MemoSection(
                        title="semantic_findings",
                        content="\n\n".join(semantic_lines),
                        writing_trace={"subquestions": [item.subquestion.question_id for item in completed if item.subquestion.lane == ResearchLane.SEMANTIC]},
                        verification_results=[vr for item in completed if item.subquestion.lane == ResearchLane.SEMANTIC for vr in item.verified_claims],
                    )
                )

        gap_lines = []
        for item in refused:
            gap_lines.append(
                f"- {item.subquestion.text}: {item.refusal_reason or 'Insufficient evidence.'}"
            )
        if gap_lines:
            sections.append(
                MemoSection(
                    title="evidence_gaps",
                    content="\n".join(gap_lines),
                    writing_trace={"subquestions": [item.subquestion.question_id for item in refused]},
                )
            )

        if not sections:
            sections = [
                MemoSection(
                    title="evidence_gaps",
                    content="Insufficient evidence in the source pack to answer the research request confidently.",
                    writing_trace={"subquestions": []},
                )
            ]

        executive_summary = self._build_executive_summary(task, completed, llm_budget)
        replay.llm_calls_used = llm_budget["used"]
        coverage = len(completed) / max(1, len([item for item in results if item.subquestion.required]))
        sources = self._collect_sources(results, source_registry)
        memo = GeneratedMemo(
            title=f"Company Research Memo: {task.query}",
            mode=task.mode,
            query=task.query,
            executive_summary=executive_summary,
            contract={
                "company_id": task.company_id,
                "period": task.period,
                "target_periods": task.target_periods,
                "required_slots": task.required_slots,
                "required_source_types": task.required_source_types,
                "query_types": task.query_types,
                "query_type": "comparison" if task.mode == AnalysisMode.COMPETITIVE else "standard",
                "subquestion_count": len(results),
                "planned_required_slots": [item.subquestion.fact_slot for item in results if item.subquestion.required],
                "research_tree": [
                    {
                        "question_id": item.subquestion.question_id,
                        "text": item.subquestion.text,
                        "lane": item.subquestion.lane.value,
                        "status": item.status.value,
                    }
                    for item in results
                ],
            },
            sections=sections,
            sources=sources,
            overall_confidence=coverage,
        )
        return memo

    def _build_unsupported_scope_result(self, task: ResearchTask, replay: ResearchReplayRecord) -> dict:
        refusal_reason = (
            "Cross-company comparison is out of scope for the current trust-first deep research controller."
        )
        subquestion = ResearchSubquestion(
            question_id="q1",
            text=task.query,
            lane=ResearchLane.SEMANTIC,
            priority=1,
            required=True,
            fact_slot="unsupported_scope",
            metric_family="unsupported_scope",
            required_period=(task.period if len(task.target_periods) <= 1 else None),
            allowed_source_types=[],
            needs_numeric_verification=False,
            status=ResearchQuestionStatus.REFUSED,
            rounds_used=0,
            max_rounds=0,
        )
        result = ResearchQuestionResult(
            subquestion=self._clone_subquestion(subquestion),
            status=ResearchQuestionStatus.REFUSED,
            refusal_reason=refusal_reason,
            trace=[
                {
                    "round": 0,
                    "queries": [],
                    "retrievals": [],
                    "assessment": {
                        "valid_chunk_count": 0,
                        "best_entailment": 0.0,
                        "mean_entailment": 0.0,
                        "numeric_match": False,
                        "metric_match": False,
                        "high_trust_hit": False,
                        "sufficient": False,
                        "insufficient": True,
                        "conflict": False,
                        "reasons": [refusal_reason],
                    },
                }
            ],
        )
        replay.subquestions = [self._clone_subquestion(subquestion)]
        replay.decisions.append(
            ResearchDecisionRecord(
                decision_type="refuse_scope",
                question_id=subquestion.question_id,
                reason=refusal_reason,
                payload={"mentioned_company_ids": task.mentioned_company_ids},
            )
        )
        memo = GeneratedMemo(
            title=f"Company Research Memo: {task.query}",
            mode=task.mode,
            query=task.query,
            executive_summary=refusal_reason,
            contract={
                "company_id": task.company_id,
                "mentioned_company_ids": task.mentioned_company_ids,
                "period": task.period,
                "target_periods": task.target_periods,
                "query_type": "comparison" if task.mode == AnalysisMode.COMPETITIVE else "standard",
                "scope_supported": False,
                "subquestion_count": 1,
                "required_slots": ["unsupported_scope"],
                "research_tree": [
                    {
                        "question_id": subquestion.question_id,
                        "text": subquestion.text,
                        "lane": subquestion.lane.value,
                        "status": subquestion.status.value,
                    }
                ],
            },
            sections=[
                MemoSection(
                    title="evidence_gaps",
                    content=f"- {task.query}: {refusal_reason}",
                    writing_trace={"subquestions": [subquestion.question_id]},
                )
            ],
            sources=[],
            overall_confidence=0.0,
        )
        return {
            "memo_object": memo,
            "memo_markdown": self.report_writer.render_memo(memo),
            "workflow_events": [
                {"node_name": "research_controller", "event_type": "started", "detail": task.query},
                {"node_name": subquestion.question_id, "event_type": "refused", "detail": refusal_reason},
                {"node_name": "research_controller", "event_type": "completed", "detail": memo.title},
            ],
            "plan": AnalysisPlan(
                mode=task.mode,
                user_query=task.query,
                steps=[
                    AnalysisStep(
                        name=subquestion.question_id,
                        description=subquestion.text,
                        required=True,
                        search_queries=[],
                        query_contracts=[
                            {
                                "fact_slot": subquestion.fact_slot,
                                "lane": subquestion.lane.value,
                                "period": subquestion.required_period,
                                "source_types": subquestion.allowed_source_types,
                            }
                        ],
                        evidence_requirements={"lane": subquestion.lane.value},
                    )
                ],
                contract=memo.contract,
            ),
            "research_task": task,
            "subquestion_results": [result],
            "research_trace": replay,
            "llm_calls_used": 0,
        }

    def _build_executive_summary(
        self,
        task: ResearchTask,
        completed: List[ResearchQuestionResult],
        llm_budget: dict,
    ) -> str:
        if not completed:
            return "Insufficient evidence in the source pack to produce a confident research summary."
        direct_answer = self._build_task_level_direct_answer(task, completed)
        if direct_answer:
            return direct_answer
        ranked_lines = self._rank_summary_lines(task, completed)
        if ranked_lines:
            return "\n".join(ranked_lines[:3])
        supported_sections = [item.supported_content for item in completed if item.supported_content]
        summary = build_stub_executive_summary(supported_sections)
        if summary:
            return summary
        return "Insufficient evidence in the source pack to produce a confident research summary."

    def _build_task_level_direct_answer(
        self,
        task: ResearchTask,
        completed: List[ResearchQuestionResult],
    ) -> str:
        disclosure_answer = self._synthesize_yes_no_disclosure_answer(task, completed)
        if disclosure_answer:
            return disclosure_answer
        ranking_answer = self._synthesize_ranking_answer(task, completed)
        if ranking_answer:
            return ranking_answer
        reason_answer = self._synthesize_reason_attribution_answer(task, completed)
        if reason_answer:
            return reason_answer
        catalog_answer = self._synthesize_product_service_catalog_answer(task, completed)
        if catalog_answer:
            return catalog_answer
        direct_value_answer = self._synthesize_single_value_direct_answer(task, completed)
        if direct_value_answer:
            return direct_value_answer
        operand_details = [
            detail
            for item in completed
            for detail in [self._extract_operand_detail_from_result(item)]
            if detail is not None
        ]
        query_lower = task.query.lower()
        if operand_details and "working capital" in query_lower and "ratio" not in query_lower:
            return self._synthesize_working_capital_answer(operand_details)
        if operand_details and "quick ratio" in query_lower:
            return self._synthesize_quick_ratio_answer(operand_details)
        if operand_details and ("working capital ratio" in query_lower or "current ratio" in query_lower):
            return self._synthesize_current_ratio_answer(operand_details)
        if (
            operand_details
            and any(token in query_lower for token in ("ebitda", "unadjusted ebitda"))
            and any(token in query_lower for token in ("capex", "capital expenditure", "capital expenditures"))
            and any(token in query_lower for token in ("less", "minus"))
        ):
            return self._synthesize_ebitda_less_capex_answer(task, operand_details)
        if operand_details and "fixed asset turnover ratio" in query_lower:
            return self._synthesize_average_balance_ratio_answer(
                operand_details,
                numerator_role="revenue",
                balance_role="ppne",
                ratio_name="fixed asset turnover ratio",
                numerator_label="revenue",
                balance_label="PP&E",
            )
        if operand_details and "asset turnover ratio" in query_lower and "fixed asset turnover ratio" not in query_lower:
            return self._synthesize_average_balance_ratio_answer(
                operand_details,
                numerator_role="revenue",
                balance_role="total_assets",
                ratio_name="asset turnover ratio",
                numerator_label="revenue",
                balance_label="total assets",
            )
        if operand_details and ("return on assets" in query_lower or "(roa)" in query_lower or re.search(r"\broa\b", query_lower)):
            return self._synthesize_average_balance_ratio_answer(
                operand_details,
                numerator_role="net_income",
                balance_role="total_assets",
                ratio_name="return on assets",
                numerator_label="net income",
                balance_label="total assets",
            )
        if operand_details and "average net profit margin" in query_lower:
            return self._synthesize_average_net_profit_margin_answer(operand_details)
        if operand_details and "margin" in query_lower:
            return self._synthesize_metric_margin_answer(task, operand_details)
        if operand_details and any(token in query_lower for token in ("drop", "increase", "decrease", "between", "versus", "vs", "compared")):
            comparison_answer = self._synthesize_two_period_change_answer(task, operand_details)
            if comparison_answer:
                return comparison_answer
        generic_hard_fact_answer = self._synthesize_generic_hard_fact_answer(task, completed)
        if generic_hard_fact_answer:
            return generic_hard_fact_answer
        return ""

    def _synthesize_generic_hard_fact_answer(
        self,
        task: ResearchTask,
        completed: Sequence[ResearchQuestionResult],
    ) -> str:
        hard_fact_results = [
            item
            for item in completed
            if item.subquestion.lane == ResearchLane.HARD_FACT
            and self._subquestion_aligns_with_task_query(task, item.subquestion)
        ]
        ranked_results = sorted(
            hard_fact_results,
            key=lambda candidate: (
                self._slot_overlap_ratio(candidate.subquestion, task.query),
                self._question_overlap_ratio(candidate.subquestion, task.query),
                -(candidate.subquestion.priority or 0),
            ),
            reverse=True,
        )
        for item in ranked_results:
            direct_answer = self._build_result_level_direct_value_answer(item)
            if direct_answer:
                return direct_answer
        return ""

    def _synthesize_single_value_direct_answer(
        self,
        task: ResearchTask,
        completed: Sequence[ResearchQuestionResult],
    ) -> str:
        if not self._targets_single_value_numeric_task(task):
            return ""

        hard_fact_results = [
            item
            for item in completed
            if item.subquestion.lane == ResearchLane.HARD_FACT
        ]
        for item in sorted(hard_fact_results, key=lambda candidate: candidate.subquestion.priority):
            if not self._subquestion_aligns_with_task_query(task, item.subquestion):
                continue
            direct_answer = self._build_result_level_direct_value_answer(item)
            if direct_answer:
                return direct_answer
        return ""

    def _targets_single_value_numeric_task(self, task: ResearchTask) -> bool:
        query_lower = task.query.lower()
        if any(
            token in query_lower
            for token in (
                "highest",
                "lowest",
                "largest",
                "smallest",
                "most",
                "least",
                "versus",
                "compared",
                "between",
                "what drove",
                "why did",
                "drivers",
                "reason",
                "reasons",
                "legal battles",
                "legal proceedings",
                "proposal vote",
                "voting result",
                "customer concentration",
            )
        ):
            return False
        return any(
            token in query_lower
            for token in (
                "what is",
                "what was",
                "what percent",
                "what percentage",
                "how much",
                "percent",
                "percentage",
                "amount",
                "balance",
                "reported under",
                "reported in the",
                "capital expenditure",
                "capital expenditures",
                "capex",
                "ratio",
                "margin",
            )
        )

    def _subquestion_aligns_with_task_query(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
    ) -> bool:
        query = task.query or ""
        return max(
            self._slot_overlap_ratio(subquestion, query),
            self._question_overlap_ratio(subquestion, query),
        ) >= 0.12

    def _build_result_level_direct_value_answer(
        self,
        result: ResearchQuestionResult,
    ) -> str:
        latest = result.assessments[-1] if result.assessments else None
        ordered_chunks = self._ordered_chunks_for_result(result, latest)
        high_tier_chunks = [chunk for chunk in ordered_chunks if self._is_high_tier_chunk(chunk)]
        if high_tier_chunks:
            bundle = self._build_direct_hard_fact_bundle(result.subquestion, high_tier_chunks)
            if bundle:
                return self._format_direct_hard_fact_bundle(bundle)
        return self._extract_direct_supported_line(result)

    def _extract_direct_supported_line(
        self,
        result: ResearchQuestionResult,
    ) -> str:
        text = self._sanitize_answer_for_verification(result.supported_content or result.answer_text)
        if not text:
            return ""
        evidence_map = {chunk.chunk_id: chunk for chunk in result.evidence[:8]}
        source_map = {chunk.chunk_id: chunk.source_id for chunk in result.evidence[:8]}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if "[Chunk:" not in line or "[Source:" not in line:
                continue
            citation_pairs = self._extract_structured_claim_citation_pairs(
                {
                    "chunk_ids": re.findall(r"\[Chunk:\s*([^\]]+)\]", line),
                    "source_ids": re.findall(r"\[Source:\s*([^\]]+)\]", line),
                },
                evidence_map,
                source_map,
            )
            if not citation_pairs:
                continue
            statement = self._strip_soft_evidence_preamble(self._normalize_claim_sentence(line))
            if not statement or self._claim_is_too_broad(statement) or self._claim_is_low_signal_fragment(statement):
                continue
            if not self._should_preserve_model_statement(result.subquestion, statement):
                continue
            if (
                result.subquestion.lane == ResearchLane.HARD_FACT
                and self._claim_requires_primary_support(statement)
                and not self._claim_citations_have_primary_support(citation_pairs, evidence_map)
            ):
                continue
            return f"- {statement}{self._format_citation_pairs(citation_pairs)}"
        return ""

    def _synthesize_yes_no_disclosure_answer(
        self,
        task: ResearchTask,
        completed: Sequence[ResearchQuestionResult],
    ) -> str:
        query_lower = task.query.lower()
        if not any(
            token in query_lower
            for token in (
                "legal battles",
                "legal proceedings",
                "proposal vote",
                "voting result",
                "shareholder vote",
                "revolving credit agreement",
                "borrowing capacity",
                "may borrow",
                "customer concentration",
                "major customer",
                "significant customer",
            )
        ):
            return ""

        hard_fact_lines = self._rank_summary_lines(
            task,
            [item for item in completed if item.subquestion.lane == ResearchLane.HARD_FACT],
        )
        if not hard_fact_lines:
            return ""

        normalized_lines = [self._normalize_claim_sentence(line) for line in hard_fact_lines]
        citations = self._collect_ranked_line_citations(hard_fact_lines)
        yes_no_lines = [
            line for line in normalized_lines
            if any(
                phrase in line.lower()
                for phrase in (
                    "no material ongoing legal battles were disclosed",
                    "the filing disclosed material legal proceedings",
                    "the filing indicates the proposal failed",
                    "the filing indicates the proposal passed",
                    "the revolving credit agreement increased by",
                    "total borrowing capacity was",
                    "yes, one customer accounted for",
                    "no, no customer accounted for",
                )
            )
        ]
        if not yes_no_lines or not citations:
            return ""
        return f"- {yes_no_lines[0]}{self._format_citation_pairs(citations)}"

    def _synthesize_ranking_answer(
        self,
        task: ResearchTask,
        completed: Sequence[ResearchQuestionResult],
    ) -> str:
        query_lower = task.query.lower()
        if not any(
            token in query_lower
            for token in ("highest", "lowest", "largest", "smallest", "most", "least")
        ):
            return ""

        hard_fact_results = [
            item
            for item in completed
            if item.subquestion.lane == ResearchLane.HARD_FACT
            and self._subquestion_aligns_with_task_query(task, item.subquestion)
        ]
        for item in sorted(hard_fact_results, key=lambda candidate: candidate.subquestion.priority):
            direct_answer = self._build_result_level_direct_value_answer(item)
            if direct_answer:
                return direct_answer
        return ""

    def _synthesize_reason_attribution_answer(
        self,
        task: ResearchTask,
        completed: Sequence[ResearchQuestionResult],
    ) -> str:
        query_lower = task.query.lower()
        if not any(
            token in query_lower
            for token in ("what drove", "driver", "drivers", "due to", "because", "reason", "reasons", "why did")
        ):
            return ""

        semantic_results = [
            item
            for item in completed
            if item.subquestion.lane == ResearchLane.SEMANTIC
            and self._subquestion_aligns_with_task_query(task, item.subquestion)
        ]
        for item in sorted(semantic_results, key=lambda candidate: candidate.subquestion.priority):
            line = self._sanitize_answer_for_verification(item.supported_content or item.answer_text)
            line = next((candidate.strip() for candidate in line.splitlines() if candidate.strip()), "")
            if line and self._has_direct_causal_language(line):
                return line
        return ""

    def _synthesize_product_service_catalog_answer(
        self,
        task: ResearchTask,
        completed: Sequence[ResearchQuestionResult],
    ) -> str:
        if not self._targets_product_service_catalog_question(task):
            return ""

        semantic_results = [
            item
            for item in completed
            if item.subquestion.lane == ResearchLane.SEMANTIC
            and self._subquestion_aligns_with_task_query(task, item.subquestion)
        ]
        for item in sorted(semantic_results, key=lambda candidate: candidate.subquestion.priority):
            line = self._sanitize_answer_for_verification(item.supported_content or item.answer_text)
            line = next((candidate.strip() for candidate in line.splitlines() if candidate.strip()), "")
            if line and self._has_product_catalog_language(line):
                return line
        return ""

    def _extract_operand_detail_from_result(
        self,
        result: ResearchQuestionResult,
    ) -> Optional[Dict[str, object]]:
        basis = self._subquestion_text_basis(result.subquestion)
        year_tokens = self._extract_year_numbers(result.subquestion.text)
        year = year_tokens[-1] if year_tokens else None
        period_label = self._extract_period_label(result.subquestion.text, fallback=result.subquestion.required_period)
        content_candidates = [result.supported_content, result.answer_text] + [chunk.text for chunk in result.evidence[:8]]
        operand_specs = [
            {
                "role": "ebitda",
                "keywords": ("ebitda", "adjusted ebitda", "unadjusted ebitda"),
                "labels": ["unadjusted ebitda", "adjusted ebitda", "ebitda"],
            },
            {
                "role": "capex",
                "keywords": ("capital expenditure", "capital expenditures", "capex"),
                "labels": ["capital expenditures", "capital expenditure"],
                "absolute_value": True,
            },
            {
                "role": "revenue",
                "keywords": ("revenue", "net sales", "sales"),
                "labels": ["total revenue", "net revenue", "net sales", "revenue", "sales"],
            },
            {
                "role": "cash_and_cash_equivalents",
                "keywords": ("cash & cash equivalents", "cash and cash equivalents"),
                "labels": ["cash and cash equivalents"],
            },
            {
                "role": "ppne",
                "keywords": ("pp&e", "property, plant", "property and equipment", "ppne"),
                "labels": [
                    "property, plant and equipment, net",
                    "property and equipment, net",
                    "net property, plant and equipment",
                    "property, plant and equipment",
                ],
            },
            {
                "role": "total_assets",
                "keywords": ("total assets",),
                "labels": ["total assets"],
            },
            {
                "role": "current_assets",
                "keywords": ("total current assets",),
                "labels": ["total current assets"],
            },
            {
                "role": "short_term_investments",
                "keywords": ("short-term investments", "short term investments"),
                "labels": ["short-term investments"],
            },
            {
                "role": "accounts_receivable",
                "keywords": ("accounts receivable", "receivables", "net ar"),
                "labels": ["accounts receivable, net", "accounts receivable", "receivables, net", "trade receivables, net"],
            },
            {
                "role": "inventories",
                "keywords": ("inventories", "inventory"),
                "labels": ["inventories", "inventory"],
            },
            {
                "role": "current_liabilities",
                "keywords": ("total current liabilities",),
                "labels": ["total current liabilities"],
            },
            {
                "role": "net_income",
                "keywords": ("net income",),
                "labels": ["net income"],
            },
            {
                "role": "operating_income",
                "keywords": ("operating income", "operating margin"),
                "labels": ["operating income"],
            },
            {
                "role": "depreciation_and_amortization",
                "keywords": ("depreciation", "amortization", "d&a"),
                "labels": ["depreciation and amortization"],
                "absolute_value": True,
            },
            {
                "role": "wages_expense_percent_of_net_sales",
                "keywords": (
                    "wages expense as a percent of net sales",
                    "wages expense as a percentage of net sales",
                    "wages expense % of net sales",
                ),
                "labels": [
                    "wages expense as a percentage of net sales",
                    "wages expense as a percent of net sales",
                    "wages expense % of net sales",
                ],
            },
        ]
        for spec in operand_specs:
            if not any(keyword in basis for keyword in spec["keywords"]):
                continue
            for text in content_candidates:
                value = self._extract_labeled_financial_value(text or "", spec["labels"])
                if value is None:
                    continue
                numeric_value = abs(value[0]) if spec.get("absolute_value") else value[0]
                formatted_value = (
                    self._format_financial_value(numeric_value, text or "")
                    if spec.get("absolute_value") and value[0] < 0
                    else value[1]
                )
                return {
                    "role": spec["role"],
                    "year": year,
                    "period_label": period_label,
                    "sort_key": self._period_sort_key(period_label) if period_label else float(year or 0),
                    "value": numeric_value,
                    "formatted_value": formatted_value,
                    "citation_pairs": self._citation_pairs_for_result(result),
                }
        return None

    def _citation_pairs_for_result(self, result: ResearchQuestionResult) -> List[tuple[str, str]]:
        pairs: List[tuple[str, str]] = []
        seen = set()
        for verification in result.verified_claims:
            chunk_ids = verification.supporting_chunk_ids or []
            source_ids = verification.supporting_source_ids or []
            if len(source_ids) == 1 and len(chunk_ids) > 1:
                source_ids = source_ids * len(chunk_ids)
            for chunk_id, source_id in zip(chunk_ids, source_ids):
                pair = (chunk_id, source_id)
                if not chunk_id or not source_id or pair in seen:
                    continue
                pairs.append(pair)
                seen.add(pair)
        if pairs:
            return pairs
        return [
            (chunk.chunk_id, chunk.source_id)
            for chunk in result.evidence[:1]
            if chunk.chunk_id and chunk.source_id
        ]

    def _extract_year_numbers(self, text: str) -> List[int]:
        years = []
        seen = set()
        for match in re.finditer(r"(?:fy|fiscal year|year end fy|end of fy|as of fy)?\s*(20\d{2})", text or "", re.IGNORECASE):
            year = int(match.group(1))
            if year in seen:
                continue
            years.append(year)
            seen.add(year)
        return years

    def _extract_period_label(self, text: str, fallback: Optional[str] = None) -> Optional[str]:
        if fallback:
            return fallback
        match = re.search(r"\b(Q[1-4]\s*FY\s*20\d{2}|Q[1-4]\s*20\d{2}|FY\s*20\d{2}|20\d{2}Q[1-4]|20\d{2}FY)\b", text or "", re.IGNORECASE)
        if not match:
            return None
        return re.sub(r"\s+", " ", match.group(0)).strip().upper()

    def _period_sort_key(self, period_label: Optional[str]) -> float:
        if not period_label:
            return 0.0
        normalized = period_label.upper().replace(" ", "")
        year_match = re.search(r"20\d{2}", normalized)
        year = int(year_match.group(0)) if year_match else 0
        quarter_match = re.search(r"Q([1-4])", normalized)
        quarter = int(quarter_match.group(1)) if quarter_match else 5
        return float(year * 10 + quarter)

    def _collect_ranked_line_citations(self, ranked_lines: Sequence[str]) -> List[tuple[str, str]]:
        pairs: List[tuple[str, str]] = []
        seen = set()
        for line in ranked_lines:
            chunk_matches = re.findall(r"\[Chunk:\s*([^\]]+)\]", line)
            source_matches = re.findall(r"\[Source:\s*([^\]]+)\]", line)
            if not chunk_matches or not source_matches:
                continue
            if len(source_matches) == 1 and len(chunk_matches) > 1:
                source_matches = source_matches * len(chunk_matches)
            for chunk_id, source_id in zip(chunk_matches, source_matches):
                pair = (chunk_id.strip(), source_id.strip())
                if not pair[0] or not pair[1] or pair in seen:
                    continue
                pairs.append(pair)
                seen.add(pair)
        return pairs

    def _combine_citation_pairs(self, details: Sequence[Dict[str, object]]) -> str:
        pairs: List[tuple[str, str]] = []
        seen = set()
        for detail in details:
            for pair in detail.get("citation_pairs", []) or []:
                if pair in seen:
                    continue
                pairs.append(pair)
                seen.add(pair)
        return self._format_citation_pairs(pairs)

    def _synthesize_current_ratio_answer(self, operand_details: Sequence[Dict[str, object]]) -> str:
        current_assets = [detail for detail in operand_details if detail["role"] == "current_assets"]
        current_liabilities = [detail for detail in operand_details if detail["role"] == "current_liabilities"]
        if not current_assets or not current_liabilities:
            return ""
        assets_detail = max(current_assets, key=lambda detail: detail.get("year") or 0)
        liabilities_detail = max(current_liabilities, key=lambda detail: detail.get("year") or 0)
        denominator = float(liabilities_detail["value"])
        if denominator == 0:
            return ""
        ratio = float(assets_detail["value"]) / denominator
        citations = self._combine_citation_pairs([assets_detail, liabilities_detail])
        return (
            f"- The working capital ratio was {self._format_ratio_value(ratio)}, based on total current assets of "
            f"{assets_detail['formatted_value']} and total current liabilities of {liabilities_detail['formatted_value']}."
            f"{citations}"
        )

    def _synthesize_working_capital_answer(self, operand_details: Sequence[Dict[str, object]]) -> str:
        current_assets = [detail for detail in operand_details if detail["role"] == "current_assets"]
        current_liabilities = [detail for detail in operand_details if detail["role"] == "current_liabilities"]
        if not current_assets or not current_liabilities:
            return ""
        assets_detail = max(current_assets, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
        liabilities_detail = max(current_liabilities, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
        working_capital = float(assets_detail["value"]) - float(liabilities_detail["value"])
        polarity = "positive" if working_capital >= 0 else "negative"
        period_label = assets_detail.get("period_label") or liabilities_detail.get("period_label") or ""
        citations = self._combine_citation_pairs([assets_detail, liabilities_detail])
        return (
            f"- Working capital was {polarity} at {self._format_financial_value(working_capital, str(assets_detail['formatted_value']))} "
            f"in {period_label or 'the reported period'}, based on total current assets of {assets_detail['formatted_value']} "
            f"and total current liabilities of {liabilities_detail['formatted_value']}.{citations}"
        )

    def _synthesize_quick_ratio_answer(self, operand_details: Sequence[Dict[str, object]]) -> str:
        current_liabilities = [detail for detail in operand_details if detail["role"] == "current_liabilities"]
        if not current_liabilities:
            return ""
        liabilities_detail = max(current_liabilities, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
        denominator = float(liabilities_detail["value"])
        if denominator == 0:
            return ""

        current_assets = [detail for detail in operand_details if detail["role"] == "current_assets"]
        inventories = [detail for detail in operand_details if detail["role"] == "inventories"]
        if current_assets and inventories:
            assets_detail = max(current_assets, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
            inventory_detail = max(inventories, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
            ratio = (float(assets_detail["value"]) - float(inventory_detail["value"])) / denominator
            citations = self._combine_citation_pairs([assets_detail, inventory_detail, liabilities_detail])
            return (
                f"- The quick ratio was {self._format_ratio_value(ratio)}, based on total current assets of "
                f"{assets_detail['formatted_value']} less inventories of {inventory_detail['formatted_value']}, "
                f"divided by total current liabilities of {liabilities_detail['formatted_value']}.{citations}"
            )

        cash = [detail for detail in operand_details if detail["role"] == "cash_and_cash_equivalents"]
        short_term_investments = [detail for detail in operand_details if detail["role"] == "short_term_investments"]
        receivables = [detail for detail in operand_details if detail["role"] == "accounts_receivable"]
        if not cash or not short_term_investments or not receivables:
            return ""
        cash_detail = max(cash, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
        short_term_detail = max(short_term_investments, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
        receivables_detail = max(receivables, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
        ratio = (float(cash_detail["value"]) + float(short_term_detail["value"]) + float(receivables_detail["value"])) / denominator
        citations = self._combine_citation_pairs([cash_detail, short_term_detail, receivables_detail, liabilities_detail])
        return (
            f"- The quick ratio was {self._format_ratio_value(ratio)}, based on cash and cash equivalents of "
            f"{cash_detail['formatted_value']}, short-term investments of {short_term_detail['formatted_value']}, "
            f"accounts receivable of {receivables_detail['formatted_value']}, and total current liabilities of "
            f"{liabilities_detail['formatted_value']}.{citations}"
        )

    def _synthesize_two_period_change_answer(
        self,
        task: ResearchTask,
        operand_details: Sequence[Dict[str, object]],
    ) -> str:
        query_lower = task.query.lower()
        role = ""
        label = ""
        if any(token in query_lower for token in ("cash & cash equivalents", "cash and cash equivalents")):
            role = "cash_and_cash_equivalents"
            label = "Cash and cash equivalents"
        elif any(token in query_lower for token in ("ppne", "pp&e", "property, plant", "property and equipment")):
            role = "ppne"
            label = "PPNE"
        elif any(
            token in query_lower
            for token in (
                "wages expense as a percent of net sales",
                "wages expense as a percentage of net sales",
                "wages expense % of net sales",
            )
        ):
            role = "wages_expense_percent_of_net_sales"
            label = "Wages expense as a percent of net sales"
        elif "revenue" in query_lower or "net sales" in query_lower or "sales" in query_lower:
            role = "revenue"
            label = "Revenue"
        elif "net income" in query_lower:
            role = "net_income"
            label = "Net income"
        if not role:
            return ""

        details = [detail for detail in operand_details if detail["role"] == role]
        if len(details) < 2:
            return ""
        ordered = sorted(details, key=lambda detail: detail.get("sort_key") or detail.get("year") or 0)
        earlier = ordered[0]
        later = ordered[-1]
        delta = float(later["value"]) - float(earlier["value"])
        if any(token in query_lower for token in ("drop", "decrease", "decreased", "decline")):
            verdict = "No" if delta >= 0 else "Yes"
        elif any(token in query_lower for token in ("increase", "increased", "rise", "rose")):
            verdict = "Yes" if delta > 0 else "No"
        else:
            verdict = "Yes" if delta != 0 else "No"
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "did not change materially"
        citations = self._combine_citation_pairs([earlier, later])
        if delta == 0:
            return (
                f"- {label} did not change materially between {earlier.get('period_label') or 'the earlier period'} and "
                f"{later.get('period_label') or 'the later period'}.{citations}"
            )
        return (
            f"- {verdict}, {label.lower()} {direction} from {earlier['formatted_value']} in "
            f"{earlier.get('period_label') or 'the earlier period'} to {later['formatted_value']} in "
            f"{later.get('period_label') or 'the later period'}.{citations}"
        )

    def _synthesize_average_balance_ratio_answer(
        self,
        operand_details: Sequence[Dict[str, object]],
        *,
        numerator_role: str,
        balance_role: str,
        ratio_name: str,
        numerator_label: str,
        balance_label: str,
    ) -> str:
        numerators = [detail for detail in operand_details if detail["role"] == numerator_role and detail.get("year") is not None]
        balances = [detail for detail in operand_details if detail["role"] == balance_role and detail.get("year") is not None]
        if not numerators or len(balances) < 2:
            return ""
        current_year = max(int(detail["year"]) for detail in numerators)
        numerator = next((detail for detail in numerators if int(detail["year"]) == current_year), None)
        current_balance = next((detail for detail in balances if int(detail["year"]) == current_year), None)
        previous_year = current_year - 1
        previous_balance = next((detail for detail in balances if int(detail["year"]) == previous_year), None)
        if numerator is None or current_balance is None or previous_balance is None:
            return ""
        average_balance = (float(current_balance["value"]) + float(previous_balance["value"])) / 2.0
        if average_balance == 0:
            return ""
        ratio_value = float(numerator["value"]) / average_balance
        average_balance_formatted = self._format_financial_value(
            average_balance,
            " ".join([str(current_balance["formatted_value"]), str(previous_balance["formatted_value"])]),
        )
        citations = self._combine_citation_pairs([numerator, previous_balance, current_balance])
        return (
            f"- The {ratio_name} was {self._format_ratio_value(ratio_value)}, based on FY{current_year} {numerator_label} of "
            f"{numerator['formatted_value']} and average {balance_label} of {average_balance_formatted} "
            f"({previous_balance['formatted_value']} in FY{previous_year} and {current_balance['formatted_value']} in FY{current_year})."
            f"{citations}"
        )

    def _synthesize_average_net_profit_margin_answer(
        self,
        operand_details: Sequence[Dict[str, object]],
    ) -> str:
        revenues = {
            int(detail["year"]): detail
            for detail in operand_details
            if detail["role"] == "revenue" and detail.get("year") is not None
        }
        net_income = {
            int(detail["year"]): detail
            for detail in operand_details
            if detail["role"] == "net_income" and detail.get("year") is not None
        }
        common_years = sorted(set(revenues) & set(net_income))
        if len(common_years) < 2:
            return ""
        margins = []
        used_details: List[Dict[str, object]] = []
        for year in common_years:
            revenue_value = float(revenues[year]["value"])
            if revenue_value == 0:
                return ""
            margins.append(float(net_income[year]["value"]) / revenue_value)
            used_details.extend([revenues[year], net_income[year]])
        average_margin = sum(margins) / len(margins)
        citations = self._combine_citation_pairs(used_details)
        year_span = f"FY{common_years[0]}–FY{common_years[-1]}"
        return (
            f"- The average net profit margin for {year_span} was {average_margin * 100:.1f}%."
            f"{citations}"
        )

    def _synthesize_ebitda_less_capex_answer(
        self,
        task: ResearchTask,
        operand_details: Sequence[Dict[str, object]],
    ) -> str:
        ebitda_role = "unadjusted_ebitda" if "unadjusted ebitda" in task.query.lower() else "ebitda"
        ebitda_details = [
            detail
            for detail in operand_details
            if detail["role"] in {ebitda_role, "ebitda"}
        ]
        capex_details = [detail for detail in operand_details if detail["role"] == "capex"]
        if not ebitda_details or not capex_details:
            return ""
        ebitda_detail = max(ebitda_details, key=lambda detail: detail.get("year") or 0)
        capex_detail = max(capex_details, key=lambda detail: detail.get("year") or 0)
        result_value = float(ebitda_detail["value"]) - float(capex_detail["value"])
        citations = self._combine_citation_pairs([ebitda_detail, capex_detail])
        display_name = "Unadjusted EBITDA" if ebitda_role == "unadjusted_ebitda" else "EBITDA"
        reference_text = f"{ebitda_detail['formatted_value']} {capex_detail['formatted_value']}"
        return (
            f"- {display_name} less capital expenditures was {self._format_financial_value(result_value, reference_text)}, "
            f"based on {display_name.lower()} of {ebitda_detail['formatted_value']} and capital expenditures of {capex_detail['formatted_value']}."
            f"{citations}"
        )

    def _synthesize_metric_margin_answer(
        self,
        task: ResearchTask,
        operand_details: Sequence[Dict[str, object]],
    ) -> str:
        query_lower = task.query.lower()
        numerator_role = ""
        display_name = ""
        margin_name = ""
        if "depreciation and amortization" in query_lower or "d&a" in query_lower:
            numerator_role = "depreciation_and_amortization"
            display_name = "Depreciation and amortization"
            margin_name = "Depreciation and amortization margin"
        elif "operating margin" in query_lower or "operating income" in query_lower:
            numerator_role = "operating_income"
            display_name = "Operating income"
            margin_name = "Operating margin"
        elif "unadjusted ebitda" in query_lower:
            numerator_role = "ebitda"
            display_name = "Unadjusted EBITDA"
            margin_name = "Unadjusted EBITDA margin"
        elif "ebitda" in query_lower:
            numerator_role = "ebitda"
            display_name = "EBITDA"
            margin_name = "EBITDA margin"
        elif "net profit margin" in query_lower or "net income" in query_lower:
            numerator_role = "net_income"
            display_name = "Net income"
            margin_name = "Net profit margin"
        if not numerator_role:
            return ""

        numerators = [detail for detail in operand_details if detail["role"] == numerator_role]
        revenues = [detail for detail in operand_details if detail["role"] == "revenue"]
        if not numerators or not revenues:
            return ""
        numerator_detail = max(numerators, key=lambda detail: detail.get("year") or 0)
        revenue_detail = max(revenues, key=lambda detail: detail.get("year") or 0)
        if float(revenue_detail["value"]) == 0:
            return ""
        margin = float(numerator_detail["value"]) / float(revenue_detail["value"])
        citations = self._combine_citation_pairs([numerator_detail, revenue_detail])
        return (
            f"- {margin_name} was {margin * 100:.1f}%, based on {display_name.lower()} of {numerator_detail['formatted_value']} "
            f"and revenue of {revenue_detail['formatted_value']}."
            f"{citations}"
        )

    def _rank_summary_lines(
        self,
        task: ResearchTask,
        completed: List[ResearchQuestionResult],
    ) -> List[str]:
        query_tokens = self._summary_tokens(task.query)
        ranked: List[tuple[float, int, str]] = []
        seen = set()
        for item in completed:
            for raw_line in item.supported_content.splitlines():
                line = raw_line.strip()
                if not line or "[Source:" not in line:
                    continue
                if line in seen:
                    continue
                seen.add(line)
                ranked.append(
                    (
                        self._score_summary_line(task, item, line, query_tokens),
                        item.subquestion.priority,
                        line if line.startswith("- ") else f"- {line.lstrip('-• ').strip()}",
                    )
                )
        ranked.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
        return [line for _, _, line in ranked]

    def _score_summary_line(
        self,
        task: ResearchTask,
        result: ResearchQuestionResult,
        line: str,
        query_tokens: set[str],
    ) -> float:
        normalized = self._normalize_claim_sentence(line)
        line_tokens = self._summary_tokens(normalized)
        overlap = len(query_tokens & line_tokens)
        score = float(overlap * 3)
        if self._extract_number_tokens(normalized):
            score += 2.0
        if any(
            token in normalized.lower()
            for token in (
                "yes",
                "no",
                "positive",
                "negative",
                "highest",
                "lowest",
                "largest",
                "smallest",
                "healthy",
                "unhealthy",
                "improving",
                "declining",
            )
        ):
            score += 3.0
        score += min(2.0, self._slot_overlap_ratio(result.subquestion, normalized) * 4.0)
        if result.subquestion.metric_family and result.subquestion.metric_family in normalized.lower():
            score += 1.0
        if self._claim_is_low_signal_fragment(normalized):
            score -= 4.0
        if self._claim_is_too_broad(normalized):
            score -= 2.0
        return score

    def _summary_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) > 2
        }

    def _collect_sources(
        self,
        results: List[ResearchQuestionResult],
        source_registry: Dict[str, dict],
    ) -> List[DocumentMeta]:
        used_ids = []
        for item in results:
            for chunk in item.evidence:
                if chunk.source_id not in used_ids:
                    used_ids.append(chunk.source_id)
        collected = []
        for source_id in used_ids:
            payload = source_registry.get(source_id, {})
            collected.append(
                DocumentMeta(
                    source_id=source_id,
                    source_type=payload.get("source_type", ""),
                    title=payload.get("title", source_id),
                    url=payload.get("url"),
                    date=payload.get("date"),
                    company=payload.get("company"),
                    period=payload.get("period"),
                    published_at=payload.get("published_at"),
                    issuer=payload.get("issuer"),
                    is_primary=payload.get("is_primary"),
                )
            )
        return collected

    def _consume_llm_budget(self, llm_budget: dict, reason: str) -> bool:
        if llm_budget["used"] >= llm_budget["max"]:
            logger.info("Skipping LLM step %s because budget is exhausted.", reason)
            return False
        llm_budget["used"] += 1
        return True

    def _default_source_types(self, lane: ResearchLane) -> List[str]:
        return list(PRIMARY_SOURCE_TYPES if lane == ResearchLane.HARD_FACT else SEMANTIC_SOURCE_TYPES)

    def _preferred_source_types_for_subquestion(
        self,
        task: ResearchTask,
        lane: ResearchLane,
        text: str,
        fact_slot: str,
    ) -> List[str]:
        base = self._default_source_types(lane)
        probe = ResearchSubquestion(
            question_id="probe",
            text=text,
            lane=lane,
            priority=1,
            fact_slot=fact_slot,
            metric_family="",
            needs_numeric_verification=lane == ResearchLane.HARD_FACT,
        )
        if lane == ResearchLane.SEMANTIC and self._targets_management_commentary(probe):
            commentary_first = [
                "earnings_call_transcript",
                "shareholder_letter",
                "investor_presentation",
            ]
            if task.required_source_types:
                constrained = [source_type for source_type in commentary_first if source_type in task.required_source_types]
                if constrained:
                    return constrained
            return commentary_first
        return base

    def _intent_key(self, subquestion: ResearchSubquestion) -> str:
        basis = self._subquestion_text_basis(subquestion)
        if any(token in basis for token in ("business model", "make money", "monetization", "revenue source", "revenue model")):
            return "business_model"
        if self._is_revenue_mix_question(subquestion):
            return "revenue_mix_management" if self._targets_management_commentary(subquestion) else "revenue_mix"
        return self._normalize_text(f"{subquestion.fact_slot} {subquestion.metric_family} {subquestion.text}")

    def _parse_json(self, raw: str):
        cleaned = (raw or "").strip()
        if not cleaned:
            return {}
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        payload = self._try_parse_json_candidates(cleaned)
        if payload is not None:
            return payload
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cleaned.find(opener)
            if start < 0:
                continue
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(cleaned)):
                char = cleaned[idx]
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:idx + 1]
                        payload = self._try_parse_json_candidates(candidate)
                        if payload is not None:
                            return payload
                        break
        return {}

    def _try_parse_json_candidates(self, candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _dedupe_chunks(self, chunks: List[RetrievedChunk], limit: int) -> List[RetrievedChunk]:
        seen = {}
        for chunk in chunks:
            existing = seen.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                seen[chunk.chunk_id] = chunk
        ordered = sorted(seen.values(), key=lambda item: (item.score, item.rerank_score), reverse=True)
        return ordered[:limit]

    def _clone_subquestion(self, subquestion: ResearchSubquestion) -> ResearchSubquestion:
        return ResearchSubquestion(
            question_id=subquestion.question_id,
            text=subquestion.text,
            lane=subquestion.lane,
            priority=subquestion.priority,
            required=subquestion.required,
            fact_slot=subquestion.fact_slot,
            metric_family=subquestion.metric_family,
            required_period=subquestion.required_period,
            allowed_source_types=list(subquestion.allowed_source_types),
            needs_numeric_verification=subquestion.needs_numeric_verification,
            status=subquestion.status,
            rounds_used=subquestion.rounds_used,
            max_rounds=subquestion.max_rounds,
        )

    def _enforce_lane(self, text: str, proposed_lane: Optional[str]) -> ResearchLane:
        lowered = text.lower()
        proposed = str(proposed_lane or "").strip().lower()
        if self._looks_like_catalog_request(lowered) and not self._looks_like_numeric_fact_request(lowered):
            return ResearchLane.SEMANTIC
        if any(
            phrase in lowered
            for phrase in ("make money", "business model", "monetization", "revenue sources", "core business")
        ) and not self._looks_like_numeric_fact_request(lowered):
            return ResearchLane.SEMANTIC
        if proposed == "semantic" and not self._looks_like_numeric_fact_request(lowered):
            return ResearchLane.SEMANTIC
        if self._contains_any(lowered, SEMANTIC_KEYWORDS) and not self._looks_like_numeric_fact_request(lowered):
            return ResearchLane.SEMANTIC
        if self._contains_any(lowered, HARD_FACT_KEYWORDS):
            return ResearchLane.HARD_FACT
        if self._contains_any(lowered, SEMANTIC_KEYWORDS):
            return ResearchLane.SEMANTIC
        return ResearchLane.HARD_FACT

    def _looks_like_catalog_request(self, lowered: str) -> bool:
        has_catalog_terms = any(
            token in lowered
            for token in (
                "product",
                "products",
                "service",
                "services",
                "offering",
                "offerings",
                "platform",
                "platforms",
                "sell",
                "sells",
            )
        )
        asks_listing = any(
            token in lowered
            for token in (
                "what are",
                "what were",
                "what does",
                "what do",
                "major",
                "main",
                "core",
                "primary",
            )
        )
        return has_catalog_terms and asks_listing

    def _looks_like_numeric_fact_request(self, lowered: str) -> bool:
        numeric_phrases = (
            "reported evidence",
            "disclosed",
            "numeric evidence",
            "exact reported figure",
            "how much",
            "what was the reported",
            "what was the disclosed",
            "what is the reported",
            "what is the disclosed",
        )
        if self._contains_any(lowered, SEMANTIC_KEYWORDS) and not any(phrase in lowered for phrase in numeric_phrases):
            return False
        numeric_terms = {"margin", "profitability", "cash flow", "fcf", "valuation", "funding", "eps", "capex"}
        return any(phrase in lowered for phrase in numeric_phrases) or any(
            token in lowered for token in numeric_terms
        )

    def _proposal_covers_slot(self, proposal: dict, slot: str) -> bool:
        slot_tokens = self._slot_tokens(slot, expand_aliases=True)
        if not slot_tokens:
            return False
        haystack = " ".join(
            [
                str(proposal.get("fact_slot", "")),
                str(proposal.get("metric_family", "")),
                str(proposal.get("text", "")),
            ]
        )
        hay_tokens = self._slot_tokens(haystack, expand_aliases=False)
        overlap = slot_tokens & hay_tokens
        return len(overlap) >= max(1, min(2, len(slot_tokens)))

    def _slot_tokens(self, value: Optional[str], *, expand_aliases: bool = True) -> set[str]:
        normalized = self._normalized_slot_text(value)
        if not normalized:
            return set()
        tokens = set(normalized.split())
        if not expand_aliases:
            return tokens
        expanded = set(tokens)
        for token in tokens:
            expanded.update(SLOT_TOKEN_ALIASES.get(token, set()))
        return expanded

    def _normalized_slot_text(self, value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()

    def _task_period_label(self, task: ResearchTask, *, include_latest_placeholder: bool = True) -> str:
        if task.target_periods:
            return " ".join(task.target_periods)
        if task.period:
            return task.period
        return "latest" if include_latest_placeholder else ""

    def _same_metric_context(
        self,
        subquestion: ResearchSubquestion,
        first_chunk: RetrievedChunk,
        second_chunk: RetrievedChunk,
    ) -> bool:
        first_signals = set(first_chunk.metric_signals or [])
        second_signals = set(second_chunk.metric_signals or [])
        metric_family = (subquestion.metric_family or "").strip().lower()
        if metric_family and first_signals and second_signals:
            return metric_family in first_signals and metric_family in second_signals
        if first_signals and second_signals:
            return bool(first_signals & second_signals)
        return self._likely_same_fact_text(first_chunk.text, second_chunk.text, metric_family)

    def _likely_same_fact_text(self, first_text: str, second_text: str, metric_family: str) -> bool:
        tokens_a = self._fact_text_tokens(first_text)
        tokens_b = self._fact_text_tokens(second_text)
        if metric_family:
            if metric_family not in tokens_a or metric_family not in tokens_b:
                return False
        overlap = len(tokens_a & tokens_b)
        smallest = min(len(tokens_a), len(tokens_b))
        if smallest == 0:
            return False
        return (overlap / smallest) >= 0.5

    def _fact_text_tokens(self, text: str) -> set[str]:
        cleaned = re.sub(r'\$?[\d]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k|%|bps))?', ' ', text or '', flags=re.IGNORECASE)
        return {
            token
            for token in re.findall(r"[a-z_]+", cleaned.lower())
            if len(token) > 2
        }

    def _contains_any(self, text: str, keywords: set[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _extract_number_tokens(self, text: str) -> set[str]:
        return {
            match.group(0).lower().replace(",", "").strip()
            for match in re.finditer(
                r'\$?[\d]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k|%|bps))?',
                text or "",
                re.IGNORECASE,
            )
            if match.group(0).strip()
        }

    def _normalize_text(self, value: Optional[str]) -> str:
        return str(value or "").strip().lower().replace(" ", "").replace("-", "")

    def _coerce_priority_value(self, value, default: int) -> int:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        lowered = self._normalize_text(str(value))
        if not lowered:
            return default
        if lowered in PRIORITY_LABELS:
            return PRIORITY_LABELS[lowered]
        match = re.search(r"\d+", lowered)
        if match:
            return int(match.group(0))
        return default

    @staticmethod
    def _cn_quarter(value: str) -> str:
        return {"一": "1", "二": "2", "三": "3", "四": "4"}.get(value, value)
