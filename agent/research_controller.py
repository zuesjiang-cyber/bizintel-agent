import json
import logging
import re
from typing import Dict, List, Optional, Sequence

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
    "management": {"management", "commentary", "tone", "outlook", "guidance", "framing", "narrative", "emphasis"},
    "evidence": {"evidence", "support", "confidence", "distinction", "weight", "weaker", "stronger"},
    "comparison": {"compare", "comparison", "change", "shift", "versus", "vs"},
    "business": {"business", "model", "platform", "offering", "offerings", "product", "products"},
    "monetization": {"monetization", "pricing", "usage", "consumption", "subscription", "commitment", "billing"},
    "pricing": {"pricing", "usage", "consumption", "subscription", "commitment", "billing"},
    "mix": {"mix", "breakdown", "components", "component", "product", "line"},
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
        return retriever, registry

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
        elif any(token in lowered for token in ("business model", "monetization", "product breadth")):
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
            allowed_source_types = self._default_source_types(lane)
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
        return normalized[: task.max_subquestions]

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
            result.trace.append(round_trace)

            if assessment.sufficient:
                subquestion.status = ResearchQuestionStatus.COMPLETED
                result.status = ResearchQuestionStatus.COMPLETED
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="complete",
                        question_id=subquestion.question_id,
                        reason="sufficient evidence reached",
                        payload={"round": total_rounds, "matched_chunk_ids": assessment.matched_chunk_ids},
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
                if subquestion.lane == ResearchLane.HARD_FACT and chunk.content_type in {"quantitative", "mixed"}:
                    support_score += 0.06
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
            sufficient = (
                len(valid_chunks) >= 1
                and (metric_match or numeric_match)
                and best_support >= 0.78
            )
            insufficient = not sufficient and (len(valid_chunks) < 1 or best_support < 0.45 or not (metric_match or numeric_match))
            if not metric_match and not numeric_match:
                reasons.append("Target metric or numeric evidence was not found in retrieved chunks.")
            if best_support < 0.78:
                reasons.append(f"Best support {best_support:.2f} below hard-fact threshold.")
        else:
            sufficient = (
                len(valid_chunks) >= 2
                and mean_support >= 0.58
                and high_trust_hit
            )
            insufficient = not sufficient and (len(valid_chunks) < 2 or best_support < 0.45)
            if len(valid_chunks) < 2:
                reasons.append("Fewer than two valid semantic evidence chunks were found.")
            if mean_support < 0.58:
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
        if subquestion.lane == ResearchLane.HARD_FACT:
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
        return deduped[:4]

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
        if assessment.conflict:
            return [
                f"{base} exact reported figure",
                f"{base} gaap adjusted reconciliation",
            ]
        if subquestion.lane == ResearchLane.HARD_FACT:
            return [
                f"{base} disclosed metric reported",
                f"{task.company_id} {period_hint} {subquestion.fact_slot} numeric evidence".strip(),
            ]
        return [
            f"{task.company_id} {period_hint} {subquestion.fact_slot} management commentary".strip(),
            f"{task.company_id} {period_hint} {subquestion.metric_family} source discussion".strip(),
        ]

    def _build_initial_queries(self, task: ResearchTask, subquestion: ResearchSubquestion) -> List[str]:
        period_hint = self._task_period_label(task, include_latest_placeholder=False)
        base = [subquestion.text]
        if subquestion.metric_family and subquestion.metric_family not in self._normalize_text(subquestion.text):
            base.append(f"{task.company_id} {period_hint} {subquestion.metric_family}".strip())
        if subquestion.fact_slot and subquestion.fact_slot not in self._normalize_text(subquestion.text):
            base.append(f"{task.company_id} {period_hint} {subquestion.fact_slot}".strip())
        return list(dict.fromkeys(item for item in base if item))

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
                self._generate_batch_answers(batch)
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
                if not item.answer_text:
                    item.answer_text = self._stub_answer_for_subquestion(item)
                self._verify_answer(item)
                replay.decisions.append(
                    ResearchDecisionRecord(
                        decision_type="answer_generated",
                        question_id=item.subquestion.question_id,
                        reason="batch answer generated",
                        payload={"lane": item.subquestion.lane.value},
                    )
                )

    def _generate_batch_answers(self, batch: List[ResearchQuestionResult]) -> None:
        prompt_lines = [
            "Return JSON with key `answers`, a list of objects.",
            "Each object must include question_id, status, claims, and insufficiency_reason.",
            "status must be either `answered` or `insufficient`.",
            "claims must be a list of at most 3 objects with keys: statement, chunk_id, source_id.",
            "Each statement must be exactly one evidence-bound sentence. No headings. No markdown emphasis. No uncited intro lines.",
            "Every claim must use exactly one chunk_id and one source_id from the provided evidence.",
            "Do not combine multiple facts into one statement.",
            "For hard-fact questions, use primary sources for numeric or financial claims. If only weaker evidence exists, return status=`insufficient`.",
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
                f"Evidence:\n{evidence}\n"
            )
        raw = generate_text_response(
            self.client,
            model=settings.openai_model,
            system_prompt=(
                "You write evidence-bound financial research answers as strict JSON. "
                "Only output supported single-sentence claims backed by the provided chunk IDs."
            ),
            user_prompt="\n".join(prompt_lines),
            max_tokens=1200,
            temperature=0.1,
            max_retries=settings.llm_request_max_retries,
        )
        payload = self._parse_json(raw)
        answers = payload.get("answers", []) if isinstance(payload, dict) else []
        answer_map = {
            str(item.get("question_id")): item
            for item in answers
            if isinstance(item, dict) and item.get("question_id")
        }
        for item in batch:
            structured = answer_map.get(item.subquestion.question_id)
            item.answer_text = self._render_structured_answer(item, structured)

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

    def _verify_answer(self, result: ResearchQuestionResult) -> None:
        result.answer_text = self._sanitize_answer_for_verification(result.answer_text)
        claims = self.claim_extractor.extract_claims(result.answer_text, result.subquestion.question_id)
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
        if not claims:
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
        verification_results = self.verifier.verify_memo(claims, evidence_store) if claims else []
        result.verified_claims = verification_results
        result.supported_content = self.report_writer._apply_verification_gating(
            result.answer_text,
            verification_results,
        )
        if verification_results and all(item.confidence == ConfidenceLevel.UNSUPPORTED for item in verification_results):
            result.status = ResearchQuestionStatus.REFUSED
            result.subquestion.status = ResearchQuestionStatus.REFUSED
            result.refusal_reason = "Generated answer did not survive verification."

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
                    chunk_id = str(claim.get("chunk_id", "")).strip()
                    source_id = str(claim.get("source_id", "")).strip()
                    chunk = evidence_map.get(chunk_id)
                    if not statement or chunk is None:
                        continue
                    if source_id != source_map.get(chunk_id):
                        continue
                    if (
                        result.subquestion.lane == ResearchLane.HARD_FACT
                        and self._claim_requires_primary_support(statement)
                        and not (chunk.is_primary or (chunk.trust_level or 0) >= 4)
                    ):
                        continue
                    if result.subquestion.lane == ResearchLane.HARD_FACT or self._claim_is_too_broad(statement):
                        derived_statement = self._derive_atomic_claim_from_chunk(result.subquestion, chunk)
                        if derived_statement:
                            statement = derived_statement
                    if self._claim_is_too_broad(statement):
                        continue
                    rendered_claims.append(
                        f"- {statement} [Chunk: {chunk_id}] [Source: {source_id}]"
                    )
            if rendered_claims:
                return "\n".join(rendered_claims)
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
        rewritten: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            chunk_match = re.search(r"\[Chunk:\s*([^\]]+)\]", line)
            source_match = re.search(r"\[Source:\s*([^\]]+)\]", line)
            if not chunk_match or not source_match:
                continue
            chunk_id = chunk_match.group(1).strip()
            source_id = source_match.group(1).strip()
            chunk = evidence_map.get(chunk_id)
            if chunk is None or source_id != chunk.source_id:
                continue
            statement = self._normalize_claim_sentence(line)
            if result.subquestion.lane == ResearchLane.HARD_FACT or self._claim_is_too_broad(statement):
                derived_statement = self._derive_atomic_claim_from_chunk(result.subquestion, chunk)
                if derived_statement:
                    statement = derived_statement
            if self._claim_is_too_broad(statement):
                continue
            if (
                result.subquestion.lane == ResearchLane.HARD_FACT
                and self._claim_requires_primary_support(statement)
                and not (chunk.is_primary or (chunk.trust_level or 0) >= 4)
            ):
                continue
            rewritten.append(f"- {statement} [Chunk: {chunk_id}] [Source: {source_id}]")
        return "\n".join(dict.fromkeys(rewritten)).strip()

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

    def _derive_atomic_claim_from_chunk(
        self,
        subquestion: ResearchSubquestion,
        chunk: RetrievedChunk,
    ) -> str:
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
            score = 0.0
            lowered = candidate.lower()
            score += self._slot_overlap_ratio(subquestion, candidate) * 10.0
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
        if self.use_stub_llm:
            return build_stub_executive_summary([item.supported_content for item in completed if item.supported_content])
        if not self._consume_llm_budget(llm_budget, reason="summary"):
            if self._strict_live_fail_fast():
                raise RuntimeError("LLM budget exhausted during strict live summary generation.")
            return build_stub_executive_summary([item.supported_content for item in completed if item.supported_content])

        content = "\n\n".join(
            f"{item.subquestion.text}\n{item.supported_content}"
            for item in completed
        )
        try:
            summary = generate_text_response(
                self.client,
                model=settings.openai_model,
                system_prompt="Summarize evidence-bound financial research findings in concise markdown bullets.",
                user_prompt=(
                f"Company: {task.company_id}\nPeriod: {self._task_period_label(task)}\n"
                    f"Research findings:\n{content}"
                ),
                max_tokens=250,
                temperature=0.1,
                max_retries=settings.llm_request_max_retries,
            ).strip()
            return summary or build_stub_executive_summary([content])
        except Exception:
            if self._strict_live_fail_fast():
                raise
            return build_stub_executive_summary([content])

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
        if proposed == "semantic" and not self._looks_like_numeric_fact_request(lowered):
            return ResearchLane.SEMANTIC
        if self._contains_any(lowered, SEMANTIC_KEYWORDS) and not self._looks_like_numeric_fact_request(lowered):
            return ResearchLane.SEMANTIC
        if self._contains_any(lowered, HARD_FACT_KEYWORDS):
            return ResearchLane.HARD_FACT
        if self._contains_any(lowered, SEMANTIC_KEYWORDS):
            return ResearchLane.SEMANTIC
        return ResearchLane.HARD_FACT

    def _looks_like_numeric_fact_request(self, lowered: str) -> bool:
        numeric_phrases = (
            "reported evidence",
            "disclosed",
            "numeric evidence",
            "exact reported figure",
            "what is ",
            "how much",
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
