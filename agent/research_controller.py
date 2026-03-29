import json
import logging
import re
from typing import Dict, List, Optional, Sequence

from agent.config import settings
from agent.llm_utils import (
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

    def _build_task(self, query: str, mode: AnalysisMode) -> ResearchTask:
        mentioned_company_ids = self._infer_company_ids(query)
        company_id = mentioned_company_ids[0] if mentioned_company_ids else self._infer_company_id(query)
        period = self._infer_period(query)
        return ResearchTask(
            query=query,
            company_id=company_id,
            period=period,
            mode=mode,
            mentioned_company_ids=mentioned_company_ids,
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
        patterns = [
            (r"\bq([1-4])\s*[-/]?\s*(20\d{2})\b", lambda m: f"{m.group(2)}Q{m.group(1)}"),
            (r"\bfy\s*(20\d{2})\b", lambda m: f"{m.group(1)}FY"),
            (r"\b(20\d{2})\s*[年]?\s*第?([一二三四1-4])\s*季", lambda m: f"{m.group(1)}Q{self._cn_quarter(m.group(2))}"),
        ]
        for pattern, formatter in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return formatter(match)
        return None

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
        return self._normalize_subquestions(task, proposals)

    def _llm_decompose_subquestions(self, task: ResearchTask) -> List[dict]:
        prompt = (
            "Return JSON with a top-level key `subquestions`.\n"
            "Each item must include: text, lane, priority, fact_slot, metric_family, needs_numeric_verification.\n"
            "Rules: at most 6 items; split mixed research questions into atomic subquestions; "
            "numerical and period-specific questions must be lane=hard_fact; "
            "risk, tone, strategy, drivers must be lane=semantic.\n\n"
            f"Company: {task.company_id}\n"
            f"Target period: {task.period or 'latest available in pack'}\n"
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
                max_retries=0,
            )
            payload = self._parse_json(raw)
            if isinstance(payload, dict) and isinstance(payload.get("subquestions"), list):
                return payload["subquestions"]
        except Exception as exc:
            logger.warning("Falling back to rule-based decomposition because LLM decomposition failed: %s", exc)
        return []

    def _fallback_subquestions(self, task: ResearchTask) -> List[dict]:
        query_lower = task.query.lower()
        proposals: List[dict] = []

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
            if any(token in query_lower for token in ("management", "管理层", "outlook", "展望", "tone", "attitude", "态度")):
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
            priority = max(1, min(task.max_subquestions, int(proposal.get("priority") or idx)))
            needs_numeric = bool(proposal.get("needs_numeric_verification") or lane == ResearchLane.HARD_FACT)
            allowed_source_types = self._default_source_types(lane)
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
                    required_period=task.period,
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
                    "periods": [task.period] if task.period else [],
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
                        "periods": [task.period] if task.period else [],
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
        if task.period:
            valid_chunks = [
                chunk for chunk in valid_chunks
                if chunk.period and self._normalize_text(chunk.period) == self._normalize_text(task.period)
            ]
        if not valid_chunks:
            return EvidenceAssessment(
                valid_chunk_count=0,
                insufficient=True,
                reasons=["No contract-valid evidence chunks were retrieved."],
            )

        scored = self.verifier.score_hypothesis_against_chunks(
            subquestion.text,
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "text": chunk.text,
                    "source_type": chunk.source_type,
                    "is_primary": chunk.is_primary,
                }
                for chunk in valid_chunks
            ],
        )
        scored.sort(key=lambda item: item["entailment"], reverse=True)
        best_entailment = scored[0]["entailment"] if scored else 0.0
        mean_entailment = sum(item["entailment"] for item in scored[:2]) / max(1, min(2, len(scored)))
        matched_chunk_ids = [item["chunk_id"] for item in scored[:2]]

        metric_match = any(
            subquestion.metric_family in (chunk.metric_signals or [])
            or subquestion.metric_family in chunk.text.lower()
            for chunk in valid_chunks
        )
        numeric_match = any(chunk.content_type in {"quantitative", "mixed"} for chunk in valid_chunks)
        high_trust_hit = any((chunk.trust_level or 0) >= 4 or chunk.is_primary for chunk in valid_chunks)
        conflict = self._detect_conflict(subquestion, scored)
        sufficient = False
        insufficient = False
        reasons: List[str] = []

        if subquestion.lane == ResearchLane.HARD_FACT:
            sufficient = (
                len(valid_chunks) >= 1
                and (metric_match or numeric_match)
                and best_entailment >= 0.80
            )
            insufficient = not sufficient and (len(valid_chunks) < 1 or best_entailment < 0.50 or not (metric_match or numeric_match))
            if not metric_match and not numeric_match:
                reasons.append("Target metric or numeric evidence was not found in retrieved chunks.")
            if best_entailment < 0.80:
                reasons.append(f"Best entailment {best_entailment:.2f} below hard-fact threshold.")
        else:
            sufficient = (
                len(valid_chunks) >= 2
                and mean_entailment >= 0.60
                and high_trust_hit
            )
            insufficient = not sufficient and (len(valid_chunks) < 2 or best_entailment < 0.50)
            if len(valid_chunks) < 2:
                reasons.append("Fewer than two valid semantic evidence chunks were found.")
            if mean_entailment < 0.60:
                reasons.append(f"Mean entailment {mean_entailment:.2f} below semantic threshold.")
            if not high_trust_hit:
                reasons.append("No high-trust source supported the semantic finding.")

        if conflict:
            reasons.append("Conflicting evidence detected across top supporting chunks.")

        return EvidenceAssessment(
            valid_chunk_count=len(valid_chunks),
            best_entailment=best_entailment,
            mean_entailment=mean_entailment,
            numeric_match=numeric_match,
            metric_match=metric_match,
            high_trust_hit=high_trust_hit,
            sufficient=sufficient and not conflict,
            insufficient=insufficient and not conflict,
            conflict=conflict,
            reasons=reasons,
            matched_chunk_ids=matched_chunk_ids,
        )

    def _detect_conflict(self, subquestion: ResearchSubquestion, scored: List[dict]) -> bool:
        if len(scored) < 2:
            return False
        top = [item for item in scored[:3] if item["entailment"] >= 0.60]
        if len(top) < 2:
            return False
        if subquestion.lane != ResearchLane.HARD_FACT:
            contradiction = any(item["contradiction"] >= 0.60 for item in top)
            return contradiction
        first_numbers = self._extract_number_tokens(top[0]["text"])
        second_numbers = self._extract_number_tokens(top[1]["text"])
        if first_numbers and second_numbers and first_numbers != second_numbers:
            return True
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
            f"Period: {task.period or 'latest'}\n"
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
                max_retries=0,
            )
            payload = self._parse_json(raw)
            if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
                return [str(query).strip() for query in payload["queries"] if str(query).strip()][:2]
        except Exception as exc:
            logger.warning("Follow-up reflection failed; using rule-based fallback: %s", exc)
        return []

    def _rule_followup_queries(
        self,
        task: ResearchTask,
        subquestion: ResearchSubquestion,
        assessment: EvidenceAssessment,
    ) -> List[str]:
        base = f"{task.company_id} {task.period or ''} {subquestion.metric_family}".strip()
        if assessment.conflict:
            return [
                f"{base} exact reported figure",
                f"{base} gaap adjusted reconciliation",
            ]
        if subquestion.lane == ResearchLane.HARD_FACT:
            return [
                f"{base} disclosed metric reported",
                f"{task.company_id} {task.period or ''} {subquestion.fact_slot} numeric evidence",
            ]
        return [
            f"{task.company_id} {task.period or ''} {subquestion.fact_slot} management commentary",
            f"{task.company_id} {task.period or ''} {subquestion.metric_family} source discussion",
        ]

    def _build_initial_queries(self, task: ResearchTask, subquestion: ResearchSubquestion) -> List[str]:
        base = [subquestion.text]
        if subquestion.metric_family and subquestion.metric_family not in self._normalize_text(subquestion.text):
            base.append(f"{task.company_id} {task.period or ''} {subquestion.metric_family}".strip())
        if subquestion.fact_slot and subquestion.fact_slot not in self._normalize_text(subquestion.text):
            base.append(f"{task.company_id} {task.period or ''} {subquestion.fact_slot}".strip())
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
                for item in batch:
                    item.answer_text = self._stub_answer_for_subquestion(item)
                    self._verify_answer(item)
                continue
            self._generate_batch_answers(batch)
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
            "Each object must include question_id and answer.",
            "Every answer must cite evidence using both [Chunk: ...] and [Source: ...].",
            "Do not answer with facts not present in the provided evidence.",
        ]
        for item in batch:
            evidence = "\n".join(
                f"[Chunk: {chunk.chunk_id}] [Source: {chunk.source_id}] {chunk.text}"
                for chunk in item.evidence[:8]
            )
            prompt_lines.append(
                f"\nQuestion ID: {item.subquestion.question_id}\n"
                f"Question: {item.subquestion.text}\n"
                f"Evidence:\n{evidence}\n"
            )
        raw = generate_text_response(
            self.client,
            model=settings.openai_model,
            system_prompt="You write evidence-bound financial research answers in concise markdown.",
            user_prompt="\n".join(prompt_lines),
            max_tokens=1200,
            temperature=0.1,
            max_retries=0,
        )
        payload = self._parse_json(raw)
        answers = payload.get("answers", []) if isinstance(payload, dict) else []
        answer_map = {
            str(item.get("question_id")): str(item.get("answer", "")).strip()
            for item in answers
            if isinstance(item, dict) and item.get("question_id")
        }
        for item in batch:
            item.answer_text = answer_map.get(item.subquestion.question_id, "").strip()

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
                "query_type": "comparison" if task.mode == AnalysisMode.COMPETITIVE else "standard",
                "subquestion_count": len(results),
                "required_slots": [item.subquestion.fact_slot for item in results if item.subquestion.required],
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
            required_period=task.period,
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
        if self.use_stub_llm or not self._consume_llm_budget(llm_budget, reason="summary"):
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
                    f"Company: {task.company_id}\nPeriod: {task.period or 'latest'}\n"
                    f"Research findings:\n{content}"
                ),
                max_tokens=250,
                temperature=0.1,
                max_retries=0,
            ).strip()
            return summary or build_stub_executive_summary([content])
        except Exception:
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
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

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
        if self._contains_any(lowered, HARD_FACT_KEYWORDS):
            return ResearchLane.HARD_FACT
        if self._contains_any(lowered, SEMANTIC_KEYWORDS):
            return ResearchLane.SEMANTIC
        if str(proposed_lane or "").strip().lower() == "semantic":
            return ResearchLane.SEMANTIC
        return ResearchLane.HARD_FACT

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

    @staticmethod
    def _cn_quarter(value: str) -> str:
        return {"一": "1", "二": "2", "三": "3", "四": "4"}.get(value, value)
