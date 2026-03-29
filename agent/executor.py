"""
分析执行器

职责：
- 接收 AnalysisPlan
- 为每个 step 创建 WorkflowNode（包含实际执行逻辑）
- 通过 WorkflowEngine 执行整个计划
- 收集所有中间结果

这个文件是 planner 和 workflow_engine 的桥梁。
"""

import logging
import json
import re
from typing import Any, Dict, List

from agent.config import settings
from agent.llm_utils import (
    build_stub_section_analysis,
    should_use_stub_llm,
)
from agent.schemas import (
    AnalysisPlan, AnalysisStep, RetrievedChunk,
)
from agent.workflow_engine import WorkflowEngine, WorkflowNode
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class AnalysisExecutor:
    def __init__(self, retriever: HybridRetriever, demo_mode: bool = False):
        self.retriever = retriever
        self.use_stub_llm = demo_mode or should_use_stub_llm(settings.llm_mode, settings.openai_api_key)

    def execute_plan(self, plan: AnalysisPlan) -> Dict[str, Any]:
        """
        执行分析计划，返回每个步骤的结果

        返回: {
            "company_profile": {"content": "...", "sources_used": [...], "raw_evidence": [...]},
            "business_model": {...},
            ...
        }
        """
        # 将 AnalysisStep 转化为 WorkflowNode
        nodes = []
        for step in plan.steps:
            node = WorkflowNode(
                name=step.name,
                executor=self._make_step_executor(step, plan.user_query, plan.mode.value),
                required=step.required,
                max_retries=settings.default_max_retries,
                timeout_seconds=settings.default_timeout_seconds,
            )
            nodes.append(node)

        # 执行
        engine = WorkflowEngine(nodes)
        results = engine.execute(initial_state={"plan": plan})

        # 提取结果
        step_outputs = {}
        for step_name, result in results.items():
            if result.data is not None:
                step_outputs[step_name] = result.data
            else:
                step_outputs[step_name] = {
                    "content": f"[This section could not be completed: {result.error}]",
                    "sources_used": [],
                    "raw_evidence": [],
                    "status": result.status.value,
                }

        return {
            "step_outputs": step_outputs,
            "workflow_events": [vars(e) for e in engine.events],
        }

    def _make_step_executor(
        self, step: AnalysisStep, user_query: str, mode: str
    ):
        """
        为一个 AnalysisStep 创建执行函数

        执行流程：
        1. 用步骤的 search_queries 做检索
        2. 将检索结果作为 context 传给 LLM
        3. LLM 根据步骤要求生成分析
        4. 返回分析结果 + 使用的来源
        """
        def executor(shared_state: dict) -> dict:
            section_requirements = step.evidence_requirements or {}
            previous_findings = self._get_previous_findings(shared_state)
            loop_result = self._run_gap_filling_loop(
                step=step,
                user_query=user_query,
                section_requirements=section_requirements,
            )
            unique_retrieved = loop_result["retrieved_chunks"]
            retrieval_trace = loop_result["retrieval_trace"]
            context = self._build_context(unique_retrieved)

            evidence_ledger = loop_result["evidence_ledger"]
            evidence_notes = self._ledger_to_notes(evidence_ledger, unique_retrieved)
            analysis = self._analyze_with_llm(
                step=step,
                user_query=user_query,
                mode=mode,
                contract=shared_state.get("plan").contract if shared_state.get("plan") else {},
                section_requirements=section_requirements,
                retrieved_chunks=unique_retrieved,
                context=context,
                previous_findings=previous_findings,
                evidence_notes=evidence_notes,
            )

            return {
                "content": analysis,
                "sources_used": [
                    {"source_id": c.source_id, "chunk_id": c.chunk_id, "score": c.score}
                    for c in unique_retrieved[:10]
                ],
                "query_contracts": step.query_contracts,
                "section_outline": {
                    "step_name": step.name,
                    "objective": step.description,
                    "writing_order": section_requirements.get("writing_order", ["facts", "management_commentary", "inference"]),
                    "fact_first": section_requirements.get("fact_first", False),
                    "required_facts": section_requirements.get("required_facts", []),
                },
                "raw_evidence": [
                    {"chunk_id": c.chunk_id, "text": c.text, "source_id": c.source_id}
                    for c in unique_retrieved[:10]
                ],
                "evidence_ledger": evidence_ledger,
                "evidence_notes": evidence_notes,
                "retrieval_trace": retrieval_trace,
                "generation_context": context,
                "previous_findings": previous_findings,
                "covered_facts": loop_result["covered_facts"],
                "missing_facts": loop_result["missing_facts"],
                "gap_reflection": loop_result["gap_reflection"],
                "gap_iterations": loop_result["iterations"],
            }

        return executor

    def _run_gap_filling_loop(
        self,
        *,
        step: AnalysisStep,
        user_query: str,
        section_requirements: dict,
    ) -> dict:
        required_facts = list(section_requirements.get("required_facts", []))
        remaining_facts = list(required_facts)
        iterations = []
        retrieval_trace = []
        unique_chunks: Dict[str, RetrievedChunk] = {}
        queries = list(step.search_queries)
        latest_ledger: List[dict] = []
        latest_gap_reflection = {
            "gaps": [],
            "confidence_assessments": [],
            "additional_queries": [],
            "should_continue": False,
            "stop_reason": "resolved",
        }

        for round_idx in range(max(1, int(settings.retrieval_gap_max_rounds))):
            round_retrieved: List[RetrievedChunk] = []
            for query in queries:
                retrieval_filters = self._filters_for_query(step, query)
                retrieved, trace = self.retriever.retrieve_with_trace(
                    query,
                    top_k=settings.retrieval_top_k,
                    mode=settings.retrieval_mode,
                    filters=retrieval_filters,
                )
                round_retrieved.extend(retrieved)
                retrieval_trace.append(trace)

            new_chunks = 0
            for chunk in round_retrieved:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk
                    new_chunks += 1

            latest_ledger = self._build_evidence_ledger(list(unique_chunks.values()), section_requirements)
            covered_facts = self._determine_covered_facts(required_facts, latest_ledger)
            remaining_facts = [fact for fact in required_facts if fact not in covered_facts]
            latest_gap_reflection = self._build_gap_reflection(
                user_query=user_query,
                step=step,
                missing_facts=remaining_facts,
                evidence_ledger=latest_ledger,
            )
            iterations.append(
                {
                    "round": round_idx + 1,
                    "queries": queries,
                    "new_chunks": new_chunks,
                    "covered_facts": covered_facts,
                    "missing_facts": remaining_facts,
                    "gap_reflection": latest_gap_reflection,
                }
            )

            if not remaining_facts:
                break
            if new_chunks == 0:
                break

            queries = self._build_gap_queries(user_query, step, remaining_facts, latest_gap_reflection)

        return {
            "retrieved_chunks": list(unique_chunks.values()),
            "retrieval_trace": retrieval_trace,
            "evidence_ledger": latest_ledger,
            "covered_facts": self._determine_covered_facts(required_facts, latest_ledger),
            "missing_facts": remaining_facts,
            "gap_reflection": latest_gap_reflection,
            "iterations": iterations,
        }

    def _filters_for_query(self, step: AnalysisStep, query: str) -> dict:
        normalized_query = (query or "").strip().lower()
        for contract in step.query_contracts or []:
            if (contract.get("query_text") or "").strip().lower() != normalized_query:
                continue
            filters = contract.get("filters") or {}
            return {
                "companies": list(filters.get("companies", [])),
                "periods": list(filters.get("periods", [])),
                "source_types": list(filters.get("source_types", [])),
            }
        return {"companies": [], "periods": [], "source_types": []}

    def _build_gap_queries(
        self,
        user_query: str,
        step: AnalysisStep,
        missing_facts: List[str],
        gap_reflection: dict,
    ) -> List[str]:
        if not missing_facts:
            return list(step.search_queries)
        reflected = [
            item.get("query_text", "")
            for item in gap_reflection.get("additional_queries", [])
            if isinstance(item, dict) and item.get("query_text")
        ]
        if reflected:
            return reflected[:3]
        gap_queries = []
        for fact in missing_facts[:2]:
            gap_queries.append(f"{user_query} {step.name.replace('_', ' ')} {fact}")
        if not gap_queries:
            return list(step.search_queries)
        return gap_queries

    def _build_evidence_ledger(self, chunks: List[RetrievedChunk], section_requirements: dict) -> List[dict]:
        ledger = []
        for fact_slot in section_requirements.get("required_facts", []):
            tokens = [
                token for token in re.findall(r"[a-z0-9]+", fact_slot.lower())
                if len(token) > 2
            ]
            matching_chunks = [
                chunk for chunk in chunks
                if any(token in chunk.text.lower() for token in tokens)
            ]
            if matching_chunks:
                best = matching_chunks[0]
                reasoning_type = "numeric_calculation" if any(char.isdigit() for char in best.text) else "direct_quote"
                ledger.append(
                    {
                        "fact_slot": fact_slot,
                        "support_status": "supported",
                        "extracted_fact": self._best_excerpt_for_fact(best.text, fact_slot),
                        "evidence_ids": [best.chunk_id],
                        "reasoning_type": reasoning_type,
                        "uncertainty_note": "",
                    }
                )
            else:
                ledger.append(
                    {
                        "fact_slot": fact_slot,
                        "support_status": "unsupported",
                        "extracted_fact": "",
                        "evidence_ids": [],
                        "reasoning_type": "inference",
                        "uncertainty_note": "No retrieved chunk directly supports this fact slot.",
                    }
                )
        return ledger

    def _build_gap_reflection(
        self,
        user_query: str,
        step: AnalysisStep,
        missing_facts: List[str],
        evidence_ledger: List[dict],
    ) -> dict:
        default = {
            "gaps": [{"fact_slot": fact, "issue": "No direct supporting evidence yet."} for fact in missing_facts],
            "confidence_assessments": [
                {
                    "fact_slot": row["fact_slot"],
                    "confidence": 85 if row["support_status"] == "supported" else 40 if row["support_status"] == "partially_supported" else 0,
                    "assumption": "Confidence depends on whether the retrieved evidence directly states the fact slot.",
                }
                for row in evidence_ledger
            ],
            "additional_queries": [
                {
                    "fact_slot": fact,
                    "query_text": f"{user_query} {step.name.replace('_', ' ')} {fact}",
                    "source_type": step.evidence_requirements.get("allowed_source_types", []),
                    "filters": {},
                }
                for fact in missing_facts[:2]
            ],
            "should_continue": bool(missing_facts),
            "stop_reason": "resolved" if not missing_facts else "no_new_angle",
        }
        return default

    def _default_fact_slot(self, step: AnalysisStep) -> str:
        facts = step.evidence_requirements.get("required_facts", [])
        return str(facts[0]) if facts else "required_fact"

    def _ledger_to_notes(self, evidence_ledger: List[dict], chunks: List[RetrievedChunk]) -> List[dict]:
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        notes = []
        for row in evidence_ledger:
            extracted_fact = row.get("extracted_fact", "")
            if not extracted_fact:
                continue
            chunk_id = row.get("evidence_ids", [""])[0] if row.get("evidence_ids") else ""
            chunk = chunk_map.get(chunk_id)
            evidence_type = "fact"
            if row.get("reasoning_type") == "inference":
                evidence_type = "inference"
            elif "commentary" in row.get("fact_slot", "").lower():
                evidence_type = "management_commentary"
            notes.append(
                {
                    "claim": extracted_fact,
                    "chunk_id": chunk_id,
                    "source_id": chunk.source_id if chunk else "unknown_source",
                    "evidence_text": self._clean_chunk_excerpt(chunk.text) if chunk else extracted_fact,
                    "evidence_type": evidence_type,
                    "direct_support": row.get("support_status") == "supported",
                    "priority": 2 if row.get("support_status") == "supported" else 1,
                    "fact_slot": row.get("fact_slot", ""),
                }
            )
        notes.sort(key=lambda row: (row["priority"], row["direct_support"]), reverse=True)
        return notes[:8]

    def _determine_covered_facts(self, required_facts: List[str], evidence_ledger: List[dict]) -> List[str]:
        if not required_facts:
            return []
        return [
            row["fact_slot"]
            for row in evidence_ledger
            if row.get("fact_slot") in required_facts and row.get("support_status") in {"supported", "partially_supported"}
        ]

    def _extract_note_sentences(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        sentences = [
            sentence.strip(" -")
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]
        if sentences:
            return [sentence for sentence in sentences if len(sentence) >= 20]
        return [cleaned[:240]]

    def _best_excerpt_for_fact(self, text: str, fact_slot: str) -> str:
        cleaned = self._clean_chunk_excerpt(text, max_chars=360)
        if not cleaned:
            return ""

        tokens = [
            token for token in re.findall(r"[a-z0-9]+", fact_slot.lower())
            if len(token) > 2
        ]
        candidates = []
        for sentence in self._extract_note_sentences(cleaned):
            lowered = sentence.lower()
            score = sum(1 for token in tokens if token in lowered)
            if score:
                candidates.append((score, len(sentence), sentence))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2]
        sentences = self._extract_note_sentences(cleaned)
        if sentences:
            return sentences[0]
        return cleaned

    def _clean_chunk_excerpt(self, text: str, max_chars: int = 280) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip(" -")
        if len(cleaned) <= max_chars:
            return cleaned
        return f"{cleaned[: max_chars - 3].rstrip()}..."

    def _classify_evidence_type(self, sentence: str) -> str:
        lowered = sentence.lower()
        if any(token in lowered for token in {"said", "noted", "management", "ceo", "cfo", "call", "commentary"}):
            return "management_commentary"
        if any(token in lowered for token in {"may", "could", "suggest", "imply", "appears"}):
            return "inference"
        return "fact"

    def _parse_json_payload(self, content: str):
        text = content.strip()
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text)

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """将检索到的 chunks 格式化为 LLM 的 context"""
        if not chunks:
            return "No relevant information found."

        parts = []
        for i, chunk in enumerate(chunks[:10]):  # 最多 10 个 chunk
            parts.append(f"[Chunk: {chunk.chunk_id}] [Source: {chunk.source_id}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _get_previous_findings(self, shared_state: dict) -> str:
        """提取之前步骤的关键发现"""
        findings = []
        for key, value in shared_state.items():
            if key == "plan":
                continue
            if isinstance(value, dict) and "content" in value:
                evidence_summary = value.get("evidence_notes") or []
                if evidence_summary:
                    snippets = [
                        f"- {note['claim']} [Source: {note['source_id']}]"
                        for note in evidence_summary[:2]
                    ]
                    findings.append(f"### {key}\n" + "\n".join(snippets))
                else:
                    findings.append(f"### {key}\n{value['content'][:500]}")

        if not findings:
            return ""
        return "Previous findings:\n\n" + "\n\n".join(findings)

    def _analyze_with_llm(
        self,
        step: AnalysisStep,
        user_query: str,
        mode: str,
        contract: dict,
        section_requirements: dict,
        retrieved_chunks: List[RetrievedChunk],
        context: str,
        previous_findings: str,
        evidence_notes: List[dict],
    ) -> str:
        del mode, contract, context, previous_findings
        if evidence_notes:
            lines = []
            for note in evidence_notes[:3]:
                lines.append(
                    f"- {note['claim']} [Chunk: {note['chunk_id']}] [Source: {note['source_id']}]"
                )
            if section_requirements.get("required_facts"):
                covered = {note.get("fact_slot", "") for note in evidence_notes if note.get("direct_support")}
                missing = [
                    fact for fact in section_requirements["required_facts"]
                    if fact not in covered
                ]
                if missing:
                    lines.append(
                        f"- Insufficient evidence to directly verify: {', '.join(missing[:2])}."
                    )
            return "\n".join(lines)
        return build_stub_section_analysis(
            step=step,
            user_query=user_query,
            chunks=retrieved_chunks,
            previous_findings="",
        )
