"""
Hello-Agents 风格的外层控制器：
- TODO Planner Agent
- Task Summarizer Agent
- Report Writer Agent

底层 trust-first RAG 仍由 ResearchController 提供工具能力。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from agent.research_controller import ResearchController
from agent.schemas import (
    AnalysisMode,
    AnalysisPlan,
    AnalysisStep,
    ResearchDecisionRecord,
    ResearchLane,
    ResearchQuestionResult,
    ResearchQuestionStatus,
    ResearchReplayRecord,
    ResearchSubquestion,
)

logger = logging.getLogger(__name__)


@dataclass
class TodoPlannerAgent:
    controller: ResearchController

    def plan(self, task, llm_budget: dict, replay: ResearchReplayRecord) -> List[ResearchSubquestion]:
        subquestions = self.controller.plan_subquestions(task, llm_budget)
        replay.decisions.append(
            ResearchDecisionRecord(
                decision_type="todo_plan",
                question_id="plan",
                reason="planned subquestions from user query",
                payload={
                    "subquestion_ids": [item.question_id for item in subquestions],
                    "lanes": [item.lane.value for item in subquestions],
                },
            )
        )
        return subquestions


@dataclass
class TaskSummarizerAgent:
    controller: ResearchController

    def research_subquestion(
        self,
        *,
        task,
        subquestion: ResearchSubquestion,
        retriever,
        llm_budget: dict,
        replay: ResearchReplayRecord,
    ) -> ResearchQuestionResult:
        subquestion.status = ResearchQuestionStatus.RETRIEVING
        result = ResearchQuestionResult(
            subquestion=self.controller._clone_subquestion(subquestion),
            status=ResearchQuestionStatus.RETRIEVING,
        )
        queries = self.controller.initial_queries(task, subquestion)
        total_rounds = 0

        while total_rounds <= subquestion.max_rounds:
            total_rounds += 1
            subquestion.rounds_used = total_rounds
            round_trace = {"round": total_rounds, "queries": queries, "retrievals": [], "tool_calls": []}

            retrieved = self.controller.retrieve_evidence(
                task=task,
                subquestion=subquestion,
                queries=queries,
                retriever=retriever,
                round_trace=round_trace,
            )
            result.evidence = self.controller.dedupe_evidence(
                result.evidence + retrieved,
                task.max_evidence_per_question,
            )
            round_trace["tool_calls"].append(
                {
                    "tool_name": "retrieve_evidence",
                    "query_count": len(queries),
                    "returned_chunk_ids": [chunk.chunk_id for chunk in retrieved],
                }
            )
            replay.decisions.append(
                ResearchDecisionRecord(
                    decision_type="tool_call",
                    question_id=subquestion.question_id,
                    reason="retrieve_evidence",
                    payload={
                        "round": total_rounds,
                        "queries": queries,
                        "returned_chunk_ids": [chunk.chunk_id for chunk in retrieved],
                    },
                )
            )

            assessment = self.controller.assess_subquestion_evidence(
                task=task,
                subquestion=subquestion,
                chunks=result.evidence,
            )
            result.assessments.append(assessment)
            round_trace["tool_calls"].append(
                {
                    "tool_name": "assess_evidence",
                    "valid_chunk_count": assessment.valid_chunk_count,
                    "sufficient": assessment.sufficient,
                    "conflict": assessment.conflict,
                    "reasons": assessment.reasons,
                }
            )
            replay.decisions.append(
                ResearchDecisionRecord(
                    decision_type="tool_call",
                    question_id=subquestion.question_id,
                    reason="assess_evidence",
                    payload={
                        "round": total_rounds,
                        "valid_chunk_count": assessment.valid_chunk_count,
                        "sufficient": assessment.sufficient,
                        "conflict": assessment.conflict,
                        "matched_chunk_ids": assessment.matched_chunk_ids,
                    },
                )
            )
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

            queries = self.controller.plan_followup_queries(
                task=task,
                subquestion=subquestion,
                assessment=assessment,
                llm_budget=llm_budget,
            )
            result.followup_queries = queries
            round_trace["tool_calls"].append(
                {
                    "tool_name": "plan_followup_queries",
                    "queries": queries,
                }
            )
            replay.decisions.append(
                ResearchDecisionRecord(
                    decision_type="tool_call",
                    question_id=subquestion.question_id,
                    reason="plan_followup_queries",
                    payload={"round": total_rounds, "queries": queries},
                )
            )

        result.subquestion = self.controller._clone_subquestion(subquestion)
        return result


@dataclass
class ReportWriterAgent:
    controller: ResearchController

    def compose(
        self,
        *,
        task,
        question_results: List[ResearchQuestionResult],
        source_registry: Dict[str, dict],
        llm_budget: dict,
        replay: ResearchReplayRecord,
    ):
        self.controller.generate_subquestion_answers(
            task=task,
            results=question_results,
            llm_budget=llm_budget,
            replay=replay,
        )
        replay.decisions.append(
            ResearchDecisionRecord(
                decision_type="report_write",
                question_id="report",
                reason="build final memo from subquestion results",
                payload={"completed": sum(1 for item in question_results if item.status == ResearchQuestionStatus.COMPLETED)},
            )
        )
        return self.controller.build_memo_from_results(
            task=task,
            results=question_results,
            source_registry=source_registry,
            replay=replay,
            llm_budget=llm_budget,
        )


class HelloDeepResearchAgent:
    """
    参考 hello-agents 的 DeepResearch 结构：
    planner -> task summarizer -> report writer

    其中检索/验证/证据护栏全部下沉为 trust-first RAG 工具调用。
    """

    def __init__(self, controller: ResearchController):
        self.controller = controller
        self.planner = TodoPlannerAgent(controller)
        self.task_summarizer = TaskSummarizerAgent(controller)
        self.report_writer_agent = ReportWriterAgent(controller)

    def research(
        self,
        query: str,
        mode: Optional[AnalysisMode] = None,
        *,
        company_id: Optional[str] = None,
        period: Optional[str] = None,
        target_periods: Optional[List[str]] = None,
        required_slots: Optional[List[str]] = None,
        required_source_types: Optional[List[str]] = None,
        query_types: Optional[List[str]] = None,
    ) -> dict:
        task = self.controller.build_task(
            query=query,
            mode=mode or AnalysisMode.COMPANY,
            company_id=company_id,
            period=period,
            target_periods=target_periods,
            required_slots=required_slots,
            required_source_types=required_source_types,
            query_types=query_types,
        )
        replay = ResearchReplayRecord(task=task)
        llm_budget = {"used": 0, "max": task.llm_call_budget}

        if task.mode == AnalysisMode.COMPETITIVE or len(task.mentioned_company_ids) > 1:
            return self.controller.build_unsupported_scope_result(task, replay)

        subquestions = self.planner.plan(task, llm_budget, replay)
        replay.subquestions = [self.controller._clone_subquestion(item) for item in subquestions]
        retriever, source_registry = self.controller.prepare_runtime(task)

        workflow_events: List[dict] = [
            {"node_name": "hello_deep_research_agent", "event_type": "started", "detail": task.query},
            {"node_name": "todo_planner", "event_type": "completed", "detail": f"{len(subquestions)} subquestions"},
        ]

        hard_fact_questions = [item for item in subquestions if item.lane == ResearchLane.HARD_FACT]
        semantic_questions = [item for item in subquestions if item.lane == ResearchLane.SEMANTIC]
        question_results: List[ResearchQuestionResult] = []

        for group_name, questions in (("hard_fact_group", hard_fact_questions), ("semantic_group", semantic_questions)):
            if not questions:
                continue
            workflow_events.append({"node_name": group_name, "event_type": "started", "detail": f"{len(questions)} subquestions"})
            for subquestion in questions:
                result = self.task_summarizer.research_subquestion(
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

        memo = self.report_writer_agent.compose(
            task=task,
            question_results=question_results,
            source_registry=source_registry,
            llm_budget=llm_budget,
            replay=replay,
        )
        replay.subquestions = [self.controller._clone_subquestion(item.subquestion) for item in question_results]
        replay.llm_calls_used = llm_budget["used"]
        memo_markdown = self.controller.report_writer.render_memo(memo)
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
        workflow_events.append({"node_name": "report_writer", "event_type": "completed", "detail": memo.title})
        workflow_events.append({"node_name": "hello_deep_research_agent", "event_type": "completed", "detail": memo.title})

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
