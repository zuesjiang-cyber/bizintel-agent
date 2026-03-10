"""
分析执行器

职责：
- 接收 AnalysisPlan
- 为每个 step 创建 WorkflowNode（包含实际执行逻辑）
- 通过 WorkflowEngine 执行整个计划
- 收集所有中间结果

这个文件是 planner 和 workflow_engine 的桥梁。
"""

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from agent.config import settings
from agent.schemas import (
    AnalysisPlan, AnalysisStep, RetrievedChunk,
)
from agent.workflow_engine import WorkflowEngine, WorkflowNode
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class AnalysisExecutor:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        # Do not initialize client here; settings might not have the key yet
        self.client = None

    def _get_client(self):
        if self.client is None:
            self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_base)
        return self.client

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
            # 1. 检索
            all_retrieved = []
            for query in step.search_queries:
                retrieved = self.retriever.retrieve(query, top_k=5)
                all_retrieved.extend(retrieved)

            # 去重
            seen_ids = set()
            unique_retrieved = []
            for chunk in all_retrieved:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    unique_retrieved.append(chunk)

            # 2. 构建 context
            context = self._build_context(unique_retrieved)

            # 3. 获取之前步骤的结果（用于上下文连贯）
            previous_findings = self._get_previous_findings(shared_state)

            # 4. LLM 分析
            analysis = self._analyze_with_llm(
                step=step,
                user_query=user_query,
                mode=mode,
                context=context,
                previous_findings=previous_findings,
            )

            return {
                "content": analysis,
                "sources_used": [
                    {"source_id": c.source_id, "chunk_id": c.chunk_id, "score": c.score}
                    for c in unique_retrieved[:10]
                ],
                "raw_evidence": [
                    {"chunk_id": c.chunk_id, "text": c.text, "source_id": c.source_id}
                    for c in unique_retrieved[:10]
                ],
            }

        return executor

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """将检索到的 chunks 格式化为 LLM 的 context"""
        if not chunks:
            return "No relevant information found."

        parts = []
        for i, chunk in enumerate(chunks[:10]):  # 最多 10 个 chunk
            parts.append(f"[Source: {chunk.source_id}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _get_previous_findings(self, shared_state: dict) -> str:
        """提取之前步骤的关键发现"""
        findings = []
        for key, value in shared_state.items():
            if key == "plan":
                continue
            if isinstance(value, dict) and "content" in value:
                findings.append(f"### {key}\n{value['content'][:500]}")

        if not findings:
            return ""
        return "Previous findings:\n\n" + "\n\n".join(findings)

    def _analyze_with_llm(
        self,
        step: AnalysisStep,
        user_query: str,
        mode: str,
        context: str,
        previous_findings: str,
    ) -> str:
        prompt = f"""You are a business research analyst. You are conducting a {mode} analysis.

User's research question: "{user_query}"

Current analysis step: {step.name}
Step objective: {step.description}

{f"Previous analysis steps have found:" + chr(10) + previous_findings if previous_findings else ""}

Available evidence:
{context}

Instructions:
1. Analyze the evidence above to address the step objective
2. Be specific and cite sources using [Source: source_id] format
3. If the evidence is insufficient, say so explicitly rather than making things up
4. Include specific numbers, dates, and facts when available
5. Keep the analysis focused and structured

Write the analysis for this section:"""

        client = self._get_client()
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()
