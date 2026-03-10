"""
主编排器 — 整个 Agent 的入口

用户输入 → 分类 → 规划 → 执行 → 验证 → 报告

这个文件是唯一需要对外暴露的接口。
"""

import json
import logging
from pathlib import Path
from typing import Optional

from agent.config import settings
from agent.schemas import AnalysisMode, AnalysisPlan, GeneratedMemo, DocumentMeta
from agent.planner import Planner
from agent.executor import AnalysisExecutor
from agent.report_writer import ReportWriter
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class BizIntelAgent:
    def __init__(self, index_dir: Optional[Path] = None):
        logger.info("Initializing BizIntel Agent...")

        # 加载检索引擎
        self.retriever = HybridRetriever(
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
        )

        idx_dir = index_dir or (settings.data_dir / "index")
        if idx_dir.exists():
            self.retriever.load_index(idx_dir)
            logger.info("Index loaded.")
        else:
            logger.warning(f"No index found at {idx_dir}. Run build_index first.")

        # 初始化组件
        self.planner = Planner()
        self.executor = AnalysisExecutor(self.retriever)
        self.report_writer = ReportWriter()

        logger.info("BizIntel Agent ready.")

    def research(
        self,
        query: str,
        mode: Optional[AnalysisMode] = None,
    ) -> dict:
        """
        执行完整研究流程

        返回: {
            "memo_markdown": str,        # 渲染后的 memo
            "memo_object": GeneratedMemo, # 结构化 memo 对象
            "plan": AnalysisPlan,         # 使用的分析计划
            "workflow_events": list,      # 工作流事件日志
        }
        """
        logger.info(f"Starting research: '{query}' (mode={mode})")

        # Step 1: Planning
        logger.info("Step 1: Creating analysis plan...")
        plan = self.planner.create_plan(query, mode=mode)
        logger.info(f"  Mode: {plan.mode.value}")
        logger.info(f"  Steps: {[s.name for s in plan.steps]}")

        # Step 2: Execution
        logger.info("Step 2: Executing analysis plan...")
        execution_result = self.executor.execute_plan(plan)
        step_outputs = execution_result["step_outputs"]
        workflow_events = execution_result["workflow_events"]

        # 统计执行结果
        success_count = sum(
            1 for v in step_outputs.values()
            if isinstance(v, dict) and v.get("status") != "failed"
        )
        logger.info(f"  {success_count}/{len(step_outputs)} steps completed")

        # Step 3: 收集来源信息
        sources = self._collect_sources(step_outputs)

        # Step 4: Report Generation + Verification
        logger.info("Step 3: Generating and verifying report...")
        memo = self.report_writer.generate_memo(plan, step_outputs, sources)

        # Step 5: Render
        memo_markdown = self.report_writer.render_memo(memo)

        logger.info(f"Research complete. Confidence: {memo.overall_confidence:.0%}")

        return {
            "memo_markdown": memo_markdown,
            "memo_object": memo,
            "plan": plan,
            "workflow_events": workflow_events,
        }

    def _collect_sources(self, step_outputs: dict) -> list:
        """从执行结果中收集所有使用过的来源"""
        # 加载来源元数据
        source_metas = {}
        processed_dir = settings.data_dir / "processed"
        for company_dir in processed_dir.iterdir():
            if company_dir.is_dir():
                sources_file = company_dir / "sources.json"
                if sources_file.exists():
                    with open(sources_file) as f:
                        for s in json.load(f):
                            source_metas[s["source_id"]] = DocumentMeta(**s)

        # 收集使用过的 source_ids
        used_source_ids = set()
        for output in step_outputs.values():
            if isinstance(output, dict):
                for src in output.get("sources_used", []):
                    used_source_ids.add(src["source_id"])

        return [source_metas[sid] for sid in used_source_ids if sid in source_metas]
