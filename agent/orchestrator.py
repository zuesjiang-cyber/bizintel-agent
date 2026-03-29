"""
主编排器 — 深度研究型可信财务 RAG Agent 的入口。
"""

import logging
from typing import Optional, Sequence

from agent.hello_research_agent import HelloDeepResearchAgent
from agent.report_writer import ReportWriter
from agent.research_controller import ResearchController
from agent.schemas import AnalysisMode

logger = logging.getLogger(__name__)


class BizIntelAgent:
    def __init__(
        self,
        load_models: bool = True,
        demo_mode: bool = False,
        verify_report: bool = True,
        index_dir=None,
    ):
        del index_dir  # 新架构按公司隔离加载 processed corpus，不再依赖全局 index_dir。
        self.demo_mode = demo_mode
        self.report_writer = ReportWriter(
            demo_mode=demo_mode,
            enable_report_verification=verify_report,
        )
        self.controller = ResearchController(
            load_models=load_models,
            demo_mode=demo_mode,
            report_writer=self.report_writer,
        )
        self.agent = HelloDeepResearchAgent(self.controller)
        logger.info("BizIntel deep-research agent ready.")

    def research(
        self,
        query: str,
        mode: Optional[AnalysisMode] = None,
        *,
        company_id: Optional[str] = None,
        period: Optional[str] = None,
        target_periods: Optional[Sequence[str]] = None,
        required_slots: Optional[Sequence[str]] = None,
        required_source_types: Optional[Sequence[str]] = None,
        query_types: Optional[Sequence[str]] = None,
    ) -> dict:
        logger.info("Starting deep research run: %s", query)
        result = self.agent.research(
            query=query,
            mode=mode,
            company_id=company_id,
            period=period,
            target_periods=list(target_periods) if target_periods else None,
            required_slots=list(required_slots) if required_slots else None,
            required_source_types=list(required_source_types) if required_source_types else None,
            query_types=list(query_types) if query_types else None,
        )
        logger.info(
            "Deep research complete. Confidence=%s llm_calls=%s",
            f"{result['memo_object'].overall_confidence:.0%}",
            result.get("llm_calls_used", 0),
        )
        return result
