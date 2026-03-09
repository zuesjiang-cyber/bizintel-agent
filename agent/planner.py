"""
分析规划器

职责：
1. 根据用户输入选择分析模式
2. 根据选定模式加载固定分析框架
3. 用 LLM 为每个步骤生成具体检索 query

设计理念：
- 框架是确定性的（来自行业最佳实践）
- Query 是动态的（需要根据具体公司/行业名称生成）
"""

import json
import re
from typing import List, Optional

from openai import OpenAI

from agent.config import settings
from agent.schemas import AnalysisMode, AnalysisPlan, AnalysisStep


# ========== 分析框架模板 ==========

COMPANY_FRAMEWORK = [
    AnalysisStep(
        name="company_profile",
        description="Company basics: founding date, headquarters, size, funding history, founders, key leadership",
        required=True,
        output_schema={"founded": "str", "hq": "str", "employees": "str",
                       "funding": "str", "founders": "list[str]"},
    ),
    AnalysisStep(
        name="business_model",
        description="Business model: core products/services, target customers, revenue model, pricing strategy, unit economics if available",
        required=True,
        output_schema={"products": "list", "target_customers": "str",
                       "revenue_model": "str", "pricing": "str"},
    ),
    AnalysisStep(
        name="market_position",
        description="Market and competitive position: addressable market, key competitors, differentiation, market share signals",
        required=True,
        output_schema={"market": "str", "tam": "str", "competitors": "list",
                       "differentiation": "str"},
    ),
    AnalysisStep(
        name="growth_signals",
        description="Growth signals: recent news, product launches, partnerships, hiring trends, revenue milestones",
        required=False,
        output_schema={"signals": "list[dict]"},
    ),
    AnalysisStep(
        name="risk_assessment",
        description="Risk assessment: competitive risks, technology risks, regulatory risks, market risks, team/execution risks",
        required=False,
        output_schema={"risks": "list[dict]"},
    ),
    AnalysisStep(
        name="due_diligence_questions",
        description="Generate a list of follow-up questions for further due diligence",
        required=False,
        output_schema={"questions": "list[str]"},
    ),
]

INDUSTRY_FRAMEWORK = [
    AnalysisStep(
        name="industry_definition",
        description="Define the industry: what it includes, key segments, boundaries",
        required=True,
        output_schema={"definition": "str", "segments": "list[str]",
                       "boundaries": "str"},
    ),
    AnalysisStep(
        name="market_sizing",
        description="Market size and growth: current size, growth rate, key growth drivers",
        required=True,
        output_schema={"market_size": "str", "growth_rate": "str",
                       "drivers": "list[str]"},
    ),
    AnalysisStep(
        name="value_chain",
        description="Value chain and industry structure: upstream/downstream, value distribution",
        required=True,
        output_schema={"structure": "str", "key_players_by_segment": "dict"},
    ),
    AnalysisStep(
        name="competitive_landscape",
        description="Key players: categorize by business model, scale, positioning",
        required=True,
        output_schema={"players": "list[dict]"},
    ),
    AnalysisStep(
        name="trends_and_risks",
        description="Key trends, disruptions, regulatory changes, and risks",
        required=False,
        output_schema={"trends": "list", "risks": "list"},
    ),
]

COMPETITIVE_FRAMEWORK = [
    AnalysisStep(
        name="company_profiles",
        description="Basic profile for each company being compared",
        required=True,
        output_schema={"profiles": "list[dict]"},
    ),
    AnalysisStep(
        name="product_comparison",
        description="Product and pricing comparison across companies",
        required=True,
        output_schema={"comparison_table": "list[dict]"},
    ),
    AnalysisStep(
        name="positioning_analysis",
        description="Market positioning, target customer, and differentiation for each",
        required=True,
        output_schema={"positioning": "list[dict]"},
    ),
    AnalysisStep(
        name="strengths_weaknesses",
        description="Strengths and weaknesses matrix",
        required=True,
        output_schema={"matrix": "list[dict]"},
    ),
    AnalysisStep(
        name="recommendation",
        description="Comparative assessment and recommendation",
        required=False,
        output_schema={"assessment": "str", "recommendation": "str"},
    ),
]

FRAMEWORKS = {
    AnalysisMode.COMPANY: COMPANY_FRAMEWORK,
    AnalysisMode.INDUSTRY: INDUSTRY_FRAMEWORK,
    AnalysisMode.COMPETITIVE: COMPETITIVE_FRAMEWORK,
}


# ========== Query 分类 ==========

class QueryClassifier:
    """
    规则优先 + LLM 兜底的模式分类

    为什么不全用 LLM：
    - 简单的情况用规则更快、更确定
    - 减少不必要的 API 调用
    - 可测试、可 debug
    """

    # 常见公司名模式（可扩展）
    KNOWN_COMPANIES = {
        "stripe", "notion", "databricks", "figma", "anthropic",
        "openai", "snowflake", "datadog", "cloudflare", "plaid",
    }

    INDUSTRY_KEYWORDS = {
        "industry", "market", "sector", "landscape", "space",
        "fintech", "saas", "ai", "cloud", "healthtech", "edtech",
        "e-commerce", "cybersecurity", "payments", "infrastructure",
    }

    def classify(self, query: str, llm_client=None) -> AnalysisMode:
        query_lower = query.lower()

        # 规则 1: 明确的比较请求
        comparison_signals = ["vs", "versus", "compare", "comparison", "vs."]
        if any(signal in query_lower for signal in comparison_signals):
            return AnalysisMode.COMPETITIVE

        # 规则 2: 检测公司名数量
        detected_companies = self._detect_companies(query_lower)
        if len(detected_companies) >= 2:
            return AnalysisMode.COMPETITIVE
        if len(detected_companies) == 1:
            return AnalysisMode.COMPANY

        # 规则 3: 行业关键词
        if any(kw in query_lower for kw in self.INDUSTRY_KEYWORDS):
            return AnalysisMode.INDUSTRY

        # LLM 兜底
        if llm_client:
            return self._llm_classify(query, llm_client)

        # 默认
        return AnalysisMode.COMPANY

    def _detect_companies(self, query: str) -> List[str]:
        found = []
        for company in self.KNOWN_COMPANIES:
            if company in query:
                found.append(company)
        return found

    def _llm_classify(self, query: str, client) -> AnalysisMode:
        prompt = f"""Classify this business research query into one of three categories:
- COMPANY: researching a specific company
- INDUSTRY: researching an industry or market
- COMPETITIVE: comparing multiple companies

Query: "{query}"

Return only one word: COMPANY, INDUSTRY, or COMPETITIVE"""

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().upper()
        mapping = {
            "COMPANY": AnalysisMode.COMPANY,
            "INDUSTRY": AnalysisMode.INDUSTRY,
            "COMPETITIVE": AnalysisMode.COMPETITIVE,
        }
        return mapping.get(answer, AnalysisMode.COMPANY)


# ========== Planner ==========

class Planner:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.classifier = QueryClassifier()

    def create_plan(self, user_query: str, mode: AnalysisMode = None) -> AnalysisPlan:
        # 1. 分类（如果没有指定模式）
        if mode is None:
            mode = self.classifier.classify(user_query, self.client)

        # 2. 加载框架
        framework = FRAMEWORKS[mode]

        # 3. 为每个步骤生成检索 query
        enriched_steps = []
        for step in framework:
            queries = self._generate_search_queries(user_query, step, mode)
            enriched_steps.append(AnalysisStep(
                name=step.name,
                description=step.description,
                required=step.required,
                search_queries=queries,
                output_schema=step.output_schema,
            ))

        return AnalysisPlan(
            mode=mode,
            user_query=user_query,
            steps=enriched_steps,
        )

    def _generate_search_queries(
        self, user_query: str, step: AnalysisStep, mode: AnalysisMode
    ) -> List[str]:
        prompt = f"""You are generating search queries for a business research system.

User's research question: "{user_query}"
Analysis mode: {mode.value}
Current analysis step: {step.name}
Step description: {step.description}

Generate 3-5 specific search queries that would help gather information for this step.
Be specific — include company names, industry names, and concrete terms.
Focus on factual, verifiable information.

Return as a JSON array of strings. Example:
["Stripe total funding history", "Stripe Series I round 2023 details"]"""

        response = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()
        # 提取 JSON 数组
        try:
            # 处理可能的 markdown code block
            if "```" in content:
                content = re.search(r'\[.*\]', content, re.DOTALL).group()
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError):
            # fallback: 根据步骤描述和用户查询生成基础 query
            return [f"{user_query} {step.name}", f"{user_query} {step.description}"]
