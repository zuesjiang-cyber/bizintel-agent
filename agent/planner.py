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

        comparison_signals = ["vs", "versus", "compare", "comparison", "vs."]
        if any(signal in query_lower for signal in comparison_signals):
            return AnalysisMode.COMPETITIVE

        detected_companies = self._detect_companies(query_lower)
        if len(detected_companies) >= 2:
            return AnalysisMode.COMPETITIVE
        if len(detected_companies) == 1:
            return AnalysisMode.COMPANY

        if any(kw in query_lower for kw in self.INDUSTRY_KEYWORDS):
            return AnalysisMode.INDUSTRY

        # Skip LLM call in dev to avoid 403 blocks
        return AnalysisMode.COMPANY

    def _detect_companies(self, query: str) -> List[str]:
        found = []
        for company in self.KNOWN_COMPANIES:
            if company in query:
                found.append(company)
        return found

# ========== Planner ==========
class Planner:
    def __init__(self):
        # We dummy-init OpenAI to avoid instant crashes, but won't call it if configured as stub
        self.client = None
        self.classifier = QueryClassifier()
        self.use_stub = True  # Added to mock OpenAI explicitly

    def create_plan(self, user_query: str, mode: AnalysisMode = None) -> AnalysisPlan:
        if mode is None:
            mode = self.classifier.classify(user_query, self.client)

        framework = FRAMEWORKS[mode]
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

    def _generate_search_queries(self, user_query: str, step: AnalysisStep, mode: AnalysisMode) -> List[str]:
        if self.use_stub:
            return [f"{user_query} {step.name}", f"{user_query} {step.description}"]
            
        try:
            prompt = f"""You are generating search queries for a business research system..."""
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            if "```" in content:
                content = re.search(r'\[.*\]', content, re.DOTALL).group()
            return json.loads(content)
        except Exception as e:
            return [f"{user_query} {step.name}", f"{user_query} {step.description}"]
