import json
import re
from typing import List, Optional
from openai import OpenAI
from agent.config import settings
from agent.schemas import AnalysisMode, AnalysisPlan, AnalysisStep

# ========== 分析框架模板 ==========
COMPANY_FRAMEWORK = [
    AnalysisStep(
        name="company_overview",
        description="Company overview, founding history, headquarters, and core mission.",
        required=True,
        output_schema={"overview": "str"},
    ),
    AnalysisStep(
        name="business_model",
        description="Business model, core products, revenue streams, and target customer segments.",
        required=True,
        output_schema={"business_model": "str"},
    ),
    AnalysisStep(
        name="financial_analysis",
        description="Financial metrics, funding history, valuation changes, and revenue data.",
        required=True,
        output_schema={"financials": "str"},
    ),
    AnalysisStep(
        name="competitive_landscape",
        description="Major competitors, market share, differentiation, and defensibility.",
        required=True,
        output_schema={"competition": "str"},
    ),
    AnalysisStep(
        name="risks_and_outlook",
        description="Risk factors, regulatory challenges, industry trends, and future outlook.",
        required=False,
        output_schema={"risks_outlook": "str"},
    ),
]

INDUSTRY_FRAMEWORK = [
    AnalysisStep(
        name="industry_definition",
        description="Define the industry, core boundaries, sub-segments, and overall ecosystem.",
        required=True,
        output_schema={"definition": "str"},
    ),
    AnalysisStep(
        name="market_sizing",
        description="Total addressable market (TAM), current size, and estimated CAGR.",
        required=True,
        output_schema={"market_sizing": "str"},
    ),
    AnalysisStep(
        name="value_chain",
        description="Value chain structure, upstream providers, downstream channels.",
        required=True,
        output_schema={"value_chain": "str"},
    ),
    AnalysisStep(
        name="major_players",
        description="Dominant companies, market share distribution, and competitive dynamics.",
        required=True,
        output_schema={"players": "str"},
    ),
    AnalysisStep(
        name="industry_trends",
        description="Emerging trends, technological disruptions, and regulatory headwinds.",
        required=True,
        output_schema={"trends": "str"},
    ),
]

COMPETITIVE_FRAMEWORK = [
    AnalysisStep(
        name="profiles_overview",
        description="Basic profiles and scale of the companies being compared.",
        required=True,
        output_schema={"profiles": "str"},
    ),
    AnalysisStep(
        name="product_comparison",
        description="Direct mapping of product suites, features, and pricing models.",
        required=True,
        output_schema={"products": "str"},
    ),
    AnalysisStep(
        name="market_positioning",
        description="Target audience overlap, go-to-market strategies, and brand positioning.",
        required=True,
        output_schema={"positioning": "str"},
    ),
    AnalysisStep(
        name="strengths_and_weaknesses",
        description="Relative advantages, technical debt, or strategic vulnerabilities of each player.",
        required=True,
        output_schema={"swot": "str"},
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
        "openai", "snowflake", "datadog", "cloudflare", "plaid", "paypal", "square", "adyen"
    }
    INDUSTRY_KEYWORDS = {
        "industry", "market", "sector", "landscape", "space",
        "fintech", "saas", "ai", "cloud", "healthtech", "edtech",
        "e-commerce", "cybersecurity", "payments", "infrastructure", "digital payment"
    }

    def classify(self, query: str, llm_client=None) -> AnalysisMode:
        query_lower = query.lower()

        comparison_signals = ["vs", "versus", "compare", "comparison", "vs."]
        if any(signal in query_lower for signal in comparison_signals):
            return AnalysisMode.COMPETITIVE

        detected_companies = self._detect_companies(query_lower)
        if len(detected_companies) >= 2:
            return AnalysisMode.COMPETITIVE

        if any(kw in query_lower for kw in self.INDUSTRY_KEYWORDS):
            return AnalysisMode.INDUSTRY

        # Default fallback is Company
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
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_base)
        self.classifier = QueryClassifier()
        self.use_stub = True  # Added to mock OpenAI explicitly in dev mode to save latency

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
        """Generate specific search queries based on the step."""
        if self.use_stub:
            # Deterministic, specific query generation logic without LLM
            company_match = self.classifier._detect_companies(user_query.lower())
            main_entity = company_match[0].title() if company_match else user_query.split()[0]
            
            if mode == AnalysisMode.COMPANY:
                if step.name == "company_overview":
                    return [f"{main_entity} founding history founders", f"{main_entity} overall company profile overview"]
                elif step.name == "business_model":
                    return [f"{main_entity} business model revenue streams", f"{main_entity} core products pricing"]
                elif step.name == "financial_analysis":
                    return [f"{main_entity} funding rounds Series valuation", f"{main_entity} annual revenue payment volume"]
                elif step.name == "competitive_landscape":
                    return [f"{main_entity} competitors market share", f"{main_entity} differentiation vs competitors"]
                elif step.name == "risks_and_outlook":
                    return [f"{main_entity} regulatory risks challenges", f"{main_entity} growth headwinds outlook"]
            
            elif mode == AnalysisMode.COMPETITIVE:
                entities = " vs ".join([c.title() for c in company_match]) if company_match else user_query
                if step.name == "product_comparison":
                    return [f"{entities} product features pricing compare", f"{entities} developer integration comparison"]
                # Similar mapping for others... fallback below
                
            elif mode == AnalysisMode.INDUSTRY:
                if step.name == "market_sizing":
                    return [f"{user_query} market size TAM CAGR 2024", f"{user_query} global revenue projections"]
            
            # General fallback
            step_hint = step.name.replace("_", " ")
            return [
                f"{user_query} {step_hint} details",
                f"{user_query} {step.description.split(',')[0]} statistics"
            ]
            
        try:
            prompt = f"Generate 2-4 highly specific search queries (e.g. 'Stripe Series I valuation 2023') for researching '{user_query}' focusing on '{step.name}'. Return JSON list of strings."
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
            return [f"{user_query} {step.name}", f"{user_query} specific data"]
