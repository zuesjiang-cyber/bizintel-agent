import json
import re
from typing import List
from agent.config import settings
from agent.llm_utils import build_openai_client, generate_text_response, should_use_stub_llm
from agent.prompts.research import (
    RESEARCH_PLANNING_SYSTEM_PROMPT,
    build_research_planning_prompt,
)
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
        name="financial_quality",
        description="Revenue quality, margin structure, cash flow signals, accounting red flags, and balance-sheet resilience.",
        required=True,
        output_schema={"financial_quality": "str"},
    ),
    AnalysisStep(
        name="competitive_landscape",
        description="Major competitors, market share, differentiation, and defensibility.",
        required=True,
        output_schema={"competition": "str"},
    ),
    AnalysisStep(
        name="investment_takeaway",
        description="Commercial implications, valuation anchors, monitorable KPIs, catalysts, and downside risks for an investor or strategy team.",
        required=True,
        output_schema={"investment_takeaway": "str"},
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
        if len(detected_companies) == 1:
            return AnalysisMode.COMPANY

        if any(self._contains_keyword(query_lower, kw) for kw in self.INDUSTRY_KEYWORDS):
            return AnalysisMode.INDUSTRY

        # Default fallback is Company
        return AnalysisMode.COMPANY

    def _contains_keyword(self, query: str, keyword: str) -> bool:
        if " " in keyword or "-" in keyword:
            return keyword in query
        return re.search(rf"\b{re.escape(keyword)}\b", query) is not None

    def _detect_companies(self, query: str) -> List[str]:
        found = []
        for company in self.KNOWN_COMPANIES:
            if company in query:
                found.append(company)
        return found


LEADING_QUERY_WORDS = {
    "analyze", "assess", "compare", "evaluation", "evaluate", "research", "review",
    "explain", "summarize", "deep", "dive", "the", "a", "an", "for", "of", "and",
    "vs", "versus", "with", "into", "in", "depth",
}

# ========== Planner ==========
class Planner:
    def __init__(self, force_stub: bool = False):
        self.classifier = QueryClassifier()
        self.use_stub = force_stub or should_use_stub_llm(settings.llm_mode, settings.openai_api_key)
        self.client = None
        if not self.use_stub:
            self.client = build_openai_client(settings.openai_api_key, settings.openai_api_base)

    def create_plan(self, user_query: str, mode: AnalysisMode = None) -> AnalysisPlan:
        if mode is None:
            mode = self.classifier.classify(user_query, self.client)

        framework = FRAMEWORKS[mode]
        contract = self._build_contract(user_query, mode)
        enriched_steps = []
        for step in framework:
            query_contracts = self._generate_query_contracts(user_query, step, mode, contract)
            queries = [item["query_text"] for item in query_contracts]
            section_requirements = contract.get("sections", {}).get(step.name, {})
            enriched_steps.append(AnalysisStep(
                name=step.name,
                description=step.description,
                required=step.required,
                search_queries=queries,
                query_contracts=query_contracts,
                output_schema=step.output_schema,
                evidence_requirements=section_requirements,
            ))

        return AnalysisPlan(
            mode=mode,
            user_query=user_query,
            steps=enriched_steps,
            contract=contract,
        )

    def _build_contract(self, user_query: str, mode: AnalysisMode) -> dict:
        query_lower = user_query.lower()
        entities = self._infer_entities(user_query)
        query_type = self._infer_query_type(query_lower, mode)

        default_required_slots_map = {
            AnalysisMode.COMPANY: [
                "business_model",
                "financial_quality",
                "competitive_position",
                "risks",
            ],
            AnalysisMode.INDUSTRY: [
                "industry_definition",
                "market_sizing",
                "major_players",
                "trends",
            ],
            AnalysisMode.COMPETITIVE: [
                "entity_a_profile",
                "entity_b_profile",
                "product_breadth_comparison",
                "monetization_comparison",
                "winner_or_tradeoff",
            ],
        }
        required_slots = default_required_slots_map[mode]
        if "business model" in query_lower and "monetization" in query_lower and "strategy" in query_lower:
            required_slots = ["business model", "monetization levers", "strategy framing"]
        elif "revenue" in query_lower and "profitability" in query_lower and "demand" in query_lower:
            required_slots = ["reported revenue", "reported profitability or margin metric", "management demand commentary"]
        elif "catalyst" in query_lower and "risk" in query_lower:
            required_slots = ["catalyst", "risk", "AI growth framing"]
        elif mode == AnalysisMode.COMPETITIVE:
            required_slots = ["product breadth", "monetization style", "growth driver comparison"]
        elif query_type == "time_sensitive":
            required_slots = ["Q3 vs Q4 comparison", "AI growth-driver framing", "management commentary shift"]
        elif "high-confidence claim" in query_lower and "lower-confidence" in query_lower:
            required_slots = ["high-confidence claim", "lower-confidence claim", "evidence distinction"]

        primary_sources = ["earnings_call_transcript", "shareholder_letter", "annual_report", "results_release"]
        if mode == AnalysisMode.COMPANY:
            primary_sources.append("company_profile")

        contract = {
            "normalized_question": self._normalize_question(user_query, entities),
            "answer_type": self._infer_answer_type(query_lower, mode),
            "query_type": query_type,
            "entities": entities,
            "plan": [
                "Identify required facts and acceptable source types.",
                "Retrieve evidence for each fact slot in source-priority order.",
                "Write only from supported evidence and flag uncertainty explicitly.",
            ],
            "estimate": self._estimate_task(query_type, mode),
            "non_goals": [
                "Do not use open-web facts outside the curated source pack.",
                "Do not invent numbers, market-share claims, or unsupported causal explanations.",
            ],
            "required_slots": required_slots,
            "optional_facts": self._build_optional_facts(mode, query_type),
            "research_subquestions": self._build_research_subquestions(required_slots, entities, query_type),
            "required_primary_source_types": primary_sources,
            "preferred_source_order": primary_sources + ["investor_presentation"],
            "hard_failures": [
                "missing_citation",
                "bad_source_id",
                "numeric_mismatch",
            ],
            "answer_rules": [
                "Only write factual claims that can be tied to retrieved evidence.",
                "State insufficient evidence explicitly when the pack does not support a claim.",
                "Keep comparisons symmetric across entities when query_type is comparison.",
            ],
            "output_outline": [
                "Direct Answer",
                "Supporting Points",
                "Analytical Inference",
                "Uncertainty",
            ],
            "sections": self._build_section_requirements(
                mode=mode,
                query_type=query_type,
                required_slots=required_slots,
                primary_sources=primary_sources,
            ),
        }

        extracted_periods = self._extract_required_periods(query_lower)
        if extracted_periods:
            contract["required_periods"] = extracted_periods
        elif query_type == "time_sensitive":
            contract["required_periods"] = ["current_period", "prior_period"]

        return contract

    def _infer_query_type(self, query_lower: str, mode: AnalysisMode) -> str:
        if mode == AnalysisMode.COMPETITIVE:
            base_type = "comparison"
        else:
            base_type = "standard"

        quarter_mentions = re.findall(r"\bq[1-4]\s*20\d{2}\b", query_lower)
        multiple_periods = len(set(quarter_mentions)) >= 2
        temporal_comparison_terms = {
            "changed",
            "change",
            "shift",
            "compare",
            "comparison",
            "versus",
            " vs ",
            "from ",
            " to ",
            "quarter-over-quarter",
            "year-over-year",
            "qoq",
            "yoy",
        }
        latest_terms = {"latest", "most recent"}

        if multiple_periods:
            return "time_sensitive"
        if any(term in query_lower for term in latest_terms):
            return "time_sensitive"
        if any(term in query_lower for term in temporal_comparison_terms) and quarter_mentions:
            return "time_sensitive"
        return base_type

    def _refine_contract_with_llm(self, user_query: str, mode: AnalysisMode, contract: dict) -> dict:
        steps = [
            {"name": step.name, "description": step.description}
            for step in FRAMEWORKS[mode]
        ]
        prompt = build_research_planning_prompt(
            user_query=user_query,
            mode=mode.value,
            draft_contract=contract,
            steps=steps,
        )
        try:
            content = generate_text_response(
                self.client,
                model=settings.openai_model,
                system_prompt=RESEARCH_PLANNING_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=1200,
                temperature=0.1,
                max_retries=0,
            )
            payload = self._parse_json_payload(content)
            if not isinstance(payload, dict):
                return contract
            for key in (
                "normalized_question",
                "answer_type",
                "estimate",
            ):
                if isinstance(payload.get(key), str) and payload[key].strip():
                    contract[key] = payload[key].strip()
            for key in (
                "plan",
                "non_goals",
                "optional_facts",
                "research_subquestions",
                "preferred_source_order",
                "output_outline",
                "answer_rules",
            ):
                if isinstance(payload.get(key), list) and payload[key]:
                    contract[key] = [str(item) for item in payload[key][:8]]
            if isinstance(payload.get("required_slots"), list) and payload["required_slots"]:
                contract["required_slots"] = [str(item) for item in payload["required_slots"][:8]]
            if isinstance(payload.get("sections"), dict):
                merged_sections = contract.get("sections", {})
                for section_name, section_payload in payload["sections"].items():
                    if section_name not in merged_sections or not isinstance(section_payload, dict):
                        continue
                    merged_sections[section_name] = self._merge_section_requirements(
                        merged_sections[section_name],
                        section_payload,
                    )
                contract["sections"] = merged_sections
        except Exception:
            return contract
        return contract

    def _merge_section_requirements(self, base: dict, override: dict) -> dict:
        merged = dict(base)
        if isinstance(override.get("required_facts"), list) and override["required_facts"]:
            merged["required_facts"] = [str(item) for item in override["required_facts"][:8]]
        if isinstance(override.get("allowed_source_types"), list) and override["allowed_source_types"]:
            merged["allowed_source_types"] = [str(item) for item in override["allowed_source_types"][:8]]
        if isinstance(override.get("banned_claim_types"), list) and override["banned_claim_types"]:
            merged["banned_claim_types"] = [str(item) for item in override["banned_claim_types"][:8]]
        if isinstance(override.get("downgrade_rule"), str) and override["downgrade_rule"].strip():
            merged["downgrade_rule"] = override["downgrade_rule"].strip()
        if isinstance(override.get("writing_order"), list) and override["writing_order"]:
            merged["writing_order"] = [str(item) for item in override["writing_order"][:4]]
        if isinstance(override.get("fact_first"), bool):
            merged["fact_first"] = override["fact_first"]
        return merged

    def _parse_json_payload(self, content: str):
        text = content.strip()
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text)

    def _build_section_requirements(
        self,
        *,
        mode: AnalysisMode,
        query_type: str,
        required_slots: List[str],
        primary_sources: List[str],
    ) -> dict:
        default_downgrade = "If direct evidence is missing, write 'insufficient evidence' instead of a factual claim."
        section_map = {
            AnalysisMode.COMPANY: {
                "company_overview": ["founding", "company description", "mission"],
                "business_model": ["core products", "monetization", "customer segments"],
                "financial_analysis": ["reported financial facts", "valuation or funding facts", "time period"],
                "financial_quality": ["revenue quality", "margin or cash flow evidence", "red flags or resilience"],
                "competitive_landscape": ["competitors", "differentiators", "positioning evidence"],
                "investment_takeaway": ["catalysts", "downside risks", "monitorables"],
                "risks_and_outlook": ["risk factors", "industry headwinds", "forward-looking caveats"],
            },
            AnalysisMode.INDUSTRY: {
                "industry_definition": ["definition", "sub-segments", "ecosystem"],
                "market_sizing": ["market size", "growth rate", "time period"],
                "value_chain": ["upstream", "downstream", "value chain structure"],
                "major_players": ["major players", "competitive dynamics", "market structure"],
                "industry_trends": ["trends", "technology shifts", "regulatory headwinds"],
            },
            AnalysisMode.COMPETITIVE: {
                "profiles_overview": ["entity A facts", "entity B facts", "scale context"],
                "product_comparison": ["product breadth", "monetization style", "pricing or packaging"],
                "market_positioning": ["target customers", "go-to-market", "strategic framing"],
                "strengths_and_weaknesses": ["growth drivers", "risks", "tradeoffs"],
            },
        }[mode]

        banned_claims = [
            "unsupported market share",
            "invented customer concentration",
            "speculative product revenue split",
        ]
        if query_type == "time_sensitive":
            banned_claims.append("smoothed trend claim without period-specific evidence")

        sections = {}
        required_slot_set = {slot.lower() for slot in required_slots}
        for step_name, step_facts in section_map.items():
            section_facts = list(step_facts)
            if "monetization" in " ".join(required_slot_set) and step_name in {"business_model", "product_comparison"}:
                section_facts.append("explicit monetization evidence")
            if query_type == "comparison" and step_name in {"product_comparison", "market_positioning", "strengths_and_weaknesses"}:
                section_facts.extend(["entity A facts before conclusion", "entity B facts before conclusion"])
            if query_type == "time_sensitive" and step_name in {"financial_analysis", "financial_quality", "market_positioning"}:
                section_facts.extend(["current-period evidence", "prior-period evidence", "management commentary shift"])

            sections[step_name] = {
                "section_name": step_name,
                "required_facts": list(dict.fromkeys(section_facts)),
                "allowed_source_types": list(dict.fromkeys(primary_sources)),
                "banned_claim_types": banned_claims,
                "downgrade_rule": default_downgrade,
                "writing_order": ["facts", "management_commentary", "inference"],
                "fact_first": query_type in {"comparison", "time_sensitive"} or step_name in {"financial_analysis", "financial_quality"},
            }

        return sections

    def _generate_query_contracts(
        self,
        user_query: str,
        step: AnalysisStep,
        mode: AnalysisMode,
        contract: dict,
    ) -> List[dict]:
        """Generate structured retrieval contracts and derive search query strings from them."""
        company_match = self.classifier._detect_companies(user_query.lower())
        entities = self._infer_entities(user_query)
        main_entity = company_match[0].title() if company_match else entities[0]

        if mode == AnalysisMode.COMPANY:
            if step.name == "company_overview":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [f"{main_entity} founding history founders", f"{main_entity} overall company profile overview"],
                )
            if step.name == "business_model":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [f"{main_entity} business model revenue streams", f"{main_entity} core products pricing"],
                )
            if step.name == "financial_analysis":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [f"{main_entity} funding rounds Series valuation", f"{main_entity} annual revenue payment volume"],
                )
            if step.name == "financial_quality":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [
                        f"{main_entity} gross margin operating margin free cash flow",
                        f"{main_entity} revenue quality deferred revenue accounting risks",
                    ],
                )
            if step.name == "competitive_landscape":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [f"{main_entity} competitors market share", f"{main_entity} differentiation vs competitors"],
                )
            if step.name == "investment_takeaway":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [
                        f"{main_entity} valuation multiples catalysts monitorable KPIs",
                        f"{main_entity} investor concerns commercial outlook downside risks",
                    ],
                )
            if step.name == "risks_and_outlook":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [f"{main_entity} regulatory risks challenges", f"{main_entity} growth headwinds outlook"],
                )

        if mode == AnalysisMode.COMPETITIVE:
            entity_pair = [c.title() for c in company_match] if company_match else entities[:2]
            entities_text = " vs ".join(entity_pair) if len(entity_pair) >= 2 else main_entity
            if step.name == "profiles_overview":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [
                        f"{entities_text} company profiles scale overview",
                        f"{entities_text} revenue growth business description",
                    ],
                )
            if step.name == "product_comparison":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [f"{entities_text} product features pricing compare", f"{entities_text} developer integration comparison"],
                )
            if step.name == "market_positioning":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [
                        f"{entities_text} target customers enterprise developer go to market",
                        f"{entities_text} monetization style usage based subscription comparison",
                    ],
                )
            if step.name == "strengths_and_weaknesses":
                return self._wrap_stub_queries(
                    contract,
                    step.evidence_requirements,
                    [
                        f"{entities_text} strategic framing strengths weaknesses comparison",
                        f"{entities_text} management commentary growth drivers risks",
                    ],
                )

        if mode == AnalysisMode.INDUSTRY and step.name == "market_sizing":
            return self._wrap_stub_queries(
                contract,
                step.evidence_requirements,
                [f"{user_query} market size TAM CAGR 2024", f"{user_query} global revenue projections"],
            )

        step_hint = step.name.replace("_", " ")
        return self._wrap_stub_queries(
            contract,
            step.evidence_requirements,
            [
                f"{user_query} {step_hint} details",
                f"{user_query} {step.description.split(',')[0]} statistics",
            ],
        )

    def _wrap_stub_queries(self, contract: dict, evidence_requirements: dict, queries: List[str]) -> List[dict]:
        fact_slot = self._default_fact_slot(evidence_requirements)
        filters = self._build_query_filters(contract, evidence_requirements)
        return [
            {
                "fact_slot": fact_slot,
                "query_text": query,
                "source_type": list(evidence_requirements.get("allowed_source_types", [])),
                "filters": filters,
                "expected_evidence_type": "narrative",
                "calculation_hint": "",
            }
            for query in queries
        ]

    def _query_contract_from_string(self, query: str, contract: dict, evidence_requirements: dict) -> dict:
        return {
            "fact_slot": self._default_fact_slot(evidence_requirements),
            "query_text": query,
            "source_type": list(evidence_requirements.get("allowed_source_types", [])),
            "filters": self._build_query_filters(contract, evidence_requirements),
            "expected_evidence_type": "narrative",
            "calculation_hint": "",
        }

    def _default_fact_slot(self, evidence_requirements: dict) -> str:
        facts = evidence_requirements.get("required_facts", [])
        return str(facts[0]) if facts else "required_fact"

    def _normalize_question(self, user_query: str, entities: List[str]) -> str:
        if entities and entities[0] != "Unknown Company":
            return user_query.strip()
        return f"Research question: {user_query.strip()}"

    def _infer_answer_type(self, query_lower: str, mode: AnalysisMode) -> str:
        if any(token in query_lower for token in {"revenue", "margin", "growth", "valuation", "cash flow"}):
            return "numerical_or_financial_analysis"
        if mode == AnalysisMode.COMPETITIVE:
            return "comparison"
        if "risk" in query_lower:
            return "risk_list"
        return "narrative_analysis"

    def _estimate_task(self, query_type: str, mode: AnalysisMode) -> str:
        if query_type == "time_sensitive":
            return "Likely answerable from period-specific filings, but period alignment and management commentary may be tricky."
        if mode == AnalysisMode.COMPETITIVE:
            return "Answer should be available if both entities exist in the corpus, with the main risk being asymmetric evidence coverage."
        return "Answer is likely available from the permitted documents, with moderate difficulty if key facts are dispersed across source types."

    def _build_optional_facts(self, mode: AnalysisMode, query_type: str) -> List[str]:
        optional = ["management commentary nuance", "counterevidence or downside caveat"]
        if mode == AnalysisMode.COMPANY:
            optional.append("monitorables or KPI context")
        if query_type == "time_sensitive":
            optional.append("prior-period comparison frame")
        return optional

    def _build_research_subquestions(self, required_slots: List[str], entities: List[str], query_type: str) -> List[str]:
        subquestions = [f"What evidence supports {slot}?" for slot in required_slots[:4]]
        if query_type == "comparison" and len(entities) >= 2:
            subquestions.append(f"What is the evidence-backed difference between {entities[0]} and {entities[1]}?")
        if query_type == "time_sensitive":
            subquestions.append("What changed across the relevant periods, and which source proves it?")
        return subquestions

    def _infer_entities(self, user_query: str) -> List[str]:
        detected = [company.title() for company in self.classifier._detect_companies(user_query.lower())]
        if detected:
            return detected

        normalized = re.sub(r"\bversus\b|\bvs\.\b|\bvs\b", " vs ", user_query, flags=re.IGNORECASE)
        parts = [part.strip() for part in re.split(r"\bvs\b|,", normalized, flags=re.IGNORECASE) if part.strip()]

        entities = []
        for part in parts:
            candidate = self._extract_entity_candidate(part)
            if candidate and candidate not in entities:
                entities.append(candidate)

        return entities or ["Unknown Company"]

    def _extract_entity_candidate(self, text: str) -> str:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9&.-]*", text)
        filtered = [token for token in tokens if token.lower() not in LEADING_QUERY_WORDS]
        if not filtered:
            return "Unknown Company"

        candidate_tokens = []
        for token in filtered:
            lower = token.lower()
            if candidate_tokens and lower in {"revenue", "quality", "valuation", "drivers", "monitorables", "industry", "market"}:
                break
            candidate_tokens.append(token)
            if len(candidate_tokens) >= 2:
                break

        return " ".join(token.title() for token in candidate_tokens)

    def _extract_required_periods(self, query_lower: str) -> List[str]:
        periods: List[str] = []

        for match in re.finditer(r"\bq([1-4])\s*(20\d{2})\b", query_lower):
            periods.append(f"{match.group(2)}Q{match.group(1)}")
        for match in re.finditer(r"\bfy\s*(20\d{2})\b", query_lower):
            periods.append(f"{match.group(1)}FY")
        for match in re.finditer(r"\b(20\d{2})\b", query_lower):
            year = match.group(1)
            if not any(period.startswith(year) for period in periods):
                periods.append(year)

        return list(dict.fromkeys(periods))

    def _build_query_filters(self, contract: dict, evidence_requirements: dict) -> dict:
        entities = [
            entity for entity in contract.get("entities", [])
            if entity and entity != "Unknown Company"
        ]
        periods = list(contract.get("required_periods", []))
        source_types = list(evidence_requirements.get("allowed_source_types", []))
        return {
            "companies": entities,
            "periods": periods,
            "source_types": source_types,
        }
