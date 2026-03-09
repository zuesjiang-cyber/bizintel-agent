"""
Memo 生成的 prompt 模板
"""

MEMO_SYSTEM_PROMPT = """You are a senior business research analyst writing a research memorandum.

Your writing style:
- Professional but clear
- Every factual claim must cite its source using [Source: source_id] format
- Use specific numbers and dates when available
- Flag uncertainty explicitly: "Based on available information..." or "Unable to verify..."
- Structure each section with clear headers and bullet points where appropriate
"""

COMPANY_MEMO_TEMPLATE = """# Company Research Memo: {company_name}

Generated: {date} | Mode: Company Deep Dive

## Executive Summary
{executive_summary}

## Company Profile
{company_profile}

## Business Model
{business_model}

## Market & Competitive Position
{market_position}

## Growth Signals
{growth_signals}

## Risk Assessment
{risk_assessment}

## Due Diligence Questions
{due_diligence_questions}

---
## Sources
{sources_list}

## Verification Summary
{verification_summary}
"""

INDUSTRY_MEMO_TEMPLATE = """# Industry Landscape: {industry_name}

Generated: {date} | Mode: Industry Landscape

## Executive Summary
{executive_summary}

## Industry Definition & Scope
{industry_definition}

## Market Size & Growth
{market_sizing}

## Value Chain & Structure
{value_chain}

## Competitive Landscape
{competitive_landscape}

## Trends & Risks
{trends_and_risks}

---
## Sources
{sources_list}

## Verification Summary
{verification_summary}
"""

COMPETITIVE_MEMO_TEMPLATE = """# Competitive Comparison: {companies}

Generated: {date} | Mode: Competitive Comparison

## Executive Summary
{executive_summary}

## Company Profiles
{company_profiles}

## Product & Pricing Comparison
{product_comparison}

## Market Positioning
{positioning_analysis}

## Strengths & Weaknesses
{strengths_weaknesses}

## Assessment & Recommendation
{recommendation}

---
## Sources
{sources_list}

## Verification Summary
{verification_summary}
"""

EXECUTIVE_SUMMARY_PROMPT = """Based on the following analysis sections, write a 2-3 sentence executive summary.
Focus on the most important findings. Cite sources.

Analysis:
{all_sections}

Write the executive summary:"""
