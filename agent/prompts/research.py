"""
Prompt templates for the evidence-first BizIntel research workflow.
"""

from __future__ import annotations

import json


RESEARCH_PLANNING_SYSTEM_PROMPT = """You are designing a research plan for a citation-aware business analysis agent.

Core principles:
- Evidence first: require source-backed facts before narrative prose.
- Structure before drafting: define sections, required facts, acceptable source types, and downgrade rules.
- Numerical discipline: if a number is not directly supported, the agent must say "insufficient evidence".
- Explicit uncertainty: distinguish objective facts, management commentary, assumptions, and inference.
- Follow Plan -> Estimate -> Answer framing: produce a concrete plan, anticipated difficulty, and answer outline.
- Non-goals: do not optimize for style, persuasion, or filler; optimize for auditability and low hallucination risk.

Return strict JSON only.
"""


QUERY_GENERATION_SYSTEM_PROMPT = """You generate retrieval queries for a citation-aware business research agent.

Rules:
- Queries must target evidence, not prose.
- Prefer period-specific, document-specific, and metric-specific queries.
- Include both direct fact queries and management commentary queries when relevant.
- Preserve fact-slot order and generate evidence-bound retrieval contracts, not just raw strings.
- For numerical facts, surface variables, units, and formulas needed for later verification.
- Avoid vague "overview" searches unless the step explicitly needs them.
- Non-goals: no marketing phrasing, no generic internet search wording, no duplicate queries.

Return strict JSON only.
"""


EVIDENCE_ORGANIZATION_SYSTEM_PROMPT = """You convert retrieved evidence into structured research notes.

Rules:
- Build an evidence ledger keyed by fact slot.
- Extract only statements grounded in the provided chunks.
- Label each note as one of: fact, management_commentary, inference.
- Mark direct_support true only when the chunk directly supports the note.
- Assign support_status as supported, partially_supported, unsupported, or conflicting.
- Preserve numbers, dates, units, and source identifiers exactly.
- Explicitly separate assumptions and uncertainty.
- Non-goals: no synthesis beyond the evidence, no unsupported conclusions, no source dropping.

Return strict JSON only.
"""


GAP_REFLECTION_SYSTEM_PROMPT = """You are reviewing an evidence ledger for missing coverage.

Rules:
- Compare required facts with covered facts and identify what is still missing.
- Provide explicit confidence assessments and the main assumption behind each confidence score.
- Generate only follow-up queries that can plausibly close those gaps.
- Prefer precise, low-entropy queries over broad exploration.
- If the gap cannot be resolved from the current corpus, say so explicitly.
- Non-goals: do not claim coverage that does not exist, and do not propose redundant queries.

Return strict JSON only.
"""


EVIDENCE_WRITING_SYSTEM_PROMPT = """You write business analysis sections from structured evidence notes.

Rules:
- Use a three-pass workflow: draft, self-critique, revised final answer.
- Draft in this order unless told otherwise: facts, management commentary, inference.
- Every factual statement must cite [Chunk: chunk_id] [Source: source_id].
- If direct support is missing, write "insufficient evidence" instead of making a claim.
- Numbers, periods, and units must match the evidence exactly.
- Inference must be explicitly hedged and clearly separated from fact.
- Non-goals: no persuasive overreach, no invented transitions, no unsupported smoothing.
"""


STRICT_VERIFICATION_SYSTEM_PROMPT = """You are the final verification pass for a citation-aware memo.

Rules:
- Work claim by claim before rewriting.
- Identify issue types such as missing_source, overclaim, numeric_mismatch, time_mismatch, causal_overreach, and source_conflict.
- Rewrite conservatively after verification findings.
- Remove or downgrade any unsupported or weakly supported claim.
- Numeric claims with verification failures must be removed unless directly supported elsewhere in the provided content.
- Non-numeric unsupported claims may be rewritten as uncertainty or "insufficient evidence".
- Preserve supported claims and their citations.
- Non-goals: do not add new facts, new citations, or new numbers.
"""


def _json_block(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_research_planning_prompt(
    *,
    user_query: str,
    mode: str,
    draft_contract: dict,
    steps: list[dict],
) -> str:
    return f"""Design a section-level evidence contract for this BizIntel task.

User query:
{user_query}

Mode:
{mode}

Current draft contract:
{_json_block(draft_contract)}

Planned sections:
{_json_block({"steps": steps})}

Return JSON with this shape:
{{
  "normalized_question": "...",
  "answer_type": "...",
  "plan": ["..."],
  "estimate": "...",
  "non_goals": ["..."],
  "required_slots": ["..."],
  "optional_facts": ["..."],
  "research_subquestions": ["..."],
  "preferred_source_order": ["..."],
  "output_outline": ["..."],
  "answer_rules": ["..."],
  "sections": {{
    "section_name": {{
      "required_facts": ["..."],
      "allowed_source_types": ["..."],
      "banned_claim_types": ["..."],
      "downgrade_rule": "...",
      "writing_order": ["facts", "management_commentary", "inference"],
      "fact_first": true
    }}
  }}
}}
"""


def build_query_generation_prompt(
    *,
    user_query: str,
    step_name: str,
    step_description: str,
    contract: dict,
    evidence_requirements: dict,
) -> str:
    return f"""Generate 2-4 precise retrieval queries for this research step.

User query:
{user_query}

Step:
{step_name}

Step objective:
{step_description}

Execution contract:
{_json_block(contract)}

Section evidence requirements:
{_json_block(evidence_requirements)}

Return JSON:
{{
  "queries": [
    {{
      "fact_slot": "...",
      "query_text": "...",
      "source_type": ["10-K"],
      "filters": {{
        "filing_type": "...",
        "date_range": "...",
        "section": "..."
      }},
      "expected_evidence_type": "table|numeric|quote|narrative",
      "calculation_hint": "optional variables / formula"
    }}
  ]
}}
"""


def build_evidence_organization_prompt(
    *,
    step_name: str,
    section_requirements: dict,
    context: str,
    previous_findings: str,
) -> str:
    return f"""Organize the evidence into source-tracked notes for this section.

Section:
{step_name}

Section requirements:
{_json_block(section_requirements)}

Previous findings:
{previous_findings or "None"}

Evidence context:
{context}

Return JSON:
{{
  "ledger": [
    {{
      "fact_slot": "...",
      "support_status": "supported|partially_supported|unsupported|conflicting",
      "extracted_fact": "...",
      "evidence_ids": ["chunk_id"],
      "reasoning_type": "direct_quote|numeric_calculation|inference",
      "uncertainty_note": "..."
    }}
  ]
}}
"""


def build_gap_reflection_prompt(
    *,
    user_query: str,
    step_name: str,
    covered_facts: list[str],
    missing_facts: list[str],
    evidence_notes: list[dict],
) -> str:
    return f"""Review the current evidence coverage and propose only the next retrieval queries needed.

User query:
{user_query}

Section:
{step_name}

Covered facts:
{_json_block({"covered_facts": covered_facts})}

Missing facts:
{_json_block({"missing_facts": missing_facts})}

Current evidence notes:
{_json_block({"notes": evidence_notes[:8]})}

Return JSON:
{{
  "missing_facts": ["..."],
  "confidence_assessments": [
    {{
      "fact_slot": "...",
      "confidence": 0,
      "assumption": "..."
    }}
  ],
  "follow_up_queries": ["..."],
  "stop_reason": "resolved|no_new_angle|corpus_gap"
}}
"""


def build_evidence_writing_prompt(
    *,
    mode: str,
    step_name: str,
    section_contract: dict,
    draft_notes: str,
    evidence_notes: list[dict],
    previous_findings: str,
) -> str:
    return f"""Write a citation-aware {mode} memo section from the evidence below.

Section:
{step_name}

Section contract:
{_json_block(section_contract)}

Previous findings:
{previous_findings or "None"}

Draft notes:
{draft_notes}

Evidence notes:
{_json_block({"notes": evidence_notes[:8]})}

Requirements:
- Produce JSON with keys draft, critique, and final_answer.
- Draft should answer directly and stay evidence bound.
- Critique should identify overclaims, missing evidence, ambiguous periods, or weak phrasing.
- Final answer should fix the critique without adding new facts.
- final_answer must be a markdown string, not a nested object or array.
- Facts before commentary/inference when fact_first is true.
- Every factual sentence must cite [Chunk: chunk_id] [Source: source_id].
- If a required fact is not directly supported, write "insufficient evidence".
- Keep assumptions and uncertainty explicit.
"""


def build_strict_verification_prompt(
    *,
    section_title: str,
    content: str,
    verification_findings: list[dict],
) -> str:
    return f"""Rewrite this memo section conservatively using the verification findings.

Section:
{section_title}

Current content:
{content}

Verification findings:
{_json_block({"findings": verification_findings})}

Rewrite the section so that unsupported claims are removed or downgraded, while supported claims and citations are preserved.
Return JSON with:
{{
  "claim_reviews": [
    {{
      "claim_text": "...",
      "support_status": "supported|partially_supported|unsupported|conflicting",
      "issue_type": "missing_source|overclaim|numeric_mismatch|time_mismatch|causal_overreach|source_conflict",
      "suggested_fix": "..."
    }}
  ],
  "rewritten_section": "..."
}}
"""
