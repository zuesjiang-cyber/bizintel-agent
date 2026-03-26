# 证据驱动可验证的企业财务研究Agent Flow Benchmark v1.0

## 1. Purpose

This benchmark is designed for **证据驱动可验证的企业财务研究Agent Flow** and similar business-research RAG / agent systems. It is intentionally **small but hard**: it does not aim to maximize dataset size; it aims to maximize **diagnostic value**.

The benchmark is built to evaluate whether the system can:

1. retrieve the right evidence from a company source pack,
2. synthesize a correct and complete answer,
3. compare companies without collapsing into vague generalities,
4. reason over time-sensitive changes across periods, and
5. detect or properly handle conflicting evidence.

This benchmark is optimized for the current project scope:

- offline source packs,
- chunked evidence,
- hybrid retrieval,
- structured memo generation,
- claim verification with artifacts.

It does **not** assume live web browsing.

---

## 2. Design Principles

This benchmark follows five principles:

### 2.1 Evidence-first
Every query should be answerable from the source pack. The model should not be rewarded for fluent but unsupported answers.

### 2.2 Small but diagnostic
A benchmark with 100–200 well-chosen questions is more useful than a large, noisy benchmark with shallow prompts.

### 2.3 Multi-stage evaluation
The benchmark should diagnose failures at three different layers:

- retrieval failure,
- synthesis failure,
- verification / evidence alignment failure.

### 2.4 Business-research realism
Questions reflect real research tasks rather than generic QA:

- company overview,
- risk and catalyst analysis,
- peer comparison,
- time-sensitive change detection,
- evidence conflict handling.

### 2.5 Reusable structure
The benchmark is written as a **templated markdown dataset** so it can be instantiated against different company packs.

---

## 3. Recommended Benchmark Size

This file contains **120 benchmark items** across five categories:

- 24 Company Overview
- 24 Risk & Catalyst
- 24 Comparison
- 24 Time-Sensitive
- 24 Evidence Conflict

This is a good default for a first serious benchmark run.

---

## 4. Recommended Company Universe

Use one of the following strategies.

### Option A: Single-sector universe
Choose 4–6 companies from one comparable sector.

Example sectors:
- digital payments,
- e-commerce infrastructure,
- vertical SaaS,
- semiconductors,
- cloud software.

### Option B: Core company + peers
For each target company, include:
- primary filings,
- investor letters,
- earnings call transcripts,
- management commentary,
- one or more peer companies,
- a small number of external secondary sources.

### Option C: Source-pack regression test
Use the benchmark only on companies whose source packs are already prepared in `data/processed/<company>/`.

---

## 5. Required Source Coverage Per Company

For reliable evaluation, each company should ideally include:

- 10-K / 20-F / annual report,
- latest 10-Q / quarterly report,
- most recent earnings call transcript,
- one prior-period transcript or letter,
- at least one management or shareholder letter,
- optional secondary analysis sources.

For **time-sensitive** and **conflict** questions, each company should have at least:

- two time periods,
- two distinct document types,
- at least one pair of potentially non-identical descriptions of the same topic.

---

## 6. Evaluation Tasks

Each benchmark item should be evaluated on three layers.

### 6.1 Retrieval
Did the system retrieve the required evidence?

Core retrieval checks:
- gold evidence hit in top-k,
- distinct-source coverage,
- source-type suitability,
- time relevance,
- peer relevance.

### 6.2 Generation
Did the final answer correctly synthesize the evidence?

Core generation checks:
- factual correctness,
- completeness,
- specificity,
- comparison quality,
- explicit uncertainty when evidence is weak.

### 6.3 Verification
Did the claim-verification module correctly label the answer’s claims?

Core verification checks:
- supported claims identified correctly,
- unsupported / contradicted claims flagged,
- partial support handled correctly,
- evidence-insufficient claims not overstated.

---

## 7. Scoring Rubric

Use the following rubric per item.

### 7.1 Retrieval Score (0–4)
- **4**: All key evidence retrieved; correct source type and correct period.
- **3**: Most key evidence retrieved; minor omissions.
- **2**: Partial retrieval; enough to answer partly but not fully.
- **1**: Weakly related evidence only.
- **0**: No useful evidence retrieved.

### 7.2 Answer Score (0–4)
- **4**: Correct, complete, specific, evidence-grounded.
- **3**: Mostly correct; minor incompleteness or slight overgeneralization.
- **2**: Partly correct but materially incomplete or vague.
- **1**: Mostly incorrect or unsupported.
- **0**: Non-answer or hallucinated answer.

### 7.3 Verification Score (0–4)
- **4**: Claim labels and evidence mapping are correct.
- **3**: Mostly correct with minor mislabeling.
- **2**: Mixed quality; some major mislabels.
- **1**: Mostly unreliable verification.
- **0**: Verification unusable.

### 7.4 Overall Judgment
Each item should also receive one of the following:
- **Pass**
- **Borderline**
- **Fail**

Suggested rule:
- Pass: total score >= 10 and no layer is 0
- Borderline: total score 7–9
- Fail: total score <= 6 or any critical hallucination

---

## 8. Annotation Schema

Use the following schema for each benchmark item during labeling.

```yaml
id: BO-001
category: company_overview
query: "What are {{COMPANY}}'s core business segments, and which segment contributes the largest share of revenue?"
companies:
  - "{{COMPANY}}"
required_source_types:
  - annual_report
  - quarterly_report
query_type:
  - factual_lookup
  - structured_summary
gold_answer_outline:
  - identify main segments
  - state largest segment
  - mention how management describes the business model
required_evidence:
  min_distinct_sources: 1
  must_cover:
    - segment names
    - revenue contribution or ranking
acceptable_uncertainty: false
expected_failure_modes:
  - vague segmentation summary
  - stale segment description
  - unsupported largest-segment claim
scoring_focus:
  - retrieval
  - synthesis
  - claim_verification
```

---

## 9. Recommended File Layout

```text
benchmark/
  README.md
  benchmark_v1.md
  labels/
    answers.jsonl
    evidence.jsonl
    scores.jsonl
  runs/
    <run_id>/
      raw_outputs.jsonl
      retrieval_traces.jsonl
      verification_outputs.jsonl
      summary.json
```

---

## 10. Benchmark Execution Protocol

For each benchmark item:

1. Run the system using the exact query text.
2. Save:
   - retrieved chunks,
   - ranked evidence,
   - final answer,
   - extracted claims,
   - verification output.
3. Score retrieval, answer quality, and verification quality separately.
4. Record the dominant failure mode.
5. Aggregate by category and by failure type.

---

## 11. Failure Taxonomy

Use the following failure tags during review.

### Retrieval failures
- `R1_missing_primary_source`
- `R2_wrong_period`
- `R3_wrong_peer`
- `R4_keyword_miss`
- `R5_semantic_miss`
- `R6_chunk_boundary_loss`
- `R7_table_miss`

### Synthesis failures
- `S1_incomplete_answer`
- `S2_vague_answer`
- `S3_wrong_comparison_dimension`
- `S4_temporal_confusion`
- `S5_overclaim`
- `S6_missing_uncertainty`

### Verification failures
- `V1_related_not_supporting`
- `V2_partial_as_full_support`
- `V3_number_or_time_mismatch`
- `V4_conflict_not_detected`
- `V5_insufficient_evidence_not_flagged`

---

## 12. Benchmark Item Format

Each item in this file uses the following structure:

```text
[ID] Category | Query Type | Difficulty
Query:
Primary Companies:
Required Source Types:
Gold Answer Outline:
Evaluation Focus:
Common Failure Modes:
```

---

# 13. Benchmark Items

## A. Company Overview (24 items)

### BO-001 | Company Overview | factual_lookup + structured_summary | Medium
**Query:** What are `{{COMPANY_A}}`'s core business segments, and which segment contributes the largest share of revenue?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, latest quarterly report
**Gold Answer Outline:** identify segments; state largest segment; mention business model framing
**Evaluation Focus:** retrieval of segment evidence; accurate summary; no unsupported ranking claim
**Common Failure Modes:** vague segmentation; stale segment labels; unsupported revenue-share statement

### BO-002 | Company Overview | factual_lookup | Easy
**Query:** How does `{{COMPANY_A}}` make money? Summarize the company's main revenue streams in plain business terms.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, shareholder letter
**Gold Answer Outline:** describe products/services; connect to monetization mechanics; avoid generic industry language
**Evaluation Focus:** synthesis grounded in source language
**Common Failure Modes:** empty buzzwords; failure to distinguish product from monetization

### BO-003 | Company Overview | structured_summary | Medium
**Query:** What customer groups does `{{COMPANY_A}}` primarily serve, and how does management describe its target market?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, earnings call transcript
**Gold Answer Outline:** identify customer segments; explain target-market framing
**Evaluation Focus:** cross-source consistency; specificity
**Common Failure Modes:** generic customer description; mixing users with customers

### BO-004 | Company Overview | factual_lookup | Medium
**Query:** What is the geographic mix of `{{COMPANY_A}}`'s business, and which regions matter most to revenue growth?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, latest quarterly report
**Gold Answer Outline:** identify geographic regions; highlight major regions; distinguish scale vs growth if available
**Evaluation Focus:** numerical grounding; no invented regional ranking
**Common Failure Modes:** geography omitted; growth regions confused with revenue base

### BO-005 | Company Overview | analytical_summary | Medium
**Query:** Describe `{{COMPANY_A}}`'s business model in one paragraph as if briefing a new equity research associate.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, shareholder letter, transcript
**Gold Answer Outline:** value proposition; monetization; customer base; delivery model; strategic positioning
**Evaluation Focus:** concise but complete synthesis
**Common Failure Modes:** too generic; no monetization logic

### BO-006 | Company Overview | factual_lookup | Medium
**Query:** Which products or services appear to be most strategically important to `{{COMPANY_A}}`, based on management commentary?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** earnings call transcript, shareholder letter
**Gold Answer Outline:** identify priority offerings; explain why management emphasizes them
**Evaluation Focus:** management-language grounding
**Common Failure Modes:** confusing current revenue leaders with strategic priorities

### BO-007 | Company Overview | structured_summary | Medium
**Query:** What does `{{COMPANY_A}}` disclose about its sales channels, go-to-market model, or distribution approach?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript
**Gold Answer Outline:** direct sales / partners / platform / self-serve channels as applicable
**Evaluation Focus:** channel-specific retrieval
**Common Failure Modes:** generic GTM statement without source support

### BO-008 | Company Overview | factual_lookup | Easy
**Query:** What are the main cost buckets or operating expense categories that matter for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, quarterly report
**Gold Answer Outline:** major opex or cost categories; business relevance
**Evaluation Focus:** accounting-grounded answer
**Common Failure Modes:** listing non-material or invented categories

### BO-009 | Company Overview | analytical_summary | Medium
**Query:** Does `{{COMPANY_A}}` appear to be more transaction-driven, subscription-driven, usage-driven, or asset-driven? Explain using company disclosures.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript
**Gold Answer Outline:** classify monetization style; justify with evidence
**Evaluation Focus:** inferential but evidence-backed classification
**Common Failure Modes:** unsupported classification

### BO-010 | Company Overview | factual_lookup | Medium
**Query:** What non-GAAP or operating metrics does `{{COMPANY_A}}` emphasize most, and what do those metrics suggest about how management runs the business?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** shareholder letter, earnings call transcript
**Gold Answer Outline:** identify metrics; explain managerial focus implied by those metrics
**Evaluation Focus:** metric extraction + interpretation
**Common Failure Modes:** confusing reported KPI with investor-created KPI

### BO-011 | Company Overview | structured_summary | Medium
**Query:** What does `{{COMPANY_A}}` disclose about customer concentration, merchant concentration, or enterprise dependence?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, risk factors
**Gold Answer Outline:** summarize concentration disclosures or absence thereof
**Evaluation Focus:** risk-aware retrieval
**Common Failure Modes:** hallucinating concentration risk when not disclosed

### BO-012 | Company Overview | factual_lookup | Medium
**Query:** What role does international expansion play in `{{COMPANY_A}}`'s strategy?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript
**Gold Answer Outline:** strategic importance; region-specific comments if present
**Evaluation Focus:** strategy synthesis
**Common Failure Modes:** generic expansion story not tied to sources

### BO-013 | Company Overview | analytical_summary | Medium
**Query:** Summarize `{{COMPANY_A}}`'s business in terms of customers, product, monetization, and key operating metrics.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript, shareholder letter
**Gold Answer Outline:** four-part structured summary
**Evaluation Focus:** completeness and structure
**Common Failure Modes:** answer misses one or more dimensions

### BO-014 | Company Overview | factual_lookup | Easy
**Query:** What are the most important revenue disclosures for understanding `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, segment note
**Gold Answer Outline:** segment, product, geography, or customer-type disclosures most relevant
**Evaluation Focus:** accounting disclosure awareness
**Common Failure Modes:** vague answer without actual disclosures

### BO-015 | Company Overview | structured_summary | Medium
**Query:** Based on company materials, is `{{COMPANY_A}}` positioned more as infrastructure, application software, marketplace, or financial intermediary?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, management commentary
**Gold Answer Outline:** choose best fit; explain with evidence
**Evaluation Focus:** evidence-backed categorization
**Common Failure Modes:** category chosen without support

### BO-016 | Company Overview | factual_lookup | Medium
**Query:** What does `{{COMPANY_A}}` say about its pricing model or pricing levers?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript
**Gold Answer Outline:** transaction fee / subscription / tiering / usage / blended pricing as applicable
**Evaluation Focus:** monetization precision
**Common Failure Modes:** generic pricing answer from industry assumptions

### BO-017 | Company Overview | analytical_summary | Medium
**Query:** How would you explain `{{COMPANY_A}}`'s value proposition to a CFO deciding whether to adopt its products?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** management commentary, product discussion
**Gold Answer Outline:** customer problem; offered solution; measurable business benefit
**Evaluation Focus:** user-centric but source-based explanation
**Common Failure Modes:** marketing-style fluff

### BO-018 | Company Overview | factual_lookup | Medium
**Query:** What does management highlight as the most differentiated aspect of `{{COMPANY_A}}`'s offering?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, shareholder letter
**Gold Answer Outline:** extract differentiation theme; connect to customer or economics
**Evaluation Focus:** management-language fidelity
**Common Failure Modes:** substituting analyst opinion for management framing

### BO-019 | Company Overview | structured_summary | Medium
**Query:** Which business lines appear mature, and which appear to still be in investment mode at `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, MD&A, shareholder letter
**Gold Answer Outline:** identify mature vs investment areas if evidence exists; state uncertainty if not explicit
**Evaluation Focus:** evidence-calibrated inference
**Common Failure Modes:** overconfident maturity assessment

### BO-020 | Company Overview | factual_lookup | Easy
**Query:** What does `{{COMPANY_A}}` disclose about gross margin drivers or unit economics drivers?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript
**Gold Answer Outline:** cost of revenue drivers; mix effects; pricing vs cost comments
**Evaluation Focus:** business-economics grounding
**Common Failure Modes:** confusing gross margin with operating margin

### BO-021 | Company Overview | structured_summary | Medium
**Query:** What are the top three things an investor must understand before analyzing `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** broad pack: annual report, transcript, letter
**Gold Answer Outline:** concise priorities anchored in source pack
**Evaluation Focus:** synthesis quality
**Common Failure Modes:** arbitrary priorities not source-backed

### BO-022 | Company Overview | factual_lookup | Medium
**Query:** What acquisitions, platform expansions, or product launches appear most important to `{{COMPANY_A}}`'s current business model?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript, investor presentation if available
**Gold Answer Outline:** identify key strategic additions and why they matter
**Evaluation Focus:** event-to-business-model linkage
**Common Failure Modes:** listing events without strategic relevance

### BO-023 | Company Overview | structured_summary | Medium
**Query:** What does `{{COMPANY_A}}` say about retention, customer engagement, or recurring usage quality?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** shareholder letter, transcript
**Gold Answer Outline:** identify retention or engagement indicators and management framing
**Evaluation Focus:** KPI retrieval and interpretation
**Common Failure Modes:** invented retention metrics

### BO-024 | Company Overview | analytical_summary | Hard
**Query:** Build a concise investment-context overview of `{{COMPANY_A}}` covering business model, scale drivers, margin structure, and strategic priorities.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, quarterly report, transcript, letter
**Gold Answer Outline:** four-part business brief
**Evaluation Focus:** full-stack overview synthesis
**Common Failure Modes:** incomplete answer; unsupported strategic claims

---

## B. Risk & Catalyst (24 items)

### RC-001 | Risk & Catalyst | risk_extraction | Medium
**Query:** What are the most important regulatory risks disclosed by `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** risk factors, annual report
**Gold Answer Outline:** identify key regulatory risks; avoid generic risk boilerplate unless disclosed as material
**Evaluation Focus:** primary-source risk extraction
**Common Failure Modes:** generic compliance risks without evidence

### RC-002 | Risk & Catalyst | catalyst_identification | Medium
**Query:** What growth drivers does management emphasize most for `{{COMPANY_A}}` over the next 12–24 months?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** earnings call transcript, shareholder letter
**Gold Answer Outline:** 2–4 management-highlighted growth drivers
**Evaluation Focus:** management framing and prioritization
**Common Failure Modes:** mixing company drivers with sector-wide narratives

### RC-003 | Risk & Catalyst | margin_analysis | Medium
**Query:** What are the main sources of margin pressure for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, quarterly report, MD&A
**Gold Answer Outline:** identify cost, pricing, mix, investment, or credit-related pressures
**Evaluation Focus:** economics-aware synthesis
**Common Failure Modes:** gross vs operating margin confusion

### RC-004 | Risk & Catalyst | risk_extraction | Medium
**Query:** What customer, merchant, or end-market concentration risks matter most for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, risk factors
**Gold Answer Outline:** concentration risk if disclosed; otherwise explicit uncertainty
**Evaluation Focus:** disciplined use of evidence
**Common Failure Modes:** hallucinated concentration exposure

### RC-005 | Risk & Catalyst | catalyst_identification | Medium
**Query:** Which new products or strategic initiatives could act as upside catalysts for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, shareholder letter, investor presentation if available
**Gold Answer Outline:** identify initiatives; explain channel to revenue or margin impact
**Evaluation Focus:** strategy-to-financial-link reasoning
**Common Failure Modes:** listing launches without explaining why they matter

### RC-006 | Risk & Catalyst | risk_extraction | Medium
**Query:** What execution risks does management implicitly or explicitly acknowledge for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, letter, risk factors
**Gold Answer Outline:** implementation, scaling, internationalization, product, sales execution, integration risks
**Evaluation Focus:** explicit + implied risk reading
**Common Failure Modes:** overstating implied risks as direct admissions

### RC-007 | Risk & Catalyst | economics_analysis | Hard
**Query:** What could cause `{{COMPANY_A}}`'s take rate, ARPU, or monetization efficiency to deteriorate?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript, KPI commentary
**Gold Answer Outline:** pricing pressure, mix shift, larger customer mix, competition, regulation, product mix
**Evaluation Focus:** business-model-specific risk reasoning
**Common Failure Modes:** generic “competition may hurt pricing” with no grounding

### RC-008 | Risk & Catalyst | catalyst_identification | Medium
**Query:** What evidence suggests that `{{COMPANY_A}}` still has meaningful cross-sell or wallet-share expansion opportunities?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, shareholder letter
**Gold Answer Outline:** cross-sell themes, attach rates, platform expansion logic
**Evaluation Focus:** opportunity extraction from management commentary
**Common Failure Modes:** platform buzzwords without evidence

### RC-009 | Risk & Catalyst | risk_extraction | Medium
**Query:** What macroeconomic risks matter most for `{{COMPANY_A}}`, according to company disclosures?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, quarterly report, transcript
**Gold Answer Outline:** demand, spending, FX, rates, credit, SMB health, enterprise budgets as applicable
**Evaluation Focus:** disclosed macro sensitivity
**Common Failure Modes:** adding macro risks not reflected in source pack

### RC-010 | Risk & Catalyst | catalyst_identification | Medium
**Query:** Which operating metrics would you monitor most closely to see whether `{{COMPANY_A}}`'s bullish case is working?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** management commentary, KPI disclosures
**Gold Answer Outline:** identify 2–4 KPIs and explain why they matter
**Evaluation Focus:** KPI-to-thesis linkage
**Common Failure Modes:** choosing metrics not emphasized by company

### RC-011 | Risk & Catalyst | risk_extraction | Hard
**Query:** What are the strongest reasons to be skeptical of `{{COMPANY_A}}`'s current growth narrative?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, quarterly report, prior period commentary
**Gold Answer Outline:** evidence-based skepticism using disclosed constraints, deceleration, or tough comps
**Evaluation Focus:** balanced analytical skepticism
**Common Failure Modes:** unsupported short thesis language

### RC-012 | Risk & Catalyst | catalyst_identification | Hard
**Query:** What is the highest-conviction upside catalyst for `{{COMPANY_A}}` that is actually supported by the source pack?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, shareholder letter, product commentary
**Gold Answer Outline:** choose one catalyst; justify with evidence and mechanism
**Evaluation Focus:** precision and selectivity
**Common Failure Modes:** naming several catalysts without prioritization

### RC-013 | Risk & Catalyst | risk_extraction | Medium
**Query:** What does `{{COMPANY_A}}` disclose about competitive pressure, and where could competition hurt the economics most?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript
**Gold Answer Outline:** competition areas + economic effect
**Evaluation Focus:** risk-to-economics linkage
**Common Failure Modes:** generic competition statements

### RC-014 | Risk & Catalyst | catalyst_identification | Medium
**Query:** Does management suggest that enterprise expansion, international expansion, or product breadth is the bigger growth lever for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, shareholder letter
**Gold Answer Outline:** choose the emphasized lever; state evidence and uncertainty
**Evaluation Focus:** prioritization from management commentary
**Common Failure Modes:** reporting all three as equal without evidence

### RC-015 | Risk & Catalyst | economics_analysis | Hard
**Query:** What could stop `{{COMPANY_A}}` from converting revenue growth into operating leverage?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** quarterly report, transcript
**Gold Answer Outline:** cost structure, reinvestment, sales efficiency, infrastructure cost, credit losses, support cost
**Evaluation Focus:** scaling-economics understanding
**Common Failure Modes:** simplistic “costs may rise” answer

### RC-016 | Risk & Catalyst | risk_extraction | Medium
**Query:** Which disclosed risks seem most underappreciated relative to management's optimistic messaging for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** risk factors, shareholder letter, transcript
**Gold Answer Outline:** identify tension between risk disclosures and optimistic framing
**Evaluation Focus:** cross-document tension analysis
**Common Failure Modes:** speculative criticism not tied to documents

### RC-017 | Risk & Catalyst | catalyst_identification | Medium
**Query:** What evidence suggests `{{COMPANY_A}}` may be early in a new monetization cycle or product adoption cycle?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, letter
**Gold Answer Outline:** early-stage signals with evidence
**Evaluation Focus:** forward-looking but grounded interpretation
**Common Failure Modes:** calling every new product an adoption cycle

### RC-018 | Risk & Catalyst | risk_extraction | Medium
**Query:** What does the source pack suggest about sales execution risk or go-to-market risk at `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, annual report, management commentary
**Gold Answer Outline:** sales hiring, enterprise cycle, channel risk, integration of GTM changes
**Evaluation Focus:** operational risk reading
**Common Failure Modes:** invented sales issues

### RC-019 | Risk & Catalyst | catalyst_identification | Medium
**Query:** What is the best evidence that `{{COMPANY_A}}` has a durable expansion runway rather than a short-term cyclical bump?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript, prior-period materials
**Gold Answer Outline:** structural vs cyclical growth evidence
**Evaluation Focus:** thesis quality
**Common Failure Modes:** confusing cyclical rebound with structural runway

### RC-020 | Risk & Catalyst | economics_analysis | Medium
**Query:** Which cost lines or operating choices matter most for protecting margins at `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** quarterly report, transcript
**Gold Answer Outline:** identify key controllable economics drivers
**Evaluation Focus:** margin-protection reasoning
**Common Failure Modes:** non-material cost drivers emphasized

### RC-021 | Risk & Catalyst | risk_extraction | Hard
**Query:** If `{{COMPANY_A}}` misses expectations, what does the source pack imply is the most likely reason?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, risk factors, recent quarter commentary
**Gold Answer Outline:** most plausible miss driver based on disclosures
**Evaluation Focus:** evidence-backed downside scenario reasoning
**Common Failure Modes:** speculative narrative detached from source pack

### RC-022 | Risk & Catalyst | catalyst_identification | Medium
**Query:** What specific evidence supports the case that `{{COMPANY_A}}` can move up-market or deepen enterprise penetration?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, product commentary, customer examples if available
**Gold Answer Outline:** enterprise traction evidence; deal-size or product breadth signals
**Evaluation Focus:** traction evidence quality
**Common Failure Modes:** assuming enterprise strategy without proof

### RC-023 | Risk & Catalyst | risk_extraction | Hard
**Query:** What appears to be the hardest part of `{{COMPANY_A}}`'s strategy to execute successfully?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, letter, risk factors
**Gold Answer Outline:** identify one or two execution bottlenecks with evidence
**Evaluation Focus:** strategic realism
**Common Failure Modes:** vague “execution risk exists” answer

### RC-024 | Risk & Catalyst | catalyst_identification | Hard
**Query:** Summarize the top three downside risks and top three upside catalysts for `{{COMPANY_A}}` in a balanced research-note style.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, transcript, letter, quarterly report
**Gold Answer Outline:** 3 risks + 3 catalysts with evidence-backed rationale
**Evaluation Focus:** balanced synthesis
**Common Failure Modes:** one-sided framing; unsupported items

---

## C. Comparison (24 items)

### CP-001 | Comparison | peer_comparison | Medium
**Query:** Compare `{{COMPANY_A}}` and `{{COMPANY_B}}` on business model and monetization structure.
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports for both companies
**Gold Answer Outline:** clear side-by-side comparison of product, customer, monetization
**Evaluation Focus:** true comparison rather than two separate mini-summaries
**Common Failure Modes:** no comparative framing

### CP-002 | Comparison | peer_comparison | Medium
**Query:** Which company appears more transaction-driven: `{{COMPANY_A}}` or `{{COMPANY_B}}`, and why?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, metric disclosures
**Gold Answer Outline:** compare monetization mechanics using evidence
**Evaluation Focus:** classification with comparative reasoning
**Common Failure Modes:** unsupported binary conclusion

### CP-003 | Comparison | economics_comparison | Medium
**Query:** How do `{{COMPANY_A}}` and `{{COMPANY_B}}` differ in gross margin structure or the main drivers of gross margin?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, quarterly reports
**Gold Answer Outline:** compare margin drivers, not just margin numbers if unavailable
**Evaluation Focus:** accounting + business reasoning
**Common Failure Modes:** comparing non-equivalent metrics

### CP-004 | Comparison | strategy_comparison | Medium
**Query:** Compare `{{COMPANY_A}}` and `{{COMPANY_B}}` on their go-to-market model and customer focus.
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, transcripts
**Gold Answer Outline:** GTM and customer segmentation differences
**Evaluation Focus:** structured comparison
**Common Failure Modes:** generic enterprise vs SMB claims without support

### CP-005 | Comparison | risk_comparison | Hard
**Query:** Which company appears more exposed to regulatory risk: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** risk factors for both companies
**Gold Answer Outline:** compare risk intensity and nature; state uncertainty if hard to rank
**Evaluation Focus:** cautious comparison
**Common Failure Modes:** overconfident ranking without basis

### CP-006 | Comparison | growth_comparison | Medium
**Query:** Compare the main growth drivers highlighted by management at `{{COMPANY_A}}` and `{{COMPANY_B}}`.
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** earnings call transcripts, shareholder letters
**Gold Answer Outline:** side-by-side driver comparison
**Evaluation Focus:** management-language extraction + comparison
**Common Failure Modes:** blending both companies' narratives together

### CP-007 | Comparison | economics_comparison | Hard
**Query:** Which company appears to have better operating leverage potential: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** quarterly reports, transcripts
**Gold Answer Outline:** discuss cost structure, reinvestment profile, margin trajectory
**Evaluation Focus:** evidence-backed inferential comparison
**Common Failure Modes:** pure opinion answer

### CP-008 | Comparison | product_comparison | Medium
**Query:** How do `{{COMPANY_A}}` and `{{COMPANY_B}}` differ in product breadth and platform strategy?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, letters, product commentary
**Gold Answer Outline:** breadth, platform approach, attach opportunities
**Evaluation Focus:** product-strategy comparison
**Common Failure Modes:** using external assumptions instead of source pack

### CP-009 | Comparison | customer_comparison | Medium
**Query:** Which company is more enterprise-oriented, and which is more SMB-oriented: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, transcripts
**Gold Answer Outline:** compare customer focus and sales motion
**Evaluation Focus:** audience segmentation accuracy
**Common Failure Modes:** relying on brand impressions rather than evidence

### CP-010 | Comparison | metric_comparison | Medium
**Query:** Which operating metrics matter most for analyzing `{{COMPANY_A}}` versus `{{COMPANY_B}}`, and why are they not the same?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** KPI disclosures, shareholder letters
**Gold Answer Outline:** compare KPI sets and managerial focus
**Evaluation Focus:** metric-aware company differentiation
**Common Failure Modes:** same KPI list for both firms

### CP-011 | Comparison | risk_comparison | Hard
**Query:** Compare the biggest execution risks facing `{{COMPANY_A}}` and `{{COMPANY_B}}`.
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** risk factors, transcripts
**Gold Answer Outline:** pairwise execution-risk comparison
**Evaluation Focus:** nuanced difference identification
**Common Failure Modes:** generic “both face execution risk” answer

### CP-012 | Comparison | strategy_comparison | Medium
**Query:** Which company appears more dependent on international expansion for future growth: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, transcripts
**Gold Answer Outline:** compare international strategy emphasis
**Evaluation Focus:** strategic emphasis ranking
**Common Failure Modes:** unsupported dependence claim

### CP-013 | Comparison | economics_comparison | Hard
**Query:** Compare `{{COMPANY_A}}` and `{{COMPANY_B}}` on margin quality rather than just margin level.
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** quarterly reports, annual reports, transcripts
**Gold Answer Outline:** discuss sustainability, mix, reinvestment, quality of earnings drivers
**Evaluation Focus:** deeper economics comparison
**Common Failure Modes:** only citing raw percentage differences

### CP-014 | Comparison | product_comparison | Medium
**Query:** Which company seems more differentiated on product capability, and which seems more differentiated on distribution or ecosystem?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** letters, transcripts, annual reports
**Gold Answer Outline:** separate product vs distribution differentiation
**Evaluation Focus:** differentiation logic
**Common Failure Modes:** collapsing both dimensions into one answer

### CP-015 | Comparison | catalyst_comparison | Medium
**Query:** Compare the most credible upside catalysts for `{{COMPANY_A}}` and `{{COMPANY_B}}`.
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** transcripts, letters
**Gold Answer Outline:** identify one or two credible catalysts per company
**Evaluation Focus:** company-specific upside logic
**Common Failure Modes:** reusing the same catalyst language for both firms

### CP-016 | Comparison | strategic_positioning | Medium
**Query:** Which company appears better positioned to cross-sell adjacent products: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** product commentary, transcript, letter
**Gold Answer Outline:** compare platform breadth and customer embedment
**Evaluation Focus:** cross-sell reasoning
**Common Failure Modes:** unsupported platform superiority claim

### CP-017 | Comparison | economics_comparison | Hard
**Query:** Which company looks more vulnerable to mix shift pressure on monetization: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** KPI commentary, annual reports, transcripts
**Gold Answer Outline:** compare exposure to enterprise mix, geography mix, product mix, low-margin volume mix
**Evaluation Focus:** nuanced economics comparison
**Common Failure Modes:** no actual mix-shift evidence

### CP-018 | Comparison | risk_comparison | Hard
**Query:** If you had to choose, which company carries the harder strategic execution burden over the next two years: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** transcripts, risk factors, letters
**Gold Answer Outline:** select one with evidence-backed reasoning; uncertainty allowed
**Evaluation Focus:** comparative strategic realism
**Common Failure Modes:** unsupported strong preference

### CP-019 | Comparison | summary_comparison | Medium
**Query:** Summarize the most important differences between `{{COMPANY_A}}` and `{{COMPANY_B}}` in a short analyst-style compare/contrast note.
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, transcripts, letters
**Gold Answer Outline:** 3–5 key differences
**Evaluation Focus:** concise comparison quality
**Common Failure Modes:** two isolated summaries instead of compare/contrast

### CP-020 | Comparison | time-aware_comparison | Hard
**Query:** Which company appears to be improving faster operationally: `{{COMPANY_A}}` or `{{COMPANY_B}}`, based on recent disclosures?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** latest and prior quarter materials for both companies
**Gold Answer Outline:** compare recent trajectory, not static quality
**Evaluation Focus:** cross-company temporal reasoning
**Common Failure Modes:** static instead of trend-based comparison

### CP-021 | Comparison | metric_comparison | Medium
**Query:** Which company relies more on scale, and which relies more on monetization depth to drive growth: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** KPI disclosures, management commentary
**Gold Answer Outline:** compare growth architecture
**Evaluation Focus:** growth-driver differentiation
**Common Failure Modes:** vague answer not tied to metrics

### CP-022 | Comparison | strategic_positioning | Hard
**Query:** Compare how `{{COMPANY_A}}` and `{{COMPANY_B}}` position themselves within their value chain. Which is closer to infrastructure, and which is closer to a full-stack operating layer?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, letters, management commentary
**Gold Answer Outline:** value-chain positioning comparison
**Evaluation Focus:** strategic taxonomy with evidence
**Common Failure Modes:** category language without explanation

### CP-023 | Comparison | risk_catalyst_comparison | Hard
**Query:** Which company offers the cleaner bull case, and which company carries the messier risk profile: `{{COMPANY_A}}` or `{{COMPANY_B}}`?
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** broad pack for both companies
**Gold Answer Outline:** balanced case comparison; may conclude unclear if evidence mixed
**Evaluation Focus:** balanced conviction under uncertainty
**Common Failure Modes:** one-line preference without evidence

### CP-024 | Comparison | multi_dimensional_comparison | Hard
**Query:** Compare `{{COMPANY_A}}` and `{{COMPANY_B}}` across business model, growth drivers, margin structure, risk profile, and strategic optionality.
**Primary Companies:** `{{COMPANY_A}}`, `{{COMPANY_B}}`
**Required Source Types:** annual reports, quarterly reports, transcripts, letters for both
**Gold Answer Outline:** five-dimension structured comparison
**Evaluation Focus:** full-stack compare/contrast ability
**Common Failure Modes:** incomplete dimensions; weak evidence grounding

---

## D. Time-Sensitive (24 items)

### TS-001 | Time-Sensitive | change_detection | Medium
**Query:** What changed most in `{{COMPANY_A}}`'s latest quarter versus the prior quarter, based on management commentary?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** latest and prior quarter transcripts
**Gold Answer Outline:** identify major change(s); tie to commentary
**Evaluation Focus:** temporal retrieval and comparison
**Common Failure Modes:** static description of company instead of change

### TS-002 | Time-Sensitive | trend_analysis | Medium
**Query:** Has management's tone on growth at `{{COMPANY_A}}` become more confident, more cautious, or largely unchanged over the last two periods?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period transcripts or letters
**Gold Answer Outline:** tone shift with textual evidence
**Evaluation Focus:** time-aware narrative analysis
**Common Failure Modes:** tone judgment without quotes or evidence basis

### TS-003 | Time-Sensitive | change_detection | Medium
**Query:** Which business segment, product line, or region appears to be changing in strategic importance for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period disclosures
**Gold Answer Outline:** detect shifts in emphasis or contribution
**Evaluation Focus:** priority-shift retrieval
**Common Failure Modes:** assuming importance changed because of one mention

### TS-004 | Time-Sensitive | metric_trend | Medium
**Query:** Which KPIs does management emphasize more now than it did in the prior period for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** prior and latest letters or transcripts
**Gold Answer Outline:** compare KPI emphasis changes
**Evaluation Focus:** discourse shift analysis
**Common Failure Modes:** KPI list with no temporal comparison

### TS-005 | Time-Sensitive | strategy_shift | Hard
**Query:** Has `{{COMPANY_A}}`'s strategy narrative shifted from growth to profitability, from expansion to focus, or in some other meaningful direction?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** multi-period management commentary
**Gold Answer Outline:** identify strategy shift or explicitly say no clear shift
**Evaluation Focus:** strategic time reasoning
**Common Failure Modes:** over-reading normal quarterly wording changes

### TS-006 | Time-Sensitive | change_detection | Medium
**Query:** What is new in `{{COMPANY_A}}`'s latest product or platform messaging relative to the previous period?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period transcripts or letters
**Gold Answer Outline:** new messaging, launches, reprioritization
**Evaluation Focus:** product narrative change detection
**Common Failure Modes:** listing latest features without prior-period comparison

### TS-007 | Time-Sensitive | margin_trend | Medium
**Query:** Has management's explanation for margin pressure or margin improvement changed recently at `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** latest and prior quarter materials
**Gold Answer Outline:** compare margin explanation across periods
**Evaluation Focus:** causal narrative comparison
**Common Failure Modes:** only describing latest quarter

### TS-008 | Time-Sensitive | risk_change | Hard
**Query:** Which risk appears more salient in the latest period than before for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period risk discussions or transcripts
**Gold Answer Outline:** identify increased salience with evidence
**Evaluation Focus:** temporal risk reading
**Common Failure Modes:** selecting a risk that is always present but not more salient

### TS-009 | Time-Sensitive | guidance_interpretation | Medium
**Query:** How has management's framing of near-term demand changed for `{{COMPANY_A}}` over the last two reporting periods?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** multi-period transcripts
**Gold Answer Outline:** compare demand tone and drivers
**Evaluation Focus:** temporal language interpretation
**Common Failure Modes:** equating tone with financial results only

### TS-010 | Time-Sensitive | change_detection | Medium
**Query:** What has become less important in the latest discussion of `{{COMPANY_A}}`, compared with prior materials?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period letters or transcripts
**Gold Answer Outline:** identify de-emphasized themes cautiously
**Evaluation Focus:** omission-aware temporal reasoning
**Common Failure Modes:** inferring too much from absence alone

### TS-011 | Time-Sensitive | investment_cycle | Hard
**Query:** Is `{{COMPANY_A}}` moving deeper into investment mode, or beginning to harvest operating leverage, relative to the prior period?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** quarterly reports, transcripts
**Gold Answer Outline:** compare investment stance across periods
**Evaluation Focus:** operational trajectory analysis
**Common Failure Modes:** unsupported operating leverage narrative

### TS-012 | Time-Sensitive | strategy_shift | Medium
**Query:** Has the company's international strategy messaging changed recently for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period transcripts or annual + latest quarter materials
**Gold Answer Outline:** compare international emphasis, markets, or rationale
**Evaluation Focus:** strategy shift identification
**Common Failure Modes:** static strategy summary

### TS-013 | Time-Sensitive | change_detection | Medium
**Query:** What does the latest quarter suggest about whether `{{COMPANY_A}}`'s earlier growth thesis is holding up?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** prior period thesis materials + latest quarter materials
**Gold Answer Outline:** evaluate whether earlier thesis evidence is confirmed, weakened, or mixed
**Evaluation Focus:** thesis tracking over time
**Common Failure Modes:** no explicit bridge between earlier thesis and latest evidence

### TS-014 | Time-Sensitive | tone_comparison | Medium
**Query:** Compare management's latest tone on enterprise demand, consumer demand, or merchant health with the previous period for `{{COMPANY_A}}`.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period transcripts
**Gold Answer Outline:** demand-segment tone comparison
**Evaluation Focus:** targeted temporal narrative analysis
**Common Failure Modes:** failing to specify demand segment

### TS-015 | Time-Sensitive | metric_trend | Hard
**Query:** Which metric or disclosure would most strongly tell you whether `{{COMPANY_A}}` is executing better now than it was one period ago?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** KPI disclosures across two periods
**Gold Answer Outline:** choose one metric and justify temporally
**Evaluation Focus:** execution-sensitive KPI reasoning
**Common Failure Modes:** choosing a generic metric without temporal relevance

### TS-016 | Time-Sensitive | change_detection | Medium
**Query:** Has management become more explicit about pricing, monetization, or cost discipline recently at `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period management commentaries
**Gold Answer Outline:** compare explicitness and emphasis
**Evaluation Focus:** nuanced narrative delta detection
**Common Failure Modes:** answering from only one period

### TS-017 | Time-Sensitive | priority_shift | Hard
**Query:** What strategic priority moved up the agenda for `{{COMPANY_A}}` in the latest period, and what moved down?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period transcripts or letters
**Gold Answer Outline:** one moved up + one moved down if supported; otherwise say unclear
**Evaluation Focus:** comparative prioritization
**Common Failure Modes:** unsupported moved-down inference

### TS-018 | Time-Sensitive | change_detection | Medium
**Query:** What is the clearest sign that `{{COMPANY_A}}`'s management narrative is evolving rather than simply repeating prior messaging?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** two period transcripts or letters
**Gold Answer Outline:** identify the strongest narrative evolution signal
**Evaluation Focus:** discourse-level temporal analysis
**Common Failure Modes:** superficial word-count style answer

### TS-019 | Time-Sensitive | trend_analysis | Hard
**Query:** Over the last two periods, has `{{COMPANY_A}}`'s story become simpler and more focused, or more complex and multi-threaded?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** multi-period commentary
**Gold Answer Outline:** characterize narrative complexity change with evidence
**Evaluation Focus:** higher-level temporal synthesis
**Common Failure Modes:** subjective answer without document grounding

### TS-020 | Time-Sensitive | execution_tracking | Medium
**Query:** Which previously announced initiative appears to be showing the most visible progress in the latest period for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** prior announcement + latest update
**Gold Answer Outline:** identify initiative and evidence of progress
**Evaluation Focus:** initiative tracking
**Common Failure Modes:** choosing a new initiative rather than previously announced one

### TS-021 | Time-Sensitive | risk_change | Medium
**Query:** Has the company's discussion of competitive pressure changed meaningfully from the previous period for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** multi-period transcripts or risk discussions
**Gold Answer Outline:** compare competition framing across periods
**Evaluation Focus:** temporal competitive analysis
**Common Failure Modes:** generic competition summary

### TS-022 | Time-Sensitive | thesis_update | Hard
**Query:** If you wrote a research note one period ago, what single paragraph would you update most aggressively today for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** prior and latest period materials
**Gold Answer Outline:** identify the research-note section most changed by new evidence
**Evaluation Focus:** analyst workflow realism
**Common Failure Modes:** no explicit before-vs-now logic

### TS-023 | Time-Sensitive | comparative_change | Hard
**Query:** Is the latest period at `{{COMPANY_A}}` best described as acceleration, deceleration, stabilization, or repositioning? Explain.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** latest and prior period materials
**Gold Answer Outline:** choose one label and justify with temporal evidence
**Evaluation Focus:** concise but grounded temporal classification
**Common Failure Modes:** classification not tied to concrete evidence

### TS-024 | Time-Sensitive | multi_dimension_change | Hard
**Query:** Summarize the three most important changes in `{{COMPANY_A}}`'s business narrative, operating trajectory, or strategic emphasis over the last two periods.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** prior and latest transcripts, letters, quarterly materials
**Gold Answer Outline:** 3 temporal changes across business/strategy/operations
**Evaluation Focus:** full-stack temporal synthesis
**Common Failure Modes:** latest-period summary with no true comparison

---

## E. Evidence Conflict (24 items)

### EC-001 | Evidence Conflict | conflict_detection | Hard
**Query:** Do different sources describe `{{COMPANY_A}}`'s main growth driver differently? If so, what is the difference, and how should it be reconciled?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** at least two source types, e.g. letter + transcript
**Gold Answer Outline:** detect difference in framing; explain whether it is true conflict or emphasis difference
**Evaluation Focus:** conflict vs nuance discrimination
**Common Failure Modes:** treating all wording differences as conflict

### EC-002 | Evidence Conflict | conflict_detection | Hard
**Query:** Is there any tension between `{{COMPANY_A}}`'s risk disclosures and management's optimistic commentary on the same topic?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** risk factors + transcript or shareholder letter
**Gold Answer Outline:** identify tension if present; quote both sides conceptually
**Evaluation Focus:** cross-document tension analysis
**Common Failure Modes:** unsupported accusation of contradiction

### EC-003 | Evidence Conflict | claim_reconciliation | Hard
**Query:** If one source suggests margin improvement and another suggests ongoing pressure at `{{COMPANY_A}}`, how should an analyst reconcile the two?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** at least two period or two source discussions of margins
**Gold Answer Outline:** reconcile based on gross vs operating, short-term vs long-term, or segment mix as applicable
**Evaluation Focus:** nuanced reconciliation
**Common Failure Modes:** picking one source and ignoring the other

### EC-004 | Evidence Conflict | discrepancy_analysis | Hard
**Query:** Do different disclosures imply different customer mixes or strategic priorities for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report + transcript or letter
**Gold Answer Outline:** identify whether difference is real, contextual, or non-material
**Evaluation Focus:** discrepancy classification
**Common Failure Modes:** overclaiming inconsistency

### EC-005 | Evidence Conflict | claim_reconciliation | Hard
**Query:** If management sounds bullish but KPIs look mixed for `{{COMPANY_A}}`, what is the fairest evidence-based conclusion?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** management commentary + KPI disclosures
**Gold Answer Outline:** balanced conclusion acknowledging mixed evidence
**Evaluation Focus:** balanced synthesis under tension
**Common Failure Modes:** blindly following management tone

### EC-006 | Evidence Conflict | conflict_detection | Medium
**Query:** Are there any topics where `{{COMPANY_A}}`'s annual report language is more cautious than the earnings call language?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report + earnings call transcript
**Gold Answer Outline:** identify topic(s) and difference in tone if present
**Evaluation Focus:** source-type sensitivity
**Common Failure Modes:** claiming caution difference without examples

### EC-007 | Evidence Conflict | claim_reconciliation | Hard
**Query:** If one source emphasizes product breadth and another emphasizes focused execution for `{{COMPANY_A}}`, are these contradictory or complementary?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** letter + transcript or annual report
**Gold Answer Outline:** classify as contradiction or complementarity with explanation
**Evaluation Focus:** semantic conflict handling
**Common Failure Modes:** simplistic contradiction label

### EC-008 | Evidence Conflict | discrepancy_analysis | Hard
**Query:** Does `{{COMPANY_A}}`'s description of competitive intensity differ across documents, and what should an analyst infer from that?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** risk factors + transcript + letter if available
**Gold Answer Outline:** identify variation in competition framing; infer cautiously
**Evaluation Focus:** multi-source competition analysis
**Common Failure Modes:** reading normal document-purpose differences as contradiction

### EC-009 | Evidence Conflict | claim_reconciliation | Hard
**Query:** If management highlights a strategic initiative as important but gives little measurable evidence of traction, how should that claim be labeled for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, letter, KPI disclosures
**Gold Answer Outline:** likely partial support or insufficient evidence unless concrete proof exists
**Evaluation Focus:** verifier discipline
**Common Failure Modes:** full support based on rhetoric alone

### EC-010 | Evidence Conflict | conflict_detection | Hard
**Query:** Are there examples where the latest quarter narrative for `{{COMPANY_A}}` appears to conflict with earlier management claims?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** prior and latest period materials
**Gold Answer Outline:** identify true conflict or explain why not a real conflict
**Evaluation Focus:** temporal contradiction detection
**Common Failure Modes:** conflating changed conditions with contradiction

### EC-011 | Evidence Conflict | discrepancy_analysis | Hard
**Query:** Does the source pack provide more than one plausible interpretation of `{{COMPANY_A}}`'s recent margin trend?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** quarterly report + transcript + letter
**Gold Answer Outline:** present multiple interpretations if evidence supports them
**Evaluation Focus:** ambiguity-aware analysis
**Common Failure Modes:** over-forcing a single narrative

### EC-012 | Evidence Conflict | claim_reconciliation | Hard
**Query:** When primary and secondary sources differ in emphasis on `{{COMPANY_A}}`, which should dominate the final answer and why?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** primary + secondary sources
**Gold Answer Outline:** primary sources should dominate; secondary may contextualize
**Evaluation Focus:** source governance reasoning
**Common Failure Modes:** equal-weighting all sources

### EC-013 | Evidence Conflict | conflict_detection | Hard
**Query:** Is there tension between what `{{COMPANY_A}}` says is strategically important and what its financial disclosures suggest is economically important?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** strategy commentary + financial disclosures
**Gold Answer Outline:** strategy-vs-economics tension if present
**Evaluation Focus:** cross-layer reconciliation
**Common Failure Modes:** no separation of strategic and financial importance

### EC-014 | Evidence Conflict | discrepancy_analysis | Medium
**Query:** Do different source types describe `{{COMPANY_A}}` more as a platform, an infrastructure provider, or a software company? How should that be interpreted?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** annual report, letter, transcript
**Gold Answer Outline:** taxonomy differences may reflect audience/purpose; reconcile carefully
**Evaluation Focus:** semantic framing analysis
**Common Failure Modes:** assuming the labels must be mutually exclusive

### EC-015 | Evidence Conflict | claim_reconciliation | Hard
**Query:** If evidence on `{{COMPANY_A}}`'s enterprise traction is mixed, what is the most defensible wording for a research memo?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript, customer commentary, KPI disclosures
**Gold Answer Outline:** calibrated wording with explicit uncertainty
**Evaluation Focus:** cautious memo-writing under mixed evidence
**Common Failure Modes:** overstated traction claim

### EC-016 | Evidence Conflict | verifier_stress_test | Hard
**Query:** Which claims about `{{COMPANY_A}}` are likely to be only partially supported rather than fully supported by the available evidence?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** broad pack
**Gold Answer Outline:** identify claim types prone to partial support: causality, future potential, strategic importance, cross-sell depth
**Evaluation Focus:** partial-support classification
**Common Failure Modes:** binary supported/unsupported thinking

### EC-017 | Evidence Conflict | conflict_detection | Medium
**Query:** Does management's narrative about demand strength conflict with disclosed risk factors about customer spending or budget pressure at `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript + risk factors
**Gold Answer Outline:** determine whether narrative tension exists; explain if standard disclosure caveat only
**Evaluation Focus:** real conflict vs boilerplate distinction
**Common Failure Modes:** treating standard risk language as contradiction automatically

### EC-018 | Evidence Conflict | claim_reconciliation | Hard
**Query:** If one source suggests `{{COMPANY_A}}` is broadening its platform while another suggests tighter prioritization, what is the most likely reconciliation?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** transcript + letter or annual report
**Gold Answer Outline:** broadening at product level but focusing execution at go-to-market or resource level, or similar reconciliation if supported
**Evaluation Focus:** multi-dimensional strategy reconciliation
**Common Failure Modes:** false contradiction

### EC-019 | Evidence Conflict | ambiguity_analysis | Hard
**Query:** What is the most important ambiguity in the current source pack for `{{COMPANY_A}}` that an analyst should not overstate as fact?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** broad pack
**Gold Answer Outline:** identify one ambiguous topic and explain why evidence is insufficient or mixed
**Evaluation Focus:** epistemic humility
**Common Failure Modes:** no ambiguity acknowledged

### EC-020 | Evidence Conflict | source_priority | Medium
**Query:** If documents differ on how prominently they discuss a strategic initiative at `{{COMPANY_A}}`, which source should be prioritized for deciding whether it is material?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** multiple source types
**Gold Answer Outline:** prioritize financially material / primary disclosures; use commentary to enrich
**Evaluation Focus:** materiality-aware source ranking
**Common Failure Modes:** prioritizing the most enthusiastic source

### EC-021 | Evidence Conflict | discrepancy_analysis | Hard
**Query:** Are there places where numeric evidence and narrative evidence point in slightly different directions for `{{COMPANY_A}}`?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** KPI or financial disclosures + management narrative
**Gold Answer Outline:** identify numeric-vs-narrative mismatch if present
**Evaluation Focus:** quantitative-narrative reconciliation
**Common Failure Modes:** ignoring one side of the mismatch

### EC-022 | Evidence Conflict | verifier_stress_test | Hard
**Query:** Which likely analyst-style claims about `{{COMPANY_A}}` should be labeled as “insufficient evidence” rather than “supported” based on the current source pack?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** broad pack
**Gold Answer Outline:** list claim types with weak direct support
**Evaluation Focus:** insufficiency detection
**Common Failure Modes:** overlabeling strategic narratives as supported facts

### EC-023 | Evidence Conflict | claim_reconciliation | Hard
**Query:** What is the most balanced way to write about `{{COMPANY_A}}` when the source pack contains both clear strengths and unresolved contradictions?
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** broad pack
**Gold Answer Outline:** balanced paragraph structure with strengths, caveats, and unresolved questions
**Evaluation Focus:** high-quality ambiguity-aware synthesis
**Common Failure Modes:** one-sided conclusion

### EC-024 | Evidence Conflict | multi_source_reasoning | Hard
**Query:** Summarize the three most important places where an analyst must reconcile different pieces of evidence before making a confident claim about `{{COMPANY_A}}`.
**Primary Companies:** `{{COMPANY_A}}`
**Required Source Types:** broad pack across multiple source types
**Gold Answer Outline:** 3 reconciliation points with explanation
**Evaluation Focus:** full-stack conflict-aware reasoning
**Common Failure Modes:** generic uncertainty statement without concrete tensions

---

## 14. Suggested Instantiation Variables

Replace the placeholders before running the benchmark.

- `{{COMPANY_A}}`: primary company under analysis
- `{{COMPANY_B}}`: peer company for comparisons

Optional extended placeholders for a richer benchmark runner:

- `{{SECTOR}}`
- `{{LATEST_PERIOD}}`
- `{{PRIOR_PERIOD}}`
- `{{PRIMARY_SOURCE}}`
- `{{SECONDARY_SOURCE}}`

---

## 15. Suggested Company Pairings

If you want to instantiate this benchmark quickly, use one of these peer sets.

### Payments / Commerce Infrastructure
- Stripe vs Adyen
- Block vs PayPal
- Shopify vs Toast

### SaaS / Cloud
- Snowflake vs Datadog
- HubSpot vs Salesforce
- MongoDB vs Elastic

### Semiconductors
- TSMC vs Samsung Electronics
- NVIDIA vs AMD
- SMIC vs Hua Hong

---

## 16. Aggregation Template

Use the following reporting summary after a benchmark run.

```yaml
run_id: benchmark_v1_run_001
model_stack:
  planner: <name>
  retriever_dense: <name>
  reranker: <name>
  generator: <name>
  verifier: <name>
summary:
  total_items: 120
  pass_rate: 0.00
  avg_retrieval_score: 0.00
  avg_answer_score: 0.00
  avg_verification_score: 0.00
category_breakdown:
  company_overview:
    pass_rate: 0.00
  risk_catalyst:
    pass_rate: 0.00
  comparison:
    pass_rate: 0.00
  time_sensitive:
    pass_rate: 0.00
  evidence_conflict:
    pass_rate: 0.00
failure_breakdown:
  R1_missing_primary_source: 0
  R2_wrong_period: 0
  R3_wrong_peer: 0
  R4_keyword_miss: 0
  R5_semantic_miss: 0
  R6_chunk_boundary_loss: 0
  R7_table_miss: 0
  S1_incomplete_answer: 0
  S2_vague_answer: 0
  S3_wrong_comparison_dimension: 0
  S4_temporal_confusion: 0
  S5_overclaim: 0
  S6_missing_uncertainty: 0
  V1_related_not_supporting: 0
  V2_partial_as_full_support: 0
  V3_number_or_time_mismatch: 0
  V4_conflict_not_detected: 0
  V5_insufficient_evidence_not_flagged: 0
```

---

## 17. Recommended Next Step After v1

After the first run, do **not** immediately expand the dataset.

Instead:

1. label 30–50 items fully,
2. run ablations,
3. inspect dominant failure modes,
4. only then refine question wording or add new items.

A useful sequence is:

- v1: this benchmark markdown
- v1.1: fully labeled gold evidence on 30 items
- v1.2: full labels on all 120 items
- v2: sector-specific benchmark extension

---

## 18. Minimal Acceptance Criteria for 证据驱动可验证的企业财务研究Agent Flow

Before claiming strong benchmark performance, the system should ideally achieve:

- Retrieval average >= 3.0 / 4.0
- Answer average >= 3.0 / 4.0
- Verification average >= 2.8 / 4.0
- Pass rate >= 70%
- No category pass rate below 55%
- Evidence Conflict not more than 15 points below Company Overview in pass rate

These are not universal standards; they are practical internal thresholds for a strong prototype.

---

## 19. Closing Note

This benchmark is intentionally built to reveal where a business-research agent fails:

- not just whether it can answer,
- but whether it can answer **with the right evidence**,
- compare firms **without hand-waving**,
- update views **over time**, and
- handle **ambiguity and conflict** without pretending certainty.

That is what separates a polished demo from a system that can actually support research work.
