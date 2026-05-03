# Benchmark Forensics Report

- Candidate: `eval/results/financebench_open150_all_live_20260410_parallel_merged.json`
- Rows: 150

## Status Distribution

- abstained: 11
- answered: 39
- partial: 100

## Trust Metrics

- unsupported_claim_rate: 0.0419 (grounding_quality; Grounding Gate)
- unsupported_numeric_claim_rate: 0.0435 (finance_hard_gate; Finance Hard Gate)
- fabricated_citation_rate: 0.0000 (citation_quality; Citation Validity Gate)
- wrong_entity_rate: 0.0000 (finance_hard_gate; Entity Gate)
- wrong_period_rate: 0.0000 (finance_hard_gate; Period Gate)
- numeric_exact_match_rate: 0.9197 (finance_hard_gate; Finance Hard Gate)
- primary_source_claim_coverage: 0.8809 (source_quality; Primary Source Gate)
- required_fact_recall: 0.1944 (retrieval_quality; Retrieval Gate)
- required_slot_coverage: 0.3611 (retrieval_quality; Retrieval Gate)
- retrieval_hit: 1.0000 (retrieval_quality; Retrieval Gate)
- hard_fact_complete_but_not_published_rate: 0.5800 (publication_gate; Publication Gate)
- answer_quality: 3.4267 (presentation_quality; Presentation Gate)
- gold_benchmark_pass: 0.1533 (benchmark_quality; Benchmark Gate)

## Failure Tags

- G1_gold_semantic_gap: 128
- S1_incomplete_answer: 120
- G1_gold_numeric_miss: 109
- S1_missing_required_slot: 95
- R1_missing_primary_source: 18
- G1_gold_citation_miss: 11
- primary_source_missing: 10
- Best support 0.74 below hard-fact threshold.: 9
- S1_question_tree_incomplete: 7
- S5_untrusted_answer: 7
- Best support 0.73 below hard-fact threshold.: 6
- Best support 0.78 below hard-fact threshold.: 6
- Best support 0.49 below hard-fact threshold.: 5
- Best support 0.57 below hard-fact threshold.: 3
- Best support 0.61 below hard-fact threshold.: 2
- Best support 0.66 below hard-fact threshold.: 2
- Conflicting evidence detected across top supporting chunks.: 2
- Generated answer did not survive verification.: 2
- Mean support 0.43 below semantic threshold.: 2
- Mean support 0.46 below semantic threshold.: 2
- Best support 0.69 below hard-fact threshold.: 1
- Mean support 0.44 below semantic threshold.: 1

## Representative Successes

- financebench_id_00563 | financebench_amd | recall=1.0 | From FY21 to FY22, excluding Embedded, in which AMD reporting segment did sales proportionally increase the most?
- financebench_id_00705 | financebench_pepsico | recall=1.0 | By how much did Pepsico increase its unsecured five year revolving credit agreement on May 26, 2023?
- financebench_id_00720 | financebench_american_express | recall=1.0 | What drove gross margin change as of the FY2022 for American Express? If gross margin is not a useful metric for a company like this, then please state that and explain why.

## Representative Failures

- financebench_id_07507 | financebench_adobe | tags=Best support 0.49 below hard-fact threshold., G1_gold_citation_miss, G1_gold_numeric_miss, G1_gold_semantic_gap, Mean support 0.43 below semantic threshold., Mean support 0.44 below semantic threshold., R1_missing_primary_source, S1_incomplete_answer, S1_missing_required_slot, S1_question_tree_incomplete | What is Adobe's year-over-year change in unadjusted operating income from FY2015 to FY2016 (in units of percents and round to one decimal place)? Give a solution to the question by using the income statement.
- financebench_id_00591 | financebench_adobe | tags=Best support 0.49 below hard-fact threshold., G1_gold_citation_miss, G1_gold_numeric_miss, G1_gold_semantic_gap, Mean support 0.43 below semantic threshold., R1_missing_primary_source, S1_incomplete_answer, S1_missing_required_slot, S1_question_tree_incomplete | Does Adobe have an improving Free cashflow conversion as of FY2022?
- financebench_id_01474 | financebench_pepsico | tags=Best support 0.57 below hard-fact threshold., Best support 0.61 below hard-fact threshold., G1_gold_citation_miss, G1_gold_semantic_gap, Mean support 0.46 below semantic threshold., R1_missing_primary_source, S1_incomplete_answer, S1_missing_required_slot, S1_question_tree_incomplete | As of FY2023Q1, why did Pepsico raise full year guidance for FY2023?

## Baseline Comparison

- Baseline: `eval/results/financebench_open150_all_live_20260410_master_v2_retry.json`
- Matched items: 12

### Metric Deltas

- unsupported_claim_rate: -0.1263
- unsupported_numeric_claim_rate: -0.1263
- fabricated_citation_rate: -0.0278
- wrong_entity_rate: +0.0000
- wrong_period_rate: +0.0000
- numeric_exact_match_rate: +0.2424
- primary_source_claim_coverage: +0.0985
- required_fact_recall: +0.0556
- required_slot_coverage: +0.0000
- retrieval_hit: +0.0000
- hard_fact_complete_but_not_published_rate: +0.0000
- answer_quality: +0.4167
- gold_benchmark_pass: -0.0833

### Safety Regressions

- none

## Standards Mapping

| Metric | Family | Gate | Judge Type | Failure Action |
|---|---|---|---|---|
| unsupported_claim_rate | grounding_quality | Grounding Gate | hybrid | delete_or_downgrade |
| unsupported_numeric_claim_rate | finance_hard_gate | Finance Hard Gate | deterministic | delete |
| fabricated_citation_rate | citation_quality | Citation Validity Gate | metadata_check | delete |
| primary_source_claim_coverage | source_quality | Primary Source Gate | metadata_check | downgrade_or_refuse |
| numeric_exact_match_rate | finance_hard_gate | Finance Hard Gate | deterministic | delete_or_mark_miss |
| required_fact_recall | retrieval_quality | Retrieval Gate | hybrid | needs_followup_or_mark_gap |
| required_slot_coverage | retrieval_quality | Retrieval Gate | hybrid | needs_followup_or_mark_gap |
| retrieval_hit | retrieval_quality | Retrieval Gate | metadata_check | needs_followup_or_refuse |
| hard_fact_complete_but_not_published_rate | publication_gate | Publication Gate | metadata_check | recover_or_report_publication_gap |
| wrong_entity_rate | finance_hard_gate | Entity Gate | deterministic | delete_or_refuse |
| wrong_period_rate | finance_hard_gate | Period Gate | deterministic | delete_or_refuse |
| gold_benchmark_pass | benchmark_quality | Benchmark Gate | hybrid | report_regression |
| answer_quality | presentation_quality | Presentation Gate | llm_judge | warn |
