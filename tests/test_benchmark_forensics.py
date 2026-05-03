import csv

from eval.benchmark_forensics import (
    cohort_summary,
    compare_payloads,
    ledger_rows,
    render_forensics_report,
    write_ledger_csv,
)


def _payload(rows):
    return {"run_id": "run-a", "retrieval_backend": "hybrid", "rows": rows}


def test_cohort_summary_groups_metrics_by_trust_family():
    payload = _payload(
        [
            {
                "item_id": "a",
                "query": "What was revenue?",
                "item_companies": ["AMD"],
                "answer_status": "answered",
                "unsupported_claim_rate": 0.0,
                "unsupported_numeric_claim_rate": 0.0,
                "required_fact_recall": 1.0,
                "failure_tags": [],
            },
            {
                "item_id": "b",
                "query": "What was margin?",
                "item_companies": ["AMD"],
                "answer_status": "partial",
                "unsupported_claim_rate": 0.2,
                "required_fact_recall": 0.0,
                "failure_tags": ["required_fact_missing"],
            },
        ]
    )

    summary = cohort_summary(payload)

    assert summary["row_count"] == 2
    assert summary["status_counts"] == {"answered": 1, "partial": 1}
    assert summary["failure_tag_counts"] == {"required_fact_missing": 1}
    assert summary["metric_averages"]["unsupported_claim_rate"] == 0.1
    assert "grounding_quality" in summary["metric_families"]
    assert "retrieval_quality" in summary["metric_families"]


def test_render_forensics_report_includes_standards_mapping_and_comparison():
    baseline = _payload(
        [
            {
                "item_id": "a",
                "answer_status": "answered",
                "unsupported_claim_rate": 0.0,
                "required_fact_recall": 0.5,
            }
        ]
    )
    candidate = _payload(
        [
            {
                "item_id": "a",
                "answer_status": "answered",
                "unsupported_claim_rate": 0.1,
                "required_fact_recall": 1.0,
                "failure_tags": ["unsupported_claim"],
            }
        ]
    )

    report = render_forensics_report(candidate, baseline=baseline)
    comparison = compare_payloads(baseline, candidate)

    assert "Standards Mapping" in report
    assert "unsupported_claim_rate: +0.1000" in report
    assert comparison["matched_items"] == 1
    assert comparison["safety_regressions"] == {"unsupported_claim_rate": 0.1}


def test_write_ledger_csv_exports_duckdb_compatible_rows(tmp_path):
    payload = _payload(
        [
            {
                "item_id": "a",
                "query_type": ["hard_fact"],
                "item_companies": ["AMD"],
                "answer_status": "answered",
                "unsupported_claim_rate": 0.0,
                "failure_tags": [],
                "trace_path": "artifacts/a/trace.json",
            }
        ]
    )
    output = tmp_path / "ledger.csv"

    write_ledger_csv(payload, output)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))

    assert ledger_rows(payload)[0]["item_id"] == "a"
    assert rows[0]["item_id"] == "a"
    assert rows[0]["company_id"] == "AMD"
    assert "unsupported_claim_rate" in rows[0]
