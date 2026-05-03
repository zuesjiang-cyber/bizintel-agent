from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from eval.trust_standards import mapping_for_metric, metric_family, metric_mappings


KEY_METRICS = (
    "unsupported_claim_rate",
    "unsupported_numeric_claim_rate",
    "fabricated_citation_rate",
    "wrong_entity_rate",
    "wrong_period_rate",
    "numeric_exact_match_rate",
    "primary_source_claim_coverage",
    "required_fact_recall",
    "required_slot_coverage",
    "retrieval_hit",
    "hard_fact_complete_but_not_published_rate",
    "answer_quality",
    "gold_benchmark_pass",
)

SAFETY_REGRESSION_METRICS = (
    "unsupported_claim_rate",
    "unsupported_numeric_claim_rate",
    "fabricated_citation_rate",
    "wrong_entity_rate",
    "wrong_period_rate",
)

LEDGER_FIELDS = (
    "run_id",
    "item_id",
    "company_id",
    "question_type",
    "answer_status",
    "retrieval_backend",
    "retrieval_path",
    "failure_tags",
    "artifact_paths",
    *KEY_METRICS,
)


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def payload_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(payload.get("items"), list):
        return [row for row in payload["items"] if isinstance(row, dict)]
    return []


def row_item_id(row: dict[str, Any], index: int = 0) -> str:
    return str(
        row.get("item_id")
        or row.get("id")
        or row.get("question_id")
        or row.get("financebench_id")
        or f"row_{index:04d}"
    )


def row_company(row: dict[str, Any]) -> str:
    companies = row.get("item_companies") or row.get("companies") or row.get("company")
    if isinstance(companies, list):
        return str(companies[0]) if companies else ""
    return str(companies or row.get("company_id") or "")


def row_question_type(row: dict[str, Any]) -> str:
    value = row.get("question_type") or row.get("query_type") or row.get("category")
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value or "")


def row_answer_status(row: dict[str, Any]) -> str:
    explicit = row.get("answer_status") or row.get("status")
    if explicit:
        return str(explicit)
    if row.get("abstained") or row.get("refused"):
        return "abstained"
    if _as_float(row.get("answer_quality")) == 0:
        return "partial"
    if row.get("memo_markdown") or row.get("answer"):
        return "answered"
    return "unknown"


def row_failure_tags(row: dict[str, Any]) -> list[str]:
    tags = [str(item) for item in _as_list(row.get("failure_tags")) if item]
    question_errors = row.get("question_error_summary") or {}
    if isinstance(question_errors, dict):
        severe = question_errors.get("most_severe_failure_type")
        if severe:
            tags.append(str(severe))
        for reason in _as_list(question_errors.get("controller_failure_reasons")):
            if reason:
                tags.append(str(reason))
    return sorted(set(tags))


def row_artifact_paths(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("memo_path", "trace_path", "summary_path", "verification_path", "artifact_dir")
        if row.get(key)
    }


def normalize_row(row: dict[str, Any], payload: dict[str, Any] | None = None, index: int = 0) -> dict[str, Any]:
    payload = payload or {}
    metrics = {
        metric: _as_float(row.get(metric))
        for metric in KEY_METRICS
        if _as_float(row.get(metric)) is not None
    }
    backend = row.get("retrieval_backend") or payload.get("retrieval_backend") or "unknown"
    return {
        "run_id": payload.get("run_id") or payload.get("created_at") or payload.get("version") or "",
        "item_id": row_item_id(row, index),
        "company_id": row_company(row),
        "question": row.get("query") or row.get("question") or "",
        "question_type": row_question_type(row),
        "answer_status": row_answer_status(row),
        "retrieval_backend": backend,
        "retrieval_path": row.get("retrieval_path") or row.get("retrieval_path_counts") or "",
        "metrics": metrics,
        "failure_tags": row_failure_tags(row),
        "claim_results": row.get("claim_results", []),
        "retrieval_candidates": row.get("retrieval_candidates", []),
        "artifact_paths": row_artifact_paths(row),
        "raw": row,
    }


def normalize_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [normalize_row(row, payload, index) for index, row in enumerate(payload_rows(payload))]


def row_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["item_id"]: row for row in normalize_rows(payload) if row.get("item_id")}


def average_metric(rows: Iterable[dict[str, Any]], metric: str) -> float | None:
    values = [
        row["metrics"][metric]
        for row in rows
        if metric in row.get("metrics", {}) and row["metrics"][metric] is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def metric_averages(rows: list[dict[str, Any]]) -> dict[str, float]:
    averages: dict[str, float] = {}
    for metric in KEY_METRICS:
        avg = average_metric(rows, metric)
        if avg is not None:
            averages[metric] = avg
    return averages


def cohort_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = normalize_rows(payload)
    status_counts = Counter(row["answer_status"] for row in rows)
    failure_counts = Counter(tag for row in rows for tag in row["failure_tags"])
    averages = metric_averages(rows)
    families: dict[str, dict[str, float]] = {}
    for metric, value in averages.items():
        families.setdefault(metric_family(metric), {})[metric] = value
    return {
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "failure_tag_counts": dict(failure_counts),
        "metric_averages": averages,
        "metric_families": families,
        "standards": metric_mappings(),
    }


def representative_cases(rows: list[dict[str, Any]], limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "item_id": row["item_id"],
            "company_id": row["company_id"],
            "answer_status": row["answer_status"],
            "failure_tags": row["failure_tags"],
            "required_fact_recall": row["metrics"].get("required_fact_recall"),
            "unsupported_claim_rate": row["metrics"].get("unsupported_claim_rate"),
            "question": row["question"],
        }

    successes = [
        row
        for row in rows
        if not row["failure_tags"]
        and (row["metrics"].get("unsupported_claim_rate", 0.0) or 0.0) == 0.0
        and row["answer_status"] in {"answered", "partial", "unknown"}
    ]
    failures = [row for row in rows if row["failure_tags"] or row["answer_status"] == "abstained"]
    successes.sort(key=lambda row: (row["metrics"].get("required_fact_recall") or 0.0), reverse=True)
    failures.sort(
        key=lambda row: (
            len(row["failure_tags"]),
            row["metrics"].get("unsupported_claim_rate") or 0.0,
            row["metrics"].get("hard_fact_complete_but_not_published_rate") or 0.0,
        ),
        reverse=True,
    )
    return {
        "successes": [compact(row) for row in successes[:limit]],
        "failures": [compact(row) for row in failures[:limit]],
    }


def compare_payloads(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = row_map(baseline)
    candidate_rows = row_map(candidate)
    common_ids = sorted(set(baseline_rows) & set(candidate_rows))
    deltas: dict[str, float] = {}
    for metric in KEY_METRICS:
        values = []
        for item_id in common_ids:
            base_value = baseline_rows[item_id]["metrics"].get(metric)
            cand_value = candidate_rows[item_id]["metrics"].get(metric)
            if base_value is None or cand_value is None:
                continue
            values.append(cand_value - base_value)
        if values:
            deltas[metric] = sum(values) / len(values)

    item_diffs = []
    for item_id in common_ids:
        base = baseline_rows[item_id]
        cand = candidate_rows[item_id]
        safety_regression = sum(
            max(0.0, (cand["metrics"].get(metric) or 0.0) - (base["metrics"].get(metric) or 0.0))
            for metric in SAFETY_REGRESSION_METRICS
        )
        completeness_delta = (cand["metrics"].get("required_fact_recall") or 0.0) - (
            base["metrics"].get("required_fact_recall") or 0.0
        )
        item_diffs.append(
            {
                "item_id": item_id,
                "company_id": cand["company_id"] or base["company_id"],
                "baseline_status": base["answer_status"],
                "candidate_status": cand["answer_status"],
                "safety_regression_score": safety_regression,
                "required_fact_recall_delta": completeness_delta,
                "baseline_failure_tags": base["failure_tags"],
                "candidate_failure_tags": cand["failure_tags"],
            }
        )
    item_diffs.sort(key=lambda row: (row["safety_regression_score"], -row["required_fact_recall_delta"]), reverse=True)
    return {
        "matched_items": len(common_ids),
        "metric_deltas": deltas,
        "safety_regressions": {metric: delta for metric, delta in deltas.items() if metric in SAFETY_REGRESSION_METRICS and delta > 0},
        "largest_item_regressions": item_diffs[:5],
        "largest_item_improvements": sorted(item_diffs, key=lambda row: row["required_fact_recall_delta"], reverse=True)[:5],
    }


def render_forensics_report(
    candidate: dict[str, Any],
    candidate_path: Path | None = None,
    baseline: dict[str, Any] | None = None,
    baseline_path: Path | None = None,
) -> str:
    rows = normalize_rows(candidate)
    summary = cohort_summary(candidate)
    cases = representative_cases(rows)
    lines = [
        "# Benchmark Forensics Report",
        "",
        f"- Candidate: `{candidate_path or 'in-memory'}`",
        f"- Rows: {summary['row_count']}",
        "",
        "## Status Distribution",
        "",
    ]
    for status, count in sorted(summary["status_counts"].items()):
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Trust Metrics", ""])
    for metric in KEY_METRICS:
        if metric not in summary["metric_averages"]:
            continue
        mapping = mapping_for_metric(metric)
        family = mapping.metric_family if mapping else "uncategorized"
        gate = mapping.fintrust_gate if mapping else "unmapped"
        lines.append(f"- {metric}: {summary['metric_averages'][metric]:.4f} ({family}; {gate})")

    lines.extend(["", "## Failure Tags", ""])
    if summary["failure_tag_counts"]:
        for tag, count in sorted(summary["failure_tag_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Representative Successes", ""])
    for row in cases["successes"]:
        lines.append(f"- {row['item_id']} | {row['company_id']} | recall={row['required_fact_recall']} | {row['question']}")
    if not cases["successes"]:
        lines.append("- none")

    lines.extend(["", "## Representative Failures", ""])
    for row in cases["failures"]:
        tags = ", ".join(row["failure_tags"]) or "none"
        lines.append(f"- {row['item_id']} | {row['company_id']} | tags={tags} | {row['question']}")
    if not cases["failures"]:
        lines.append("- none")

    if baseline is not None:
        comparison = compare_payloads(baseline, candidate)
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                f"- Baseline: `{baseline_path or 'in-memory'}`",
                f"- Matched items: {comparison['matched_items']}",
                "",
                "### Metric Deltas",
                "",
            ]
        )
        for metric in KEY_METRICS:
            if metric in comparison["metric_deltas"]:
                lines.append(f"- {metric}: {comparison['metric_deltas'][metric]:+.4f}")
        lines.extend(["", "### Safety Regressions", ""])
        if comparison["safety_regressions"]:
            for metric, delta in comparison["safety_regressions"].items():
                lines.append(f"- {metric}: {delta:+.4f}")
        else:
            lines.append("- none")

    lines.extend(["", "## Standards Mapping", ""])
    lines.append("| Metric | Family | Gate | Judge Type | Failure Action |")
    lines.append("|---|---|---|---|---|")
    for mapping in metric_mappings():
        lines.append(
            f"| {mapping['metric_name']} | {mapping['metric_family']} | "
            f"{mapping['fintrust_gate']} | {mapping['judge_type']} | {mapping['failure_action']} |"
        )
    return "\n".join(lines) + "\n"


def ledger_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in normalize_rows(payload):
        ledger = {
            "run_id": row["run_id"],
            "item_id": row["item_id"],
            "company_id": row["company_id"],
            "question_type": row["question_type"],
            "answer_status": row["answer_status"],
            "retrieval_backend": row["retrieval_backend"],
            "retrieval_path": _jsonish(row["retrieval_path"]),
            "failure_tags": _jsonish(row["failure_tags"]),
            "artifact_paths": _jsonish(row["artifact_paths"]),
        }
        for metric in KEY_METRICS:
            ledger[metric] = row["metrics"].get(metric, "")
        output.append(ledger)
    return output


def write_ledger_csv(payload: dict[str, Any], output_path: Path) -> None:
    rows = ledger_rows(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
