import importlib.util
import json
from pathlib import Path
import sys

import httpx
from openai import APIStatusError

from agent.config import settings


RUNNER_PATH = Path(__file__).resolve().parents[1] / "eval" / "benchmark_runner.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SPEC = importlib.util.spec_from_file_location("bizintel_eval_benchmark_runner", RUNNER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

answer_quality = MODULE.answer_quality
assess_target_metrics = MODULE.assess_target_metrics
benchmark_index_key = MODULE.benchmark_index_key
benchmark_index_profile = MODULE.benchmark_index_profile
build_payload = MODULE.build_payload
compute_averages = MODULE.compute_averages
failure_tags = MODULE.failure_tags
is_retryable_live_error = MODULE.is_retryable_live_error
load_existing_rows = MODULE.load_existing_rows
load_evidence_store_for_companies = MODULE.load_evidence_store_for_companies
resolve_profile_item_ids = MODULE.resolve_profile_item_ids
requested_item_ids = MODULE.requested_item_ids
retrieval_hit = MODULE.retrieval_hit
selection_item_ids = MODULE.selection_item_ids
validate_profile_items = MODULE.validate_profile_items


def test_answer_quality_scales_with_metrics():
    quality = answer_quality(
        {"required_evidence": {"min_distinct_sources": 2}},
        {
            "verified_claim_coverage": 0.8,
            "unsupported_claim_rate": 0.1,
            "required_fact_recall": 0.8,
        },
        2,
    )

    assert quality == 5


def test_answer_quality_caps_when_question_tree_is_incomplete():
    quality = answer_quality(
        {"required_evidence": {"min_distinct_sources": 1}},
        {
            "verified_claim_coverage": 0.9,
            "unsupported_claim_rate": 0.0,
            "required_fact_recall": 1.0,
            "required_subquestion_coverage": 0.4,
            "required_slot_coverage": 0.4,
            "decision_replay_consistency": 1.0,
        },
        1,
    )

    assert quality == 3


def test_failure_tags_include_unsupported_scope_when_controller_refuses_comparison():
    tags = failure_tags(
        {
            "category": "comparison",
            "required_evidence": {"min_distinct_sources": 1},
        },
        {
            "verified_claim_coverage": 0.0,
            "unsupported_claim_rate": 1.0,
            "required_fact_recall": 0.0,
            "scope_supported": False,
        },
        "",
        [],
        [],
        0,
    )

    assert "A1_unsupported_scope_multi_company" in tags


def test_retrieval_hit_detects_cited_gold_doc():
    memo = "This is supported [Source: cloudflare_q4_2025_results]."
    gold_entries = [{"doc_id": "cloudflare_q4_2025_results"}]

    assert retrieval_hit(gold_entries, memo, [], 1) == 1


def test_retrieval_hit_detects_source_ids_from_memo_object():
    gold_entries = [{"doc_id": "cloudflare_q4_2025_results"}]

    assert retrieval_hit(gold_entries, "no citation here", ["cloudflare_q4_2025_results"], 1) == 1


def test_benchmark_index_key_sorts_company_names():
    assert benchmark_index_key(["fastly", "cloudflare"]) == "cloudflare__fastly"


def test_benchmark_index_profile_separates_real_and_dummy_indexes():
    real_profile = benchmark_index_profile(load_models=True)
    dummy_profile = benchmark_index_profile(load_models=False)

    assert real_profile != dummy_profile
    assert real_profile.startswith("real__")
    assert dummy_profile.startswith("dummy__")


def test_requested_item_ids_supports_all_split():
    splits = {
        "dev": ["BO-001", "RC-001"],
        "test": ["BO-002"],
    }

    assert requested_item_ids(splits, "all") == ["BO-001", "RC-001", "BO-002"]


def test_build_payload_includes_per_split_summary_for_all():
    rows = [
        {
            "item_id": "BO-001",
            "split": "dev",
            "category": "company_overview",
            "difficulty": "medium",
            "query_type": ["structured_summary"],
            "retrieval_hit": 1,
            "citation_marker_coverage": 1.0,
            "verified_claim_coverage": 0.5,
            "unsupported_claim_rate": 0.2,
            "required_fact_recall": 0.4,
            "avg_nli_score": 0.7,
            "answer_quality": 3,
        },
        {
            "item_id": "BO-002",
            "split": "test",
            "category": "company_overview",
            "difficulty": "medium",
            "query_type": ["structured_summary"],
            "retrieval_hit": 0,
            "citation_marker_coverage": 0.5,
            "verified_claim_coverage": 0.3,
            "unsupported_claim_rate": 0.6,
            "required_fact_recall": 0.2,
            "avg_nli_score": 0.4,
            "answer_quality": 2,
        },
    ]

    payload = build_payload("v2", "all", "live_llm_real_retrieval_real_nli", ["cloudflare"], rows)

    assert payload["averages"] == compute_averages(rows)
    assert payload["per_split"]["dev"]["count"] == 1
    assert payload["per_split"]["dev"]["averages"] == compute_averages([rows[0]])
    assert payload["per_split"]["test"]["count"] == 1
    assert payload["per_split"]["test"]["averages"] == compute_averages([rows[1]])
    assert "required_subquestion_coverage" in payload["averages"]
    assert "decision_replay_consistency" in payload["averages"]


def test_build_payload_includes_profile_target_assessment():
    rows = [
        {
            "item_id": "NUM-001",
            "split": "dev",
            "category": "company_overview",
            "difficulty": "medium",
            "query_type": ["numeric_grounding"],
            "retrieval_hit": 1,
            "citation_marker_coverage": 1.0,
            "verified_claim_coverage": 0.9,
            "unsupported_claim_rate": 0.1,
            "required_fact_recall": 1.0,
            "avg_nli_score": 0.8,
            "answer_quality": 5,
        }
    ]

    payload = build_payload(
        "v2",
        "dev",
        "live_llm_real_retrieval_real_nli",
        ["fastly"],
        rows,
        profile_name="trust_showcase_v1",
        profile={
            "track": "showcase",
            "purpose": "demo",
            "target_metrics": {
                "verified_claim_coverage": {"target": 0.828, "direction": "at_least"},
                "unsupported_claim_rate": {"target": 0.1225, "direction": "at_most"},
            },
            "question_constraints": {"allowed_difficulties": ["medium"]},
        },
    )

    assert payload["profile"] == "trust_showcase_v1"
    assert payload["target_assessment"]["all_met"] is True
    assert payload["portfolio_summary"]["item_count"] == 1


def test_resolve_profile_item_ids_uses_split_when_needed():
    profile = {"split": "dev"}
    splits = {"dev": ["NUM-001"], "test": ["BO-002"]}

    assert resolve_profile_item_ids(profile, splits) == ["NUM-001"]


def test_selection_item_ids_returns_profile():
    splits = {"dev": ["NUM-001"], "test": ["BO-002"]}
    profiles = {"trust_showcase_v1": {"item_ids": ["BO-002"], "track": "showcase"}}

    item_ids, profile = selection_item_ids(splits, "dev", profiles, "trust_showcase_v1")

    assert item_ids == ["BO-002"]
    assert profile["track"] == "showcase"


def test_validate_profile_items_flags_items_outside_constraints():
    items = {
        "RC-001": {
            "category": "risk_catalyst",
            "difficulty": "hard",
            "query_type": ["risk_extraction"],
            "target_periods": ["2025Q4"],
            "required_evidence": {"must_cover": ["a", "b", "c", "d"], "min_distinct_sources": 3},
        }
    }
    profile = {
        "item_ids": ["RC-001"],
        "question_constraints": {
            "allowed_categories": ["company_overview"],
            "allowed_difficulties": ["medium"],
            "allowed_query_types": ["numeric_grounding"],
            "max_must_cover": 3,
            "max_distinct_sources": 2,
            "require_target_periods": True,
        },
    }

    violations = validate_profile_items("trust_showcase_v1", profile, items, {"dev": [], "test": []})

    assert "RC-001" in violations
    assert "category:risk_catalyst" in violations["RC-001"]
    assert "difficulty:hard" in violations["RC-001"]
    assert "must_cover>3" in violations["RC-001"]
    assert "min_distinct_sources>2" in violations["RC-001"]


def test_assess_target_metrics_handles_mixed_directions():
    assessment = assess_target_metrics(
        {
            "verified_claim_coverage": 0.83,
            "unsupported_claim_rate": 0.14,
        },
        {
            "verified_claim_coverage": {"target": 0.828, "direction": "at_least"},
            "unsupported_claim_rate": {"target": 0.1225, "direction": "at_most"},
        },
    )

    assert assessment["metrics"]["verified_claim_coverage"]["met"] is True
    assert assessment["metrics"]["unsupported_claim_rate"]["met"] is False
    assert assessment["all_met"] is False


def test_load_existing_rows_validates_resume_metadata(tmp_path):
    output_path = tmp_path / "resume.json"
    output_path.write_text(
        """
        {
          "version": "v2",
          "split": "all",
          "mode": "live_llm_real_retrieval_real_nli",
          "profile": null,
          "rows": [
            {"item_id": "BO-001", "split": "dev"}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    rows = load_existing_rows(output_path, "v2", "all", "live_llm_real_retrieval_real_nli", profile_name=None)

    assert rows == {"BO-001": {"item_id": "BO-001", "split": "dev"}}


def test_load_evidence_store_for_companies_preserves_chunk_metadata(tmp_path, monkeypatch):
    normalized_dir = tmp_path / "normalized" / "fastly"
    normalized_dir.mkdir(parents=True)
    chunks_path = normalized_dir / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "fastly_q4_2025_results",
                        "source_id": "fastly_q4_2025_results",
                        "chunk_id": "chunk-001",
                        "text": "Revenue was $144.5 million.",
                        "source_type": "quarterly_results",
                        "is_primary": True,
                    }
                )
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    evidence_store = load_evidence_store_for_companies(["fastly"])

    assert evidence_store["fastly_q4_2025_results"][0]["chunk_id"] == "chunk-001"
    assert evidence_store["fastly_q4_2025_results"][0]["text"] == "Revenue was $144.5 million."


def test_is_retryable_live_error_recognizes_provider_502_message():
    request = httpx.Request("POST", "https://callflow.top/v1/chat/completions")
    response = httpx.Response(502, request=request)
    error = APIStatusError(
        "unknown provider",
        response=response,
        body={"error": {"message": "unknown provider for model gpt-5.2", "type": "server_error"}},
    )

    assert is_retryable_live_error(error) is True
