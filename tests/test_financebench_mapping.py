import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "financebench_mapping.py"
SPEC = importlib.util.spec_from_file_location("bizintel_eval_financebench_mapping", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

score_gold_answer_mapping = MODULE.score_gold_answer_mapping


def test_score_gold_answer_mapping_detects_numeric_and_citation_hits():
    markdown = "Capital expenditures were $1577.00 [Source: financebench_3m_2018_10k]."
    answer_row = {
        "evaluation_mapping": {
            "mode": "numeric_exact",
            "expected_answer": "$1577.00",
            "normalized_answer_tokens": ["$1577.00"],
            "required_doc_id": "financebench_3m_2018_10k",
        }
    }

    metrics = score_gold_answer_mapping(markdown, answer_row)

    assert metrics["gold_answer_hit"] == 1.0
    assert metrics["gold_numeric_hit"] == 1.0
    assert metrics["gold_citation_hit"] == 1.0
    assert metrics["gold_semantic_hit"] == 1.0


def test_score_gold_answer_mapping_returns_unmapped_defaults():
    metrics = score_gold_answer_mapping("Revenue improved.", {})

    assert metrics["gold_answer_mode"] == "unmapped"
    assert metrics["gold_answer_hit"] == 0.0
    assert metrics["gold_semantic_similarity"] == 0.0


def test_score_gold_answer_mapping_detects_semantic_match_without_exact_verbatim():
    markdown = (
        "The company did not have positive working capital in FY2022 and instead "
        "reported negative working capital of about $1.56 billion. "
        "[Source: financebench_american_water_works_2022_10k]"
    )
    answer_row = {
        "evaluation_mapping": {
            "mode": "semantic_gold_answer",
            "expected_answer": "No, American Water Works had negative working capital of -$1561M in FY 2022.",
            "expected_justification": "Working capital was negative in FY2022.",
            "required_doc_id": "financebench_american_water_works_2022_10k",
        }
    }

    metrics = score_gold_answer_mapping(markdown, answer_row)

    assert metrics["gold_answer_hit"] == 0.0
    assert metrics["gold_semantic_similarity"] >= 0.35
    assert metrics["gold_semantic_hit"] == 1.0
