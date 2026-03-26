from tools.resolve_benchmark_evidence import normalize, resolve_entry


def test_normalize_collapses_whitespace():
    assert normalize("A   B\nC") == "a b c"


def test_resolve_entry_matches_anchor_text():
    chunks_by_doc = {
        "doc_a": [
            {"doc_id": "doc_a", "chunk_id": "chunk-1", "chunk_index": 0, "text": "Revenue grew 20 percent year over year."},
            {"doc_id": "doc_a", "chunk_id": "chunk-2", "chunk_index": 1, "text": "Margin expanded materially."},
        ]
    }

    resolved = resolve_entry(
        {"doc_id": "doc_a", "anchor_text": "Revenue grew 20 percent year over year"},
        chunks_by_doc,
    )

    assert resolved["chunk_id"] == "chunk-1"
    assert resolved["chunk_index"] == 0
    assert resolved["binding_method"] == "anchor_text_substring"
