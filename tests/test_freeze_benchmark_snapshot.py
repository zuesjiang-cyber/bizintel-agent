from tools.freeze_benchmark_snapshot import build_snapshot


def test_build_snapshot_is_stable_for_same_inputs():
    snapshot_a = build_snapshot("v2", ["cloudflare"])
    snapshot_b = build_snapshot("v2", ["cloudflare"])

    assert snapshot_a["snapshot_id"] == snapshot_b["snapshot_id"]
    assert snapshot_a["companies"][0]["company"] == "cloudflare"
    assert snapshot_a["companies"][0]["document_count"] >= 1
