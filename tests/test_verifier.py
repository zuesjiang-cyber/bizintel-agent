from verification.claim_extractor import ClaimExtractor
from verification.evidence_verifier import EvidenceVerifier
from agent.schemas import Claim, ConfidenceLevel
from agent.config import settings
import numpy as np


class TestClaimExtractor:
    def setup_method(self):
        self.extractor = ClaimExtractor()

    def test_extract_basic_claim(self):
        text = "Stripe was founded in 2010 [Source: stripe_profile]. The company is headquartered in San Francisco [Source: stripe_about]."
        claims = self.extractor.extract_claims(text, "company_profile")
        assert len(claims) == 2

    def test_extract_numbers(self):
        text = "Stripe has raised $8.7 billion in total funding [Source: stripe_profile]."
        claims = self.extractor.extract_claims(text, "financials")
        assert len(claims) == 1
        assert claims[0].contains_numbers
        assert any("8.7" in n for n in claims[0].extracted_numbers)

    def test_extract_citations(self):
        text = "The company serves over 1 million businesses [Source: stripe_about]."
        claims = self.extractor.extract_claims(text, "overview")
        assert "stripe_about" in claims[0].cited_sources

    def test_extract_numbers_ignores_citation_ids(self):
        text = "Fastly revenue was $100 million [Source: fastly_q4]."
        claims = self.extractor.extract_claims(text, "overview")

        assert claims[0].extracted_numbers == ["$100 million"]

    def test_filter_filler(self):
        text = "In summary, the analysis shows positive signals. Overall, the company is well-positioned."
        claims = self.extractor.extract_claims(text, "summary")
        assert len(claims) == 0

    def test_extract_markdown_bullets_without_merging_headings(self):
        text = """## Executive Summary
- Cloudflare is a connectivity cloud company [Chunk: c1] [Source: cloudflare_10k].

## Coverage Contract
- business_model

## Business Model
- Cloudflare sells network and security services [Chunk: c2] [Source: cloudflare_q4_call].
"""
        claims = self.extractor.extract_claims(text, "summary")

        assert len(claims) == 2
        assert claims[0].cited_sources == ["cloudflare_10k"]
        assert claims[1].cited_sources == ["cloudflare_q4_call"]

    def test_filter_insufficient_evidence_lines(self):
        text = "- Insufficient evidence in the source pack to verify this point directly."
        claims = self.extractor.extract_claims(text, "summary")
        assert claims == []


class TestEvidenceVerifier:
    def setup_method(self):
        self.verifier = EvidenceVerifier()

    def test_supported_claim(self):
        claim = Claim(
            claim_id="test1",
            text="Stripe was founded in 2010 by Patrick and John Collison",
            section="profile",
            cited_sources=["stripe_about"],
            contains_numbers=True,
            extracted_numbers=["2010"],
        )
        evidence_store = {
            "stripe_about": [
                "Stripe is a technology company founded in 2010 by Irish entrepreneur brothers Patrick and John Collison."
            ]
        }

        results = self.verifier.verify_memo([claim], evidence_store)
        assert len(results) == 1
        assert results[0].confidence in (ConfidenceLevel.STRONG, ConfidenceLevel.MODERATE)
        assert results[0].supporting_source_ids == ["stripe_about"]
        assert results[0].supporting_chunk_ids

    def test_unsupported_claim(self):
        claim = Claim(
            claim_id="test2",
            text="Stripe has 50,000 employees worldwide",
            section="profile",
            cited_sources=["nonexistent_source"],
        )
        evidence_store = {}

        results = self.verifier.verify_memo([claim], evidence_store)
        assert results[0].confidence == ConfidenceLevel.UNSUPPORTED
        assert results[0].failure_reason == "bad_source_id"

    def test_missing_citation_claim_records_reason(self):
        claim = Claim(
            claim_id="test_missing",
            text="Stripe has 50,000 employees worldwide",
            section="profile",
            cited_sources=["no_citation"],
        )
        evidence_store = {
            "stripe_about": ["Stripe has roughly 8,000 employees."]
        }

        results = self.verifier.verify_memo([claim], evidence_store)

        assert results[0].confidence == ConfidenceLevel.UNSUPPORTED
        assert results[0].failure_reason == "missing_citation"

    def test_summary_stats(self):
        claims = [
            Claim(claim_id="c1", text="Claim 1", section="s1", cited_sources=["src1"]),
            Claim(claim_id="c2", text="Claim 2", section="s1", cited_sources=["src2"]),
        ]
        evidence_store = {
            "src1": ["Supporting evidence for claim 1"],
            # src2 is missing
        }
        results = self.verifier.verify_memo(claims, evidence_store)
        stats = self.verifier.summary_stats(results)
        assert stats["total_claims"] == 2
        assert "citation_marker_coverage" in stats
        assert "verified_claim_coverage" in stats

    def test_predict_pairs_batches_requests(self, monkeypatch):
        verifier = EvidenceVerifier(use_dummy_model=True)
        monkeypatch.setattr(settings, "nli_batch_size", 2)

        calls = []

        class FakeModel:
            def predict(self, pairs, show_progress_bar=False, batch_size=32):
                calls.append((len(pairs), batch_size))
                return np.array([[0.1, 0.8, 0.1]] * len(pairs))

        pairs = [("e1", "c1"), ("e2", "c2"), ("e3", "c3"), ("e4", "c4"), ("e5", "c5")]
        scores = verifier._predict_pairs(FakeModel(), pairs)

        assert scores.shape == (5, 3)
        assert calls == [(2, 2), (2, 2), (1, 2)]

    def test_predict_pairs_retries_with_smaller_batch_on_mps_oom(self, monkeypatch):
        verifier = EvidenceVerifier(use_dummy_model=True)
        verifier.device = "mps"
        monkeypatch.setattr(settings, "nli_batch_size", 4)
        monkeypatch.setattr(verifier, "_empty_device_cache", lambda: None)

        calls = []

        class FakeModel:
            def predict(self, pairs, show_progress_bar=False, batch_size=32):
                calls.append((len(pairs), batch_size))
                if batch_size > 2:
                    raise RuntimeError("MPS backend out of memory")
                return np.array([[0.1, 0.8, 0.1]] * len(pairs))

        pairs = [("e1", "c1"), ("e2", "c2"), ("e3", "c3")]
        scores = verifier._predict_pairs(FakeModel(), pairs)

        assert scores.shape == (3, 3)
        assert calls == [(3, 4), (2, 2), (1, 2)]

    def test_select_candidate_chunks_limits_per_source_and_prefers_overlap(self, monkeypatch):
        verifier = EvidenceVerifier(use_dummy_model=True)
        monkeypatch.setattr(settings, "verification_max_chunks_per_source", 2)
        claim = Claim(
            claim_id="c3",
            text="Cloudflare sells developer services and security products with usage-based expansion",
            section="business_model",
            cited_sources=["cloudflare_2024_10k"],
            contains_numbers=False,
            extracted_numbers=[],
        )
        chunks = [
            "Unrelated board governance language with no product detail.",
            "Cloudflare security products include WAF, DDoS mitigation, and Zero Trust services.",
            "Developers adopt Workers and then expand usage-based spend over time.",
            "Another unrelated paragraph about facilities.",
        ]

        selected = verifier._select_candidate_chunks(claim, chunks)

        assert len(selected) == 2
        selected_texts = [item["text"] for item in selected]
        assert chunks[1] in selected_texts
        assert chunks[2] in selected_texts

    def test_numeric_alignment_does_not_consume_following_word_initial(self):
        verifier = EvidenceVerifier(use_dummy_model=True)

        assert verifier._normalize_numeric_haystack(
            "Stripe was founded in 2010 by Patrick and John Collison."
        ) == {"2010"}
