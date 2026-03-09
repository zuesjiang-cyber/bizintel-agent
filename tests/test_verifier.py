import pytest
from verification.claim_extractor import ClaimExtractor
from verification.evidence_verifier import EvidenceVerifier
from agent.schemas import Claim, ConfidenceLevel


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

    def test_filter_filler(self):
        text = "In summary, the analysis shows positive signals. Overall, the company is well-positioned."
        claims = self.extractor.extract_claims(text, "summary")
        assert len(claims) == 0


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
        assert "citation_coverage" in stats
