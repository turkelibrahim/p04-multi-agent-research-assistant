"""Girdi doğrulama (K02), URL normalleştirme, kopya tespiti ve programatik alıntı kuralları (K04/K06)."""

from __future__ import annotations

from p04_research.config import redact_secrets, validate_question
from p04_research.models import (
    ClaimRecord,
    EvidenceChunk,
    EvidenceLevel,
    EvidenceRole,
    SourceRecord,
    VerdictStatus,
    VerifiedClaimLLM,
    VerifierEvidenceLLM,
)
from p04_research.verification_rules import (
    compute_content_fingerprint,
    enforce_verification_guardrails,
    is_quote_cut_at_boundary,
    normalize_url,
    quote_matches_source,
    truncate_source_text,
)


def test_question_validation_bounds():
    ok_empty, _, err_empty = validate_question("   ", "tr")
    assert not ok_empty
    assert "boş" in err_empty.lower()

    ok_short, _, err_short = validate_question("kısa soru", "tr")
    assert not ok_short
    assert "10" in err_short

    ok_long, _, err_long = validate_question("a" * 2001, "en")
    assert not ok_long
    assert "2000" in err_long

    ok_valid, cleaned, err_valid = validate_question("   Kampüs   kütüphanesi   bekleme   süresi?  ", "tr")
    assert ok_valid
    assert cleaned == "Kampüs kütüphanesi bekleme süresi?"
    assert err_valid == ""


def test_redact_secrets():
    msg = "Auth failed with key sk-test1234567890abcdef and tvly-abcd1234efgh"
    masked = redact_secrets(msg, ["sk-test1234567890abcdef"])
    assert "sk-test" not in masked
    assert "tvly-abcd" not in masked
    assert "[GIZLI_ANAHTAR]" in masked


def test_normalize_url_and_duplicate_fingerprint():
    u1 = normalize_url("https://www.Example.com/rapor/?utm_source=x&b=2&a=1#sec")
    u2 = normalize_url("https://example.com/rapor?a=1&b=2&fbclid=zzz")
    assert u1 == u2 == "https://example.com/rapor?a=1&b=2"

    fp1 = compute_content_fingerprint("Kampüs bisiklet pilotunda Ekim ayında 1.840 sürüş oldu!")
    fp2 = compute_content_fingerprint("kampüs bisiklet pilotunda ekim ayında 1 840 sürüş oldu")
    assert fp1 == fp2


def test_quote_matching_and_truncation_boundary():
    src_text = "Hafta içi ortalama bekleme süresi   8 dakikadan\n6 dakikaya düştü."
    assert quote_matches_source("ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü.", src_text)
    assert not quote_matches_source("ortalama bekleme süresi 8 dakikadan 4 dakikaya düştü.", src_text)

    long_text = ("A" * 3490) + "yarım kalan cümle burada kesilir"
    truncated, flag = truncate_source_text(long_text, max_chars=3500)
    assert flag is True
    assert len(truncated) == 3500
    chunk = EvidenceChunk(
        chunk_id="E1",
        source_id="S1",
        round_no=1,
        excerpt_text=truncated,
        char_count=len(truncated),
        is_truncated=True,
    )
    assert is_quote_cut_at_boundary(truncated[-25:], chunk) is True


def test_guardrails_reject_unknown_source_fabricated_quote_and_duplicate_conflict():
    s1 = SourceRecord(
        source_id="S1",
        url="https://example.com/1",
        normalized_url="https://example.com/1",
        title="Kaynak 1",
        publisher="P1",
        retrieved_at="2026-11-20T00:00:00Z",
        content_fingerprint="fp_same",
        evidence_level=EvidenceLevel.SECONDARY_REPORT,
    )
    s2_dup = SourceRecord(
        source_id="S2",
        url="https://mirror.example.com/2",
        normalized_url="https://mirror.example.com/2",
        title="Kopya Kaynak 2",
        publisher="P2",
        retrieved_at="2026-11-20T00:00:00Z",
        content_fingerprint="fp_same",
        duplicate_of_source_id="S1",
        evidence_level=EvidenceLevel.SECONDARY_REPORT,
    )
    e1 = EvidenceChunk(
        chunk_id="E1",
        source_id="S1",
        round_no=1,
        excerpt_text="Ölçülen gerçek değer 120 ziyarettir.",
        char_count=35,
    )
    e2 = EvidenceChunk(
        chunk_id="E2",
        source_id="S2",
        round_no=1,
        excerpt_text="Ölçülen gerçek değer 120 ziyarettir.",
        char_count=35,
    )
    sources = {"S1": s1, "S2": s2_dup}
    chunks = {"E1": e1, "E2": e2}
    claim = ClaimRecord(
        claim_id="C1",
        sq_id="SQ1",
        statement="Test iddiası",
        initial_source_id="S1",
        initial_chunk_id="E1",
        initial_quote="Ölçülen gerçek değer 120 ziyarettir.",
    )

    # 1) Bilinmeyen S99 -> reddedilir ve Kanıt yetersiz'e düşer
    v_unknown = enforce_verification_guardrails(
        claim,
        VerifiedClaimLLM(
            claim_id="C1",
            verdict=VerdictStatus.SUPPORTED,
            rationale="S99 doğruluyor.",
            evidence_items=[
                VerifierEvidenceLLM(
                    source_id="S99",
                    chunk_id="E1",
                    quote="Ölçülen gerçek değer 120 ziyarettir.",
                    role=EvidenceRole.SUPPORTING,
                )
            ],
        ),
        round_no=1,
        sources_by_id=sources,
        chunks_by_id=chunks,
    )
    assert v_unknown.verdict == VerdictStatus.INSUFFICIENT

    # 2) Uydurma alıntı -> reddedilir ve Kanıt yetersiz'e düşer
    v_fake = enforce_verification_guardrails(
        claim,
        VerifiedClaimLLM(
            claim_id="C1",
            verdict=VerdictStatus.SUPPORTED,
            rationale="Uydurma alıntı.",
            evidence_items=[
                VerifierEvidenceLLM(
                    source_id="S1",
                    chunk_id="E1",
                    quote="Ölçülen gerçek değer 500 ziyarettir.",
                    role=EvidenceRole.SUPPORTING,
                )
            ],
        ),
        round_no=1,
        sources_by_id=sources,
        chunks_by_id=chunks,
    )
    assert v_fake.verdict == VerdictStatus.INSUFFICIENT

    # 3) Kopya S2 ile çelişki kararı verilemez (tek bağımsız kaynak var)
    v_dup_conf = enforce_verification_guardrails(
        claim,
        VerifiedClaimLLM(
            claim_id="C1",
            verdict=VerdictStatus.CONFLICTING,
            rationale="İki kaynak çelişiyor.",
            evidence_items=[
                VerifierEvidenceLLM(
                    source_id="S1",
                    chunk_id="E1",
                    quote="Ölçülen gerçek değer 120 ziyarettir.",
                    role=EvidenceRole.CONFLICTING,
                ),
                VerifierEvidenceLLM(
                    source_id="S2",
                    chunk_id="E2",
                    quote="Ölçülen gerçek değer 120 ziyarettir.",
                    role=EvidenceRole.CONFLICTING,
                ),
            ],
        ),
        round_no=1,
        sources_by_id=sources,
        chunks_by_id=chunks,
    )
    assert v_dup_conf.verdict == VerdictStatus.INSUFFICIENT
