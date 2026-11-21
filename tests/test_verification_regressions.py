"""Kanıt Doğrulama (Evidence Verification) regresyon testleri: C7, C5, C4, C3, C2 ve tek kaynak sınırı."""

from __future__ import annotations

from p04_research.models import (
    ClaimRecord,
    EvidenceChunk,
    EvidenceLevel,
    EvidenceRole,
    SourceRecord,
    SourceType,
    VerdictStatus,
    VerifiedClaimLLM,
    VerifierEvidenceLLM,
)
from p04_research.verification_rules import enforce_verification_guardrails


def _make_source_and_chunk(
    *,
    sid: str = "S1",
    cid: str = "E1",
    url: str = "https://example.com/doc",
    title: str = "Test Belgesi",
    publisher: str = "Test Kurumu",
    text: str,
    source_type: SourceType = SourceType.TRUSTED_SECONDARY_ANALYSIS,
) -> tuple[dict[str, SourceRecord], dict[str, EvidenceChunk]]:
    src = SourceRecord(
        source_id=sid,
        url=url,
        normalized_url=url,
        title=title,
        publisher=publisher,
        retrieved_at="2026-11-20T00:00:00Z",
        evidence_level=EvidenceLevel.SECONDARY_REPORT,
        source_type=source_type,
        is_accessible=True,
    )
    chunk = EvidenceChunk(
        chunk_id=cid,
        source_id=sid,
        round_no=1,
        excerpt_text=text,
        char_count=len(text),
    )
    return {sid: src}, {cid: chunk}


def test_regression_c7_question_sentence_2035_rejected():
    text = "Sektör oturumunda sorulan temel soru: 2035'te şirketlerde insan başına kaç dijital çalışan olacak?"
    sources, chunks = _make_source_and_chunk(text=text)
    claim = ClaimRecord(
        claim_id="C7",
        sq_id="SQ1",
        statement="2035'te şirketlerde insan başına kaç dijital çalışan olacak?",
        initial_source_id="S1",
        initial_chunk_id="E1",
        initial_quote="2035'te şirketlerde insan başına kaç dijital çalışan olacak?",
    )
    raw = VerifiedClaimLLM(
        claim_id="C7",
        verdict=VerdictStatus.SUPPORTED,
        rationale="Alıntı metinde geçiyor.",
        evidence_items=[
            VerifierEvidenceLLM(
                source_id="S1",
                chunk_id="E1",
                quote="2035'te şirketlerde insan başına kaç dijital çalışan olacak?",
                role=EvidenceRole.SUPPORTING,
            )
        ],
    )
    res = enforce_verification_guardrails(
        claim,
        raw,
        round_no=1,
        sources_by_id=sources,
        chunks_by_id=chunks,
        research_question="2026 yılında yapay zekâ ajanlarının iş dünyasına etkisi nedir?",
    )
    assert res.verdict != VerdictStatus.SUPPORTED
    assert res.verdict == VerdictStatus.INSUFFICIENT
    assert res.claim_nature == "question_or_heading"


def test_regression_c5_2022_llm_background_irrelevant_for_2026():
    text = "Büyük dil modelleri (LLM) 2022 yılında kamuya açılan sohbet botlarıyla yaygınlaşmaya başlamıştır."
    sources, chunks = _make_source_and_chunk(text=text)
    claim = ClaimRecord(
        claim_id="C5",
        sq_id="SQ2",
        statement="LLM'lerin 2022 yılında yaygınlaşması 2026 istihdamını artırmıştır.",
        initial_source_id="S1",
        initial_chunk_id="E1",
        initial_quote=text,
    )
    raw = VerifiedClaimLLM(
        claim_id="C5",
        verdict=VerdictStatus.SUPPORTED,
        rationale="2022 bilgisi kaynakta mevcut.",
        evidence_items=[
            VerifierEvidenceLLM(
                source_id="S1",
                chunk_id="E1",
                quote=text,
                role=EvidenceRole.SUPPORTING,
            )
        ],
    )
    res = enforce_verification_guardrails(
        claim,
        raw,
        round_no=1,
        sources_by_id=sources,
        chunks_by_id=chunks,
        research_question="2026 yılında yapay zekâ ajanlarının istihdam etkisi nedir?",
    )
    assert res.verdict == VerdictStatus.IRRELEVANT


def test_regression_c4_youtube_speaker_99_percent_not_verified_statistic():
    text = "YouTube videosunda konuşmacı şirketlerin %99 oranında ajana geçeceğini iddia etti."
    sources, chunks = _make_source_and_chunk(
        url="https://www.youtube.com/watch?v=xyz",
        title="YouTube AI Sohbeti",
        publisher="youtube.com",
        text=text,
        source_type=SourceType.VIDEO_SOCIAL_MEDIA,
    )
    claim = ClaimRecord(
        claim_id="C4",
        sq_id="SQ1",
        statement="Şirketlerin %99 oranında yapay zekâ ajanlarına geçeceği ölçülmüştür.",
        initial_source_id="S1",
        initial_chunk_id="E1",
        initial_quote=text,
    )
    raw = VerifiedClaimLLM(
        claim_id="C4",
        verdict=VerdictStatus.SUPPORTED,
        rationale="Videoda söylenmiş.",
        evidence_items=[
            VerifierEvidenceLLM(
                source_id="S1",
                chunk_id="E1",
                quote=text,
                role=EvidenceRole.SUPPORTING,
            )
        ],
    )
    res = enforce_verification_guardrails(
        claim,
        raw,
        round_no=1,
        sources_by_id=sources,
        chunks_by_id=chunks,
        research_question="2026 yılında şirketlerde yapay zekâ ajanı kullanımı nasıldır?",
    )
    assert res.verdict != VerdictStatus.SUPPORTED
    assert res.verdict == VerdictStatus.INSUFFICIENT
    assert res.claim_nature == "opinion_commentary"


def test_regression_c3_footnote_number_22_not_accepted_as_statistic():
    text = "Yapay zekâ ajanlarının kurumsal kullanımı hızla artmaktadır [22] ve iş akışlarını etkilemektedir."
    sources, chunks = _make_source_and_chunk(text=text)
    claim = ClaimRecord(
        claim_id="C3",
        sq_id="SQ2",
        statement="Kurumsal yapay zekâ ajanı kullanımı 22 düzeyinde artmıştır.",
        initial_source_id="S1",
        initial_chunk_id="E1",
        initial_quote=text,
    )
    raw = VerifiedClaimLLM(
        claim_id="C3",
        verdict=VerdictStatus.SUPPORTED,
        rationale="22 rakamı alıntıda var.",
        evidence_items=[
            VerifierEvidenceLLM(
                source_id="S1",
                chunk_id="E1",
                quote=text,
                role=EvidenceRole.SUPPORTING,
            )
        ],
    )
    res = enforce_verification_guardrails(
        claim,
        raw,
        round_no=1,
        sources_by_id=sources,
        chunks_by_id=chunks,
        research_question="Kurumsal yapay zekâ ajanı kullanımı nasıl artmaktadır?",
    )
    assert res.verdict != VerdictStatus.SUPPORTED
    assert res.verdict == VerdictStatus.INSUFFICIENT


def test_regression_c2_undefined_population_67_and_32_percent_rejected():
    text = "Yapay zekâ ajanları kapsamında %67 ve %32 oranları öne çıkmaktadır."
    sources, chunks = _make_source_and_chunk(text=text)
    claim = ClaimRecord(
        claim_id="C2",
        sq_id="SQ2",
        statement="Genel iş gücü etkisi olarak %67 ve %32 oranında dönüşüm yaşanmaktadır.",
        initial_source_id="S1",
        initial_chunk_id="E1",
        initial_quote=text,
    )
    raw = VerifiedClaimLLM(
        claim_id="C2",
        verdict=VerdictStatus.SUPPORTED,
        rationale="Yüzde oranları metinde var.",
        evidence_items=[
            VerifierEvidenceLLM(
                source_id="S1",
                chunk_id="E1",
                quote=text,
                role=EvidenceRole.SUPPORTING,
            )
        ],
    )
    res = enforce_verification_guardrails(
        claim,
        raw,
        round_no=1,
        sources_by_id=sources,
        chunks_by_id=chunks,
        research_question="2026 yılında yapay zekâ ajanlarının genel iş gücü etkisi nedir?",
    )
    assert res.verdict != VerdictStatus.SUPPORTED
    assert res.verdict == VerdictStatus.INSUFFICIENT
