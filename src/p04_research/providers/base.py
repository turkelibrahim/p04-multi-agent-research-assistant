"""Dil modeli ve web arama sağlayıcı sözleşmeleri ile hata türleri."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from p04_research.models import (
    AnalystOutput,
    ClaimRecord,
    EvidenceChunk,
    EvidenceLevel,
    PlannerOutput,
    ReporterSynthesisOutput,
    SourceRecord,
    SourceType,
    SubQuestion,
    VerificationRecord,
    VerifierOutput,
)


class ProviderError(Exception):
    """Dış veya iç sağlayıcı hatası (yetki, kota, zaman aşımı, ağ, API)."""

    def __init__(
        self,
        error_type: str,
        message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type  # auth_error | quota_error | timeout | network_error | api_error
        self.message = message
        self.retryable = retryable


@dataclass(slots=True)
class UsageDelta:
    model_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    measurement_type: str = "unmeasured"  # provider | estimated | unmeasured


@dataclass(slots=True)
class RawSearchItem:
    url: str
    title: str
    publisher: str
    content: str
    published_date: str | None = None
    language: str = "tr"
    provider: str = "mock_demo"
    is_mock: bool = False
    evidence_level: EvidenceLevel = EvidenceLevel.SECONDARY_REPORT
    author: str | None = None
    retrieval_method: str = "search_snippet"
    source_type: SourceType | None = None
    is_accessible: bool = True


@dataclass(slots=True)
class SearchResponse:
    query: str
    items: list[RawSearchItem] = field(default_factory=list)
    search_calls: int = 1


class LLMProvider(Protocol):
    """Dört model rolü için ortak sözleşme."""

    def plan(
        self,
        question: str,
        language: str,
        depth: str,
        max_initial_queries: int,
    ) -> tuple[PlannerOutput, UsageDelta]: ...

    def extract_claims(
        self,
        question: str,
        language: str,
        round_no: int,
        sub_questions: list[SubQuestion],
        sources: list[SourceRecord],
        chunks: list[EvidenceChunk],
        existing_claims: list[ClaimRecord],
    ) -> tuple[AnalystOutput, UsageDelta]: ...

    def verify_claims(
        self,
        question: str,
        language: str,
        round_no: int,
        claims: list[ClaimRecord],
        sources: list[SourceRecord],
        chunks: list[EvidenceChunk],
    ) -> tuple[VerifierOutput, UsageDelta]: ...

    def synthesize_report(
        self,
        question: str,
        language: str,
        sub_questions: list[SubQuestion],
        claims: list[ClaimRecord],
        verifications: list[VerificationRecord],
        sources: list[SourceRecord],
    ) -> tuple[ReporterSynthesisOutput, UsageDelta]: ...


class SearchProvider(Protocol):
    """Web araması düğümü için ortak sözleşme."""

    def search(
        self,
        query: str,
        max_results: int,
        depth: str,
    ) -> SearchResponse: ...
