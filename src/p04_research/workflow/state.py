"""LangGraph ResearchGraphState tanımı."""

from __future__ import annotations

from typing import TypedDict

from p04_research.models import (
    ClaimRecord,
    EvidenceChunk,
    FinalReport,
    RunEvent,
    SearchRoundRecord,
    SourceRecord,
    SubQuestion,
    VerificationRecord,
)


class ResearchGraphState(TypedDict):
    run_id: str
    idempotency_key: str
    question: str
    language: str
    depth: str
    mode: str
    model_name: str
    started_at: str
    started_monotonic: float
    round_no: int
    plan_summary: str
    scope_notes: str
    research_purpose: str
    timeframe: str
    key_concepts: list[str]
    sub_questions: list[SubQuestion]
    pending_queries: list[str]
    executed_queries: list[str]
    follow_up_reason: str | None
    follow_up_claim_id: str | None
    follow_up_missing_info: str | None
    search_rounds: list[SearchRoundRecord]
    sources: list[SourceRecord]
    chunks: list[EvidenceChunk]
    claims: list[ClaimRecord]
    verifications: list[VerificationRecord]
    events: list[RunEvent]
    model_calls_used: int
    search_calls_used: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    token_measurement_type: str
    fatal_error_type: str | None
    fatal_error_message: str | None
    technical_search_failure: bool
    status: str
    quality_retry_count: int
    quality_feedback: list[str]
    quality_passed: bool
    research_quality: str
    report: FinalReport | None
