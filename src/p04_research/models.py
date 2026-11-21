"""Pydantic v2 veri sözleşmeleri, alan modelleri ve yapılandırılmış LLM çıktıları."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RunLanguage(StrEnum):
    TR = "tr"
    EN = "en"


class ResearchDepth(StrEnum):
    FAST = "fast"
    STANDARD = "standard"


class RunMode(StrEnum):
    DEMO = "demo"
    LIVE = "live"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class SubQuestionStatus(StrEnum):
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    CONFLICTING = "conflicting"
    UNANSWERED = "unanswered"


class VerdictStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"
    INSUFFICIENT_EVIDENCE = "insufficient"
    IRRELEVANT = "irrelevant"


class EvidenceLevel(StrEnum):
    PRIMARY_RECORD = "primary_record"
    OFFICIAL_DOC = "official_doc"
    SECONDARY_REPORT = "secondary_report"
    SNIPPET_ONLY = "snippet_only"


class SourceType(StrEnum):
    ACADEMIC_PEER_REVIEWED = "academic_peer_reviewed"
    OFFICIAL_INSTITUTION = "official_institution"
    RESEARCH_ORG_REPORT = "research_org_report"
    CORPORATE_PRIMARY_DATA = "corporate_primary_data"
    TRUSTED_SECONDARY_ANALYSIS = "trusted_secondary_analysis"
    NEWS_BLOG_OPINION = "news_blog_opinion"
    VIDEO_SOCIAL_MEDIA = "video_social_media"


class EvidenceRole(StrEnum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    CONFLICTING = "conflicting"
    CONTEXT = "context"


class SubQuestion(BaseModel):
    sq_id: str
    order_index: int
    text: str
    scope_note: str = ""
    search_intent: str = ""
    required_evidence: str = ""
    status: SubQuestionStatus = SubQuestionStatus.UNANSWERED
    findings: list[str] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    plan_summary: str
    scope_notes: str = ""
    research_purpose: str = ""
    timeframe: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    initial_queries: list[str] = Field(default_factory=list)


class SearchRoundRecord(BaseModel):
    round_no: int
    query: str
    result_count: int
    source_ids: list[str] = Field(default_factory=list)
    triggered_by_claim_id: str | None = None
    trigger_reason: str | None = None
    missing_info: str | None = None
    error_message: str | None = None


class SourceRecord(BaseModel):
    source_id: str
    url: str
    normalized_url: str
    title: str
    publisher: str
    author: str | None = None
    published_date: str | None = None
    retrieved_at: str
    language: str = "tr"
    provider: str = "mock_demo"
    is_mock: bool = False
    evidence_level: EvidenceLevel = EvidenceLevel.SECONDARY_REPORT
    source_type: SourceType = SourceType.TRUSTED_SECONDARY_ANALYSIS
    retrieval_method: str = "search_snippet"  # search_snippet | full_text | primary_record | inaccessible
    retrieved_content: str = ""
    relevance_score: float = 1.0
    relevance_assessment: str = ""
    credibility_score: float = 0.7
    is_accessible: bool = True
    content_fingerprint: str = ""
    duplicate_of_source_id: str | None = None


class EvidenceChunk(BaseModel):
    chunk_id: str
    source_id: str
    round_no: int
    excerpt_text: str
    char_count: int
    is_truncated: bool = False
    query_used: str = ""


class ClaimRecord(BaseModel):
    claim_id: str
    sq_id: str
    statement: str
    initial_source_id: str
    initial_chunk_id: str
    initial_quote: str
    initial_quote_language: str = "tr"
    explanatory_note: str | None = None
    round_created: int = 1


class VerdictEvidenceItem(BaseModel):
    source_id: str
    chunk_id: str
    quote: str
    role: EvidenceRole = EvidenceRole.SUPPORTING
    quote_language: str = "tr"
    explanatory_note: str | None = None


class VerificationRecord(BaseModel):
    claim_id: str
    round_no: int
    verdict: VerdictStatus
    rationale: str
    missing_info: str | None = None
    follow_up_query: str | None = None
    evidence_items: list[VerdictEvidenceItem] = Field(default_factory=list)
    is_final: bool = True
    causation_warning: str | None = None
    guardrail_notes: list[str] = Field(default_factory=list)
    claim_nature: str = "direct_quote_measurement"
    numeric_context: str = ""
    independent_support_count: int = 1
    single_source_warning: str | None = None


class RunEvent(BaseModel):
    seq: int
    node_name: str
    round_no: int
    status: str  # start | finish | decision | error | info
    started_at: str
    finished_at: str | None = None
    duration_ms: int = 0
    message: str
    payload_summary: str = ""


class ReportFindingItem(BaseModel):
    claim_id: str
    sq_id: str
    statement: str
    citation_ids: list[str] = Field(default_factory=list)
    quotes: list[VerdictEvidenceItem] = Field(default_factory=list)
    rationale: str = ""
    causation_warning: str | None = None
    numeric_context: str = ""
    single_source_warning: str | None = None


class ConflictComparisonItem(BaseModel):
    claim_id: str
    sq_id: str
    statement: str
    explanation: str
    side_a: VerdictEvidenceItem
    side_b: VerdictEvidenceItem


class LimitationItem(BaseModel):
    claim_id: str
    sq_id: str
    statement: str
    verdict: VerdictStatus
    rationale: str
    missing_info: str | None = None
    initial_verdict: VerdictStatus | None = None
    initial_rationale: str | None = None
    counter_evidence: list[VerdictEvidenceItem] = Field(default_factory=list)


class BibliographyItem(BaseModel):
    source_id: str
    title: str
    publisher: str
    author: str | None = None
    published_date: str | None = None
    retrieved_at: str
    url: str
    language: str
    provider: str
    is_mock: bool
    evidence_level: EvidenceLevel
    source_type: SourceType = SourceType.TRUSTED_SECONDARY_ANALYSIS
    retrieval_method: str = "search_snippet"
    relevance_assessment: str = ""
    credibility_score: float = 0.7
    is_accessible: bool = True
    duplicate_of_source_id: str | None = None


class FinalReport(BaseModel):
    title: str
    language: RunLanguage
    short_answer: str
    research_purpose: str = ""
    methodology_and_scope: str = ""
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    supported_findings: list[ReportFindingItem] = Field(default_factory=list)
    quantitative_comparisons: list[str] = Field(default_factory=list)
    conflicts: list[ConflictComparisonItem] = Field(default_factory=list)
    counter_evidence_summary: list[str] = Field(default_factory=list)
    limitations: list[LimitationItem] = Field(default_factory=list)
    overall_assessment: str = ""
    open_questions: list[str] = Field(default_factory=list)
    bibliography: list[BibliographyItem] = Field(default_factory=list)
    report_markdown: str = ""
    error_notice: str | None = None
    research_quality: str = "high"
    quality_notes: list[str] = Field(default_factory=list)


class RunMetadata(BaseModel):
    run_id: str
    idempotency_key: str
    question: str
    language: RunLanguage
    depth: ResearchDepth
    mode: RunMode
    model_name: str
    started_at: str
    finished_at: str | None = None
    duration_ms: int = 0
    status: RunStatus = RunStatus.RUNNING
    plan_summary: str = ""
    scope_notes: str = ""
    research_purpose: str = ""
    timeframe: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    model_calls: int = 0
    search_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    token_measurement_type: str = "unmeasured"  # provider | estimated | unmeasured
    estimated_cost_usd: float | None = None
    research_quality: str = "high"
    error_type: str | None = None
    error_message: str | None = None


class RunBundle(BaseModel):
    metadata: RunMetadata
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    search_rounds: list[SearchRoundRecord] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)
    verifications: list[VerificationRecord] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)
    report: FinalReport | None = None


class ClaimProvenanceView(BaseModel):
    claim: ClaimRecord
    sub_question: SubQuestion | None = None
    initial_source: SourceRecord | None = None
    initial_chunk: EvidenceChunk | None = None
    round1_verification: VerificationRecord | None = None
    follow_up_search_round: SearchRoundRecord | None = None
    final_verification: VerificationRecord | None = None
    all_verifications: list[VerificationRecord] = Field(default_factory=list)


# --- LLM Yapılandırılmış Çıktı Sözleşmeleri (serbest dict içermez) ---


class PlannedSubQuestionLLM(BaseModel):
    sq_id: str
    text: str
    scope_note: str = ""
    search_intent: str = ""
    required_evidence: str = ""


class PlannerOutput(BaseModel):
    plan_summary: str
    scope_notes: str = ""
    research_purpose: str = ""
    timeframe: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    sub_questions: list[PlannedSubQuestionLLM]
    initial_queries: list[str]


class ExtractedClaimLLM(BaseModel):
    claim_id: str
    sq_id: str
    statement: str
    initial_source_id: str
    initial_chunk_id: str
    initial_quote: str
    initial_quote_language: str = "tr"
    explanatory_note: str | None = None


class AnalystOutput(BaseModel):
    claims: list[ExtractedClaimLLM]


class VerifierEvidenceLLM(BaseModel):
    source_id: str
    chunk_id: str
    quote: str
    role: EvidenceRole
    quote_language: str = "tr"
    explanatory_note: str | None = None


class VerifiedClaimLLM(BaseModel):
    claim_id: str
    verdict: VerdictStatus
    rationale: str
    missing_info: str | None = None
    follow_up_query: str | None = None
    causation_warning: str | None = None
    claim_nature: str = "direct_quote_measurement"
    numeric_context: str = ""
    evidence_items: list[VerifierEvidenceLLM] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    verifications: list[VerifiedClaimLLM]


class SubQuestionUpdateLLM(BaseModel):
    sq_id: str
    status: SubQuestionStatus


class ReporterSynthesisOutput(BaseModel):
    title: str
    short_answer: str
    sub_question_updates: list[SubQuestionUpdateLLM] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    limitations_summary: str | None = None
    overall_assessment: str = ""
