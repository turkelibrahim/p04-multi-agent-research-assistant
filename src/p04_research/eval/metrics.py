"""K11 akademik değerlendirme ve kalite güvence metrikleri (sıfır paydada 'Ölçülemedi' ve başarısız koşum hariç tutma)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalClaimItem:
    claim_id: str
    statement: str
    cited_source_ids: list[str]
    quoted_text: str
    quoted_source_id: str
    verdict: str  # supported | contradicted | conflicting | insufficient | irrelevant
    human_supported: bool | None = None


@dataclass(slots=True)
class EvalRunRecord:
    case_id: str
    method: str  # single_agent | multi_agent
    status: str  # completed | partial | failed
    duration_ms: int
    model_calls: int
    search_calls: int
    sources_by_id: dict[str, str]  # source_id -> stored excerpt
    claims: list[EvalClaimItem] = field(default_factory=list)
    collected_sources: int = 0
    accessible_sources: int = 0
    answered_sub_questions: int = 0
    total_sub_questions: int = 0
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


@dataclass(slots=True)
class MethodMetrics:
    method: str
    total_cases: int
    completed_cases: int
    total_claims: int
    supported_claims: int
    citation_valid_count: int
    citation_total_count: int
    citation_validity_rate: str
    quote_match_count: int
    quote_checked_count: int
    quote_match_rate: str
    human_supported_count: int
    human_scored_count: int
    human_support_rate: str
    mean_duration_ms: float
    total_model_calls: int
    total_search_calls: int
    total_collected_sources: int = 0
    total_accessible_sources: int = 0
    rejected_claims: int = 0
    irrelevant_claims: int = 0
    irrelevant_rate: str = "Ölçülemedi"
    sub_question_coverage_rate: str = "Ölçülemedi"
    total_tokens_reported: int | None = None
    estimated_cost_usd_total: float | None = None


def _format_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "Ölçülemedi"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def evaluate_method_runs(
    method: str,
    runs: list[EvalRunRecord],
) -> MethodMetrics:
    """Bölüm 11 deney ölçütlerini ve ek QA metriklerini hesaplar; başarısız koşumların taslağını puana katmaz."""
    from p04_research.verification_rules import quote_matches_source

    completed = [r for r in runs if r.status in ("completed", "partial")]
    total_claims = 0
    supported_claims = 0
    rejected_claims = 0
    irrelevant_claims = 0
    cit_valid = 0
    cit_total = 0
    q_match = 0
    q_checked = 0
    h_supp = 0
    h_scored = 0
    sq_ans = 0
    sq_tot = 0
    col_sources = 0
    acc_sources = 0
    tok_sum: int | None = None
    cost_sum: float | None = None

    for r in completed:
        col_sources += r.collected_sources or len(r.sources_by_id)
        acc_sources += r.accessible_sources or len(r.sources_by_id)
        sq_ans += r.answered_sub_questions
        sq_tot += r.total_sub_questions
        if r.total_tokens is not None:
            tok_sum = (tok_sum or 0) + r.total_tokens
        if r.estimated_cost_usd is not None:
            cost_sum = round((cost_sum or 0.0) + r.estimated_cost_usd, 6)

        for c in r.claims:
            total_claims += 1
            if c.verdict == "supported":
                supported_claims += 1
            elif c.verdict == "irrelevant":
                irrelevant_claims += 1
                rejected_claims += 1
            else:
                rejected_claims += 1

            for sid in c.cited_source_ids:
                cit_total += 1
                if sid in r.sources_by_id:
                    cit_valid += 1

            if c.quoted_text or c.verdict == "supported":
                q_checked += 1
                src_txt = r.sources_by_id.get(c.quoted_source_id, "")
                if c.quoted_text and src_txt and quote_matches_source(c.quoted_text, src_txt):
                    q_match += 1

            if c.human_supported is not None:
                h_scored += 1
                if c.human_supported:
                    h_supp += 1

    mean_dur = sum(r.duration_ms for r in runs) / len(runs) if runs else 0.0

    return MethodMetrics(
        method=method,
        total_cases=len(runs),
        completed_cases=len(completed),
        total_claims=total_claims,
        supported_claims=supported_claims,
        citation_valid_count=cit_valid,
        citation_total_count=cit_total,
        citation_validity_rate=_format_rate(cit_valid, cit_total),
        quote_match_count=q_match,
        quote_checked_count=q_checked,
        quote_match_rate=_format_rate(q_match, q_checked),
        human_supported_count=h_supp,
        human_scored_count=h_scored,
        human_support_rate=_format_rate(h_supp, h_scored),
        mean_duration_ms=round(mean_dur, 1),
        total_model_calls=sum(r.model_calls for r in runs),
        total_search_calls=sum(r.search_calls for r in runs),
        total_collected_sources=col_sources,
        total_accessible_sources=acc_sources,
        rejected_claims=rejected_claims,
        irrelevant_claims=irrelevant_claims,
        irrelevant_rate=_format_rate(irrelevant_claims, total_claims),
        sub_question_coverage_rate=_format_rate(sq_ans, sq_tot),
        total_tokens_reported=tok_sum,
        estimated_cost_usd_total=cost_sum,
    )
