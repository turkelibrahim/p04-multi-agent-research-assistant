"""LangGraph 5 araştırma düğümü, Quality Gate kalite geçidi ve olay günlüğü yardımcıları."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import UTC, datetime

from p04_research.config import AppConfig, redact_secrets
from p04_research.models import (
    BibliographyItem,
    ClaimRecord,
    ConflictComparisonItem,
    EvidenceChunk,
    FinalReport,
    LimitationItem,
    ReportFindingItem,
    ResearchDepth,
    RunBundle,
    RunEvent,
    RunLanguage,
    RunMetadata,
    RunMode,
    RunStatus,
    SearchRoundRecord,
    SourceRecord,
    SubQuestion,
    SubQuestionStatus,
    VerdictStatus,
    VerificationRecord,
)
from p04_research.providers.base import LLMProvider, ProviderError, SearchProvider, UsageDelta
from p04_research.providers.live_extractive import _build_academic_synthesis
from p04_research.providers.mock_demo import clean_question_title
from p04_research.verification_rules import (
    compute_content_fingerprint,
    enforce_verification_guardrails,
    evaluate_source_quality_and_relevance,
    normalize_url,
    normalize_whitespace_nfc,
    truncate_source_text,
)
from p04_research.workflow.i18n import t, verdict_label
from p04_research.workflow.state import ResearchGraphState

EventCallback = Callable[[RunEvent], None]
_CITATION_TAG_RE = re.compile(r"\[(S\d+)\]")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _add_usage(state: ResearchGraphState, delta: UsageDelta) -> dict:
    p_tok = state["prompt_tokens"]
    c_tok = state["completion_tokens"]
    t_tok = state["total_tokens"]
    if delta.prompt_tokens is not None:
        p_tok = (p_tok or 0) + delta.prompt_tokens
    if delta.completion_tokens is not None:
        c_tok = (c_tok or 0) + delta.completion_tokens
    if delta.total_tokens is not None:
        t_tok = (t_tok or 0) + delta.total_tokens

    cur_type = state.get("token_measurement_type", "unmeasured")
    if delta.measurement_type == "provider":
        new_type = "provider"
    elif delta.measurement_type == "estimated" and cur_type != "provider":
        new_type = "estimated"
    else:
        new_type = cur_type

    return {
        "model_calls_used": state["model_calls_used"] + delta.model_calls,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": t_tok,
        "token_measurement_type": new_type,
    }


def _append_event(
    events: list[RunEvent],
    node_name: str,
    round_no: int,
    status: str,
    started_at: str,
    duration_ms: int,
    message: str,
    payload_summary: str = "",
    on_event: EventCallback | None = None,
) -> list[RunEvent]:
    ev = RunEvent(
        seq=len(events) + 1,
        node_name=node_name,
        round_no=round_no,
        status=status,
        started_at=started_at,
        finished_at=_utc_now_iso(),
        duration_ms=duration_ms,
        message=message,
        payload_summary=payload_summary,
    )
    if on_event is not None:
        on_event(ev)
    return [*events, ev]


def _elapsed_sec(state: ResearchGraphState) -> float:
    return time.monotonic() - state["started_monotonic"]


def _budget_allows_model(state: ResearchGraphState, config: AppConfig) -> tuple[bool, str]:
    if state["model_calls_used"] >= config.max_model_calls:
        msg = (
            f"Model çağrı bütçesi doldu ({state['model_calls_used']}/{config.max_model_calls})."
            if state["language"] == "tr"
            else f"Model call budget reached ({state['model_calls_used']}/{config.max_model_calls})."
        )
        return False, msg
    if _elapsed_sec(state) >= config.total_run_timeout_sec:
        msg = (
            f"Toplam süre sınırı ({config.total_run_timeout_sec} sn) doldu."
            if state["language"] == "tr"
            else f"Total run timeout ({config.total_run_timeout_sec}s) reached."
        )
        return False, msg
    return True, ""


def _pick_followup_candidate(
    state: ResearchGraphState,
) -> tuple[VerificationRecord, str] | None:
    """2. tur arama için uygun ilk çözülmemiş kanıt boşluğunu seçer."""
    executed_norm = {normalize_whitespace_nfc(q).lower() for q in state["executed_queries"]}
    final_vers = [v for v in state["verifications"] if v.is_final]
    for ver in final_vers:
        if ver.verdict != VerdictStatus.INSUFFICIENT:
            continue
        q = (ver.follow_up_query or "").strip()
        if not q:
            continue
        if normalize_whitespace_nfc(q).lower() in executed_norm:
            continue
        return ver, q
    return None


def _deduplicate_sub_questions(raw_sqs) -> list[SubQuestion]:
    """Tekrarlanan alt soruları metin benzerliğiyle eler (en çok 6 benzersiz alt soru)."""
    seen_texts: list[set[str]] = []
    result: list[SubQuestion] = []
    for sq in raw_sqs:
        norm = normalize_whitespace_nfc(sq.text).lower()
        words = {w for w in re.findall(r"\w+", norm) if len(w) >= 3}
        is_dup = False
        for prev_words in seen_texts:
            if not words or not prev_words:
                continue
            jaccard = len(words & prev_words) / len(words | prev_words)
            if jaccard >= 0.8:
                is_dup = True
                break
        if is_dup:
            continue
        seen_texts.append(words)
        result.append(
            SubQuestion(
                sq_id=sq.sq_id,
                order_index=len(result) + 1,
                text=sq.text,
                scope_note=sq.scope_note,
                search_intent=getattr(sq, "search_intent", ""),
                required_evidence=getattr(sq, "required_evidence", ""),
                status=SubQuestionStatus.UNANSWERED,
            )
        )
        if len(result) >= 6:
            break
    return result


def _deduplicate_sentences(text: str) -> str:
    """Rapor özetindeki tekrarlanan cümleleri temizler."""
    parts = re.split(r"(?<=[.!?])\s+", normalize_whitespace_nfc(text))
    seen: set[str] = set()
    kept: list[str] = []
    for p in parts:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            kept.append(p.strip())
    return " ".join(kept)


def make_nodes(
    config: AppConfig,
    llm_provider: LLMProvider,
    search_provider: SearchProvider,
    on_event: EventCallback | None = None,
) -> dict[str, Callable[[ResearchGraphState], dict]]:
    """Yapılandırma ve sağlayıcılarla bağlı düğümleri ve kalite geçidini oluşturur."""
    secrets = config.known_secrets()

    def planner_node(state: ResearchGraphState) -> dict:
        t0 = time.monotonic()
        started_at = _utc_now_iso()
        lang = state["language"]
        events = _append_event(
            list(state["events"]),
            "planner",
            state["round_no"],
            "start",
            started_at,
            0,
            "Soru alt sorulara ayrılıyor ve ilk arama sorguları hazırlanıyor."
            if lang == "tr"
            else "Splitting question into sub-questions and initial queries.",
            on_event=on_event,
        )

        ok, budget_msg = _budget_allows_model(state, config)
        if not ok:
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(events, "planner", 1, "error", started_at, dur, budget_msg, on_event=on_event)
            return {
                "events": events,
                "fatal_error_type": "budget_exhausted",
                "fatal_error_message": budget_msg,
            }

        max_q = 1 if state["depth"] == "fast" else config.max_initial_queries_standard
        try:
            plan_out, usage = llm_provider.plan(
                question=state["question"],
                language=lang,
                depth=state["depth"],
                max_initial_queries=max_q,
            )
            sub_questions = _deduplicate_sub_questions(plan_out.sub_questions)
            queries = [q.strip() for q in plan_out.initial_queries if q.strip()][:max_q]
            if not queries:
                queries = [state["question"][:180]]
            dur = int((time.monotonic() - t0) * 1000)
            summary = f"{len(sub_questions)} alt soru, {len(queries)} başlangıç sorgusu: {', '.join(queries)}"
            events = _append_event(
                events,
                "planner",
                1,
                "finish",
                started_at,
                dur,
                "Araştırma planı oluşturuldu." if lang == "tr" else "Research plan generated.",
                payload_summary=summary,
                on_event=on_event,
            )
            return {
                "plan_summary": plan_out.plan_summary,
                "scope_notes": plan_out.scope_notes,
                "research_purpose": plan_out.research_purpose or plan_out.plan_summary,
                "timeframe": plan_out.timeframe,
                "key_concepts": list(plan_out.key_concepts),
                "sub_questions": sub_questions,
                "pending_queries": queries,
                "events": events,
                **_add_usage(state, usage),
            }
        except ProviderError as exc:
            safe_msg = redact_secrets(exc.message, secrets)
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(events, "planner", 1, "error", started_at, dur, safe_msg, on_event=on_event)
            return {
                "events": events,
                "fatal_error_type": exc.error_type,
                "fatal_error_message": safe_msg,
            }

    def researcher_node(state: ResearchGraphState) -> dict:
        t0 = time.monotonic()
        started_at = _utc_now_iso()
        lang = state["language"]
        round_no = state["round_no"]
        events = _append_event(
            list(state["events"]),
            "researcher",
            round_no,
            "start",
            started_at,
            0,
            f"{round_no}. tur web/kayıt taraması yürütülüyor."
            if lang == "tr"
            else f"Running round {round_no} source search.",
            payload_summary="; ".join(state["pending_queries"]),
            on_event=on_event,
        )

        max_searches = config.max_search_calls_for_depth(state["depth"])
        search_calls_used = state["search_calls_used"]
        sources = list(state["sources"])
        chunks = list(state["chunks"])
        search_rounds = list(state["search_rounds"])
        executed_queries = list(state["executed_queries"])

        url_to_source: dict[str, SourceRecord] = {s.normalized_url: s for s in sources}
        fp_to_source: dict[str, SourceRecord] = {
            s.content_fingerprint: s for s in sources if s.content_fingerprint
        }

        def _title_key_tokens(raw_title: str) -> set[str]:
            clean_t = re.split(r"\s+[|\-—]\s+", raw_title or "", maxsplit=1)[0]
            clean_t = re.sub(r"\.\.\.$", "", clean_t).strip()
            folded = clean_t.translate(str.maketrans("çğıİöşüÇĞÖŞÜ", "cgiiosucgosu")).lower()
            words = re.findall(r"[a-z0-9]+", folded)
            return {w for w in words if len(w) >= 3 or w.isdigit()}

        def _find_title_duplicate(raw_title: str, excerpt_text: str) -> str | None:
            cand_toks = _title_key_tokens(raw_title)
            if len(cand_toks) < 3:
                return None
            cand_nums = {w for w in cand_toks if w.isdigit()}
            for prev in sources:
                if prev.is_mock or not prev.is_accessible or prev.duplicate_of_source_id:
                    continue
                prev_toks = _title_key_tokens(prev.title)
                if len(prev_toks) < 3:
                    continue
                prev_nums = {w for w in prev_toks if w.isdigit()}
                if cand_nums != prev_nums:
                    continue
                inter = len(cand_toks & prev_toks)
                min_len = min(len(cand_toks), len(prev_toks))
                jaccard = inter / len(cand_toks | prev_toks)
                if jaccard >= 0.82 or (min_len >= 4 and inter == min_len):
                    return prev.source_id
            del excerpt_text
            return None

        tech_error_type: str | None = None
        tech_error_msg: str | None = None
        new_sources_count = 0
        new_chunks_count = 0

        for query in state["pending_queries"]:
            if search_calls_used >= max_searches:
                break
            if _elapsed_sec(state) >= config.total_run_timeout_sec:
                tech_error_type = "timeout"
                tech_error_msg = (
                    f"Toplam süre sınırı ({config.total_run_timeout_sec} sn) arama aşamasında doldu."
                    if lang == "tr"
                    else f"Total timeout ({config.total_run_timeout_sec}s) reached during search."
                )
                break

            executed_queries.append(query)
            try:
                resp = search_provider.search(
                    query=query,
                    max_results=config.max_results_per_query,
                    depth=state["depth"],
                )
                search_calls_used += resp.search_calls
                round_source_ids: list[str] = []

                for item in resp.items[: config.max_results_per_query]:
                    norm_url = normalize_url(item.url)
                    if not norm_url:
                        continue
                    excerpt, is_trunc = truncate_source_text(item.content, config.max_source_chars)
                    fp = compute_content_fingerprint(excerpt)
                    eval_info = evaluate_source_quality_and_relevance(
                        url=item.url,
                        title=item.title,
                        publisher=item.publisher,
                        content=item.content,
                        question=state["question"],
                        author=item.author,
                        published_date=item.published_date,
                        evidence_level=item.evidence_level,
                        retrieval_method=item.retrieval_method if item.retrieval_method != "search_snippet" else None,
                        source_type=item.source_type,
                        is_accessible=item.is_accessible,
                    )

                    existing_src = url_to_source.get(norm_url)
                    if existing_src is None:
                        dup_of: str | None = None
                        if fp and fp in fp_to_source and eval_info["is_accessible"]:
                            dup_of = fp_to_source[fp].source_id
                        elif not item.is_mock and eval_info["is_accessible"]:
                            dup_of = _find_title_duplicate(item.title, excerpt)
                        sid = f"S{len(sources) + 1}"
                        src = SourceRecord(
                            source_id=sid,
                            url=item.url,
                            normalized_url=norm_url,
                            title=item.title,
                            publisher=item.publisher,
                            author=item.author,
                            published_date=item.published_date,
                            retrieved_at=_utc_now_iso(),
                            language=item.language,
                            provider=item.provider,
                            is_mock=item.is_mock,
                            evidence_level=eval_info["evidence_level"],
                            source_type=eval_info["source_type"],
                            retrieval_method=eval_info["retrieval_method"],
                            retrieved_content=excerpt if eval_info["is_accessible"] else "",
                            relevance_score=eval_info["relevance_score"],
                            relevance_assessment=eval_info["relevance_assessment"],
                            credibility_score=eval_info["credibility_score"],
                            is_accessible=eval_info["is_accessible"],
                            content_fingerprint=fp,
                            duplicate_of_source_id=dup_of,
                        )
                        sources.append(src)
                        url_to_source[norm_url] = src
                        if fp and fp not in fp_to_source and src.is_accessible:
                            fp_to_source[fp] = src
                        new_sources_count += 1
                    else:
                        src = existing_src

                    if src.source_id not in round_source_ids:
                        round_source_ids.append(src.source_id)

                    # Erişilemeyen sayfalardan okuma parçası oluşturulmaz
                    if not src.is_accessible or not excerpt:
                        continue

                    chunk_id = f"E{len(chunks) + 1}"
                    chunk = EvidenceChunk(
                        chunk_id=chunk_id,
                        source_id=src.source_id,
                        round_no=round_no,
                        excerpt_text=excerpt,
                        char_count=len(excerpt),
                        is_truncated=is_trunc,
                        query_used=query,
                    )
                    chunks.append(chunk)
                    new_chunks_count += 1

                search_rounds.append(
                    SearchRoundRecord(
                        round_no=round_no,
                        query=query,
                        result_count=len(resp.items),
                        source_ids=round_source_ids,
                        triggered_by_claim_id=state["follow_up_claim_id"] if round_no > 1 else None,
                        trigger_reason=state["follow_up_reason"] if round_no > 1 else None,
                        missing_info=state["follow_up_missing_info"] if round_no > 1 else None,
                    )
                )

            except ProviderError as exc:
                search_calls_used += 1
                safe_msg = redact_secrets(exc.message, secrets)
                tech_error_type = exc.error_type
                tech_error_msg = safe_msg
                search_rounds.append(
                    SearchRoundRecord(
                        round_no=round_no,
                        query=query,
                        result_count=0,
                        source_ids=[],
                        triggered_by_claim_id=state["follow_up_claim_id"] if round_no > 1 else None,
                        trigger_reason=state["follow_up_reason"] if round_no > 1 else None,
                        missing_info=state["follow_up_missing_info"] if round_no > 1 else None,
                        error_message=safe_msg,
                    )
                )
                break

        dur = int((time.monotonic() - t0) * 1000)
        had_tech_fail = tech_error_msg is not None
        if had_tech_fail and len(chunks) == 0:
            events = _append_event(
                events,
                "researcher",
                round_no,
                "error",
                started_at,
                dur,
                tech_error_msg or "Arama servisi teknik hatası",
                on_event=on_event,
            )
            return {
                "sources": sources,
                "chunks": chunks,
                "search_rounds": search_rounds,
                "executed_queries": executed_queries,
                "pending_queries": [],
                "search_calls_used": search_calls_used,
                "events": events,
                "technical_search_failure": True,
                "fatal_error_type": tech_error_type,
                "fatal_error_message": tech_error_msg,
            }

        summary = (
            f"Yeni benzersiz kaynak: {new_sources_count}, yeni kanıt parçası: {new_chunks_count}, toplam kaynak: {len(sources)}"
        )
        events = _append_event(
            events,
            "researcher",
            round_no,
            "finish",
            started_at,
            dur,
            "Kaynak toplama adımı tamamlandı." if lang == "tr" else "Source collection completed.",
            payload_summary=summary,
            on_event=on_event,
        )
        update: dict = {
            "sources": sources,
            "chunks": chunks,
            "search_rounds": search_rounds,
            "executed_queries": executed_queries,
            "pending_queries": [],
            "search_calls_used": search_calls_used,
            "events": events,
            "technical_search_failure": had_tech_fail,
        }
        if had_tech_fail and tech_error_msg:
            update["fatal_error_type"] = tech_error_type
            update["fatal_error_message"] = tech_error_msg
        return update

    def analyst_node(state: ResearchGraphState) -> dict:
        t0 = time.monotonic()
        started_at = _utc_now_iso()
        lang = state["language"]
        round_no = state["round_no"]
        events = _append_event(
            list(state["events"]),
            "analyst",
            round_no,
            "start",
            started_at,
            0,
            f"{round_no}. tur kanıt parçalarından tekil iddialar çıkarılıyor."
            if lang == "tr"
            else f"Extracting atomic claims for round {round_no}.",
            on_event=on_event,
        )

        ok, budget_msg = _budget_allows_model(state, config)
        if not ok:
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(events, "analyst", round_no, "error", started_at, dur, budget_msg, on_event=on_event)
            return {
                "events": events,
                "fatal_error_type": "budget_exhausted",
                "fatal_error_message": budget_msg,
            }

        try:
            analyst_out, usage = llm_provider.extract_claims(
                question=state["question"],
                language=lang,
                round_no=round_no,
                sub_questions=state["sub_questions"],
                sources=state["sources"],
                chunks=state["chunks"],
                existing_claims=state["claims"],
            )
            claims = list(state["claims"])
            existing_ids = {c.claim_id for c in claims}
            existing_stmts = {normalize_whitespace_nfc(c.statement).lower() for c in claims}
            valid_sq_ids = {sq.sq_id for sq in state["sub_questions"]}
            sources_by_id = {s.source_id: s for s in state["sources"]}
            chunks_by_id = {ch.chunk_id: ch for ch in state["chunks"]}

            added = 0
            for raw_claim in analyst_out.claims:
                if raw_claim.claim_id in existing_ids:
                    continue
                src = sources_by_id.get(raw_claim.initial_source_id)
                if src is None or not src.is_accessible:
                    continue
                ch = chunks_by_id.get(raw_claim.initial_chunk_id)
                if ch is None or ch.source_id != raw_claim.initial_source_id:
                    continue
                stmt_norm = normalize_whitespace_nfc(raw_claim.statement).lower()
                if stmt_norm in existing_stmts:
                    continue
                sq_id = raw_claim.sq_id if raw_claim.sq_id in valid_sq_ids else (
                    state["sub_questions"][0].sq_id if state["sub_questions"] else "SQ1"
                )
                claims.append(
                    ClaimRecord(
                        claim_id=raw_claim.claim_id,
                        sq_id=sq_id,
                        statement=raw_claim.statement,
                        initial_source_id=raw_claim.initial_source_id,
                        initial_chunk_id=raw_claim.initial_chunk_id,
                        initial_quote=normalize_whitespace_nfc(raw_claim.initial_quote),
                        initial_quote_language=raw_claim.initial_quote_language,
                        explanatory_note=raw_claim.explanatory_note,
                        round_created=round_no,
                    )
                )
                existing_ids.add(raw_claim.claim_id)
                existing_stmts.add(stmt_norm)
                added += 1

            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(
                events,
                "analyst",
                round_no,
                "finish",
                started_at,
                dur,
                "İddia çıkarımı tamamlandı." if lang == "tr" else "Claim extraction completed.",
                payload_summary=f"Eklenen yeni iddia: {added}, toplam iddia: {len(claims)}",
                on_event=on_event,
            )
            return {
                "claims": claims,
                "events": events,
                **_add_usage(state, usage),
            }
        except ProviderError as exc:
            safe_msg = redact_secrets(exc.message, secrets)
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(events, "analyst", round_no, "error", started_at, dur, safe_msg, on_event=on_event)
            return {
                "events": events,
                "fatal_error_type": exc.error_type,
                "fatal_error_message": safe_msg,
            }

    def verifier_node(state: ResearchGraphState) -> dict:
        t0 = time.monotonic()
        started_at = _utc_now_iso()
        lang = state["language"]
        round_no = state["round_no"]
        events = _append_event(
            list(state["events"]),
            "verifier",
            round_no,
            "start",
            started_at,
            0,
            f"{round_no}. tur kanıt denetimi yapılıyor."
            if lang == "tr"
            else f"Verifying claims against stored excerpts (round {round_no}).",
            on_event=on_event,
        )

        if not state["claims"]:
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(
                events,
                "verifier",
                round_no,
                "finish",
                started_at,
                dur,
                "Denetlenecek iddia bulunamadı." if lang == "tr" else "No claims to verify.",
                on_event=on_event,
            )
            return {"events": events}

        ok, budget_msg = _budget_allows_model(state, config)
        if not ok:
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(events, "verifier", round_no, "error", started_at, dur, budget_msg, on_event=on_event)
            return {
                "events": events,
                "fatal_error_type": "budget_exhausted",
                "fatal_error_message": budget_msg,
            }

        try:
            ver_out, usage = llm_provider.verify_claims(
                question=state["question"],
                language=lang,
                round_no=round_no,
                claims=state["claims"],
                sources=state["sources"],
                chunks=state["chunks"],
            )
            sources_by_id = {s.source_id: s for s in state["sources"]}
            chunks_by_id = {c.chunk_id: c for c in state["chunks"]}
            claims_by_id = {c.claim_id: c for c in state["claims"]}
            raw_by_claim = {v.claim_id: v for v in ver_out.verifications}

            prior_verifications = [v.model_copy() for v in state["verifications"]]
            new_verifications: list[VerificationRecord] = []

            for claim in state["claims"]:
                raw_v = raw_by_claim.get(claim.claim_id)
                if raw_v is None:
                    existing_final = next(
                        (pv for pv in prior_verifications if pv.claim_id == claim.claim_id and pv.is_final),
                        None,
                    )
                    if existing_final is not None:
                        continue
                    checked = VerificationRecord(
                        claim_id=claim.claim_id,
                        round_no=round_no,
                        verdict=VerdictStatus.INSUFFICIENT,
                        rationale=(
                            "Denetçi bu iddia için karar üretmedi; kanıt yetersiz sayıldı."
                            if lang == "tr"
                            else "Verifier produced no decision for this claim; marked insufficient."
                        ),
                        is_final=True,
                    )
                else:
                    for pv in prior_verifications:
                        if pv.claim_id == claim.claim_id and pv.is_final:
                            pv.is_final = False
                    checked = enforce_verification_guardrails(
                        claim=claims_by_id[claim.claim_id],
                        raw_verification=raw_v,
                        round_no=round_no,
                        sources_by_id=sources_by_id,
                        chunks_by_id=chunks_by_id,
                        language=lang,
                        research_question=state["question"],
                    )
                new_verifications.append(checked)

            all_verifications = [*prior_verifications, *new_verifications]
            summary_parts = [
                f"{v.claim_id}={verdict_label(v.verdict, lang)}"
                for v in all_verifications
                if v.is_final
            ]
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(
                events,
                "verifier",
                round_no,
                "finish",
                started_at,
                dur,
                "Kanıt denetimi tamamlandı." if lang == "tr" else "Claim verification completed.",
                payload_summary=", ".join(summary_parts),
                on_event=on_event,
            )
            return {
                "verifications": all_verifications,
                "events": events,
                **_add_usage(state, usage),
            }
        except ProviderError as exc:
            safe_msg = redact_secrets(exc.message, secrets)
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(events, "verifier", round_no, "error", started_at, dur, safe_msg, on_event=on_event)
            return {
                "events": events,
                "fatal_error_type": exc.error_type,
                "fatal_error_message": safe_msg,
            }

    def plan_followup_node(state: ResearchGraphState) -> dict:
        started_at = _utc_now_iso()
        lang = state["language"]
        candidate = _pick_followup_candidate(state)
        if candidate is None:
            return {}
        ver, query = candidate
        reason = (
            f"{ver.claim_id} iddiası için 1. turda '{verdict_label(ver.verdict, lang)}' kararı verildi; "
            f"eksik kanıt: {ver.missing_info or ver.rationale}"
            if lang == "tr"
            else f"Round 1 marked {ver.claim_id} as '{verdict_label(ver.verdict, lang)}'; "
            f"missing evidence: {ver.missing_info or ver.rationale}"
        )
        events = _append_event(
            list(state["events"]),
            "router",
            2,
            "decision",
            started_at,
            0,
            "Yönlendirici koşullu 2. arama turunu başlattı."
            if lang == "tr"
            else "Router triggered conditional Round 2 follow-up search.",
            payload_summary=f"{ver.claim_id} -> '{query}' ({reason})",
            on_event=on_event,
        )
        return {
            "round_no": 2,
            "pending_queries": [query],
            "follow_up_claim_id": ver.claim_id,
            "follow_up_reason": reason,
            "follow_up_missing_info": ver.missing_info,
            "events": events,
        }

    def reporter_node(state: ResearchGraphState) -> dict:
        t0 = time.monotonic()
        started_at = _utc_now_iso()
        lang = state["language"]
        is_retry = state.get("quality_retry_count", 0) > 0
        events = _append_event(
            list(state["events"]),
            "reporter",
            state["round_no"],
            "start",
            started_at,
            0,
            "Kalite geçidi geri bildirimi ile rapor düzeltiliyor."
            if is_retry and lang == "tr"
            else ("Son rapor ve kaynakça derleniyor." if lang == "tr" else "Compiling final report and bibliography."),
            on_event=on_event,
        )

        sources_by_id = {s.source_id: s for s in state["sources"]}
        claims_by_id = {c.claim_id: c for c in state["claims"]}
        final_vers = [v for v in state["verifications"] if v.is_final]
        round1_by_claim = {
            v.claim_id: v for v in state["verifications"] if v.round_no == 1
        }

        supported_findings: list[ReportFindingItem] = []
        conflicts: list[ConflictComparisonItem] = []
        limitations: list[LimitationItem] = []
        quantitative_comparisons: list[str] = []
        counter_evidence_summary: list[str] = []

        for ver in final_vers:
            claim = claims_by_id.get(ver.claim_id)
            if claim is None:
                continue
            if ver.verdict == VerdictStatus.SUPPORTED:
                cite_ids: list[str] = []
                for ev in ver.evidence_items:
                    if ev.source_id in sources_by_id and ev.source_id not in cite_ids:
                        cite_ids.append(ev.source_id)
                if not cite_ids:
                    limitations.append(
                        LimitationItem(
                            claim_id=claim.claim_id,
                            sq_id=claim.sq_id,
                            statement=claim.statement,
                            verdict=VerdictStatus.INSUFFICIENT,
                            rationale=ver.rationale,
                            missing_info=ver.missing_info,
                        )
                    )
                    continue
                supported_findings.append(
                    ReportFindingItem(
                        claim_id=claim.claim_id,
                        sq_id=claim.sq_id,
                        statement=claim.statement,
                        citation_ids=cite_ids,
                        quotes=ver.evidence_items,
                        rationale=ver.rationale,
                        causation_warning=ver.causation_warning,
                        numeric_context=ver.numeric_context,
                        single_source_warning=ver.single_source_warning,
                    )
                )
                if ver.numeric_context or re.search(
                    r"\d", re.sub(r"\b(?:19|20)\d{2}\b", "", claim.statement)
                ):
                    cites = " ".join(f"[{sid}]" for sid in cite_ids)
                    ctx = f" — Bağlam: {ver.numeric_context}" if ver.numeric_context else ""
                    quantitative_comparisons.append(
                        f"**{claim.claim_id}** {cites}: {claim.statement}{ctx}"
                    )
            elif ver.verdict == VerdictStatus.CONFLICTING and len(ver.evidence_items) >= 2:
                conflicts.append(
                    ConflictComparisonItem(
                        claim_id=claim.claim_id,
                        sq_id=claim.sq_id,
                        statement=claim.statement,
                        explanation=ver.rationale,
                        side_a=ver.evidence_items[0],
                        side_b=ver.evidence_items[1],
                    )
                )
                counter_evidence_summary.append(
                    f"{claim.claim_id} ([{ver.evidence_items[0].source_id}] vs [{ver.evidence_items[1].source_id}]): {ver.rationale}"
                )
            else:
                r1 = round1_by_claim.get(claim.claim_id)
                init_v = r1.verdict if (r1 and r1.round_no < ver.round_no) else None
                init_r = r1.rationale if (r1 and r1.round_no < ver.round_no) else None
                limitations.append(
                    LimitationItem(
                        claim_id=claim.claim_id,
                        sq_id=claim.sq_id,
                        statement=claim.statement,
                        verdict=ver.verdict,
                        rationale=ver.rationale,
                        missing_info=ver.missing_info,
                        initial_verdict=init_v,
                        initial_rationale=init_r,
                        counter_evidence=ver.evidence_items,
                    )
                )
                if ver.verdict == VerdictStatus.CONTRADICTED:
                    c_sids = " ".join(f"[{e.source_id}]" for e in ver.evidence_items)
                    counter_evidence_summary.append(
                        f"{claim.claim_id} ({c_sids}): {ver.rationale}"
                    )

        referenced_sids = (
            {sid for f in supported_findings for sid in f.citation_ids}
            | {c.side_a.source_id for c in conflicts}
            | {c.side_b.source_id for c in conflicts}
            | {ev.source_id for lim in limitations for ev in lim.counter_evidence}
        )
        if referenced_sids:
            bib_sources = [
                s
                for s in state["sources"]
                if s.is_mock
                or (
                    s.source_id in referenced_sids
                    and not (
                        s.duplicate_of_source_id is not None
                        and s.duplicate_of_source_id in referenced_sids
                    )
                )
            ]
        else:
            bib_sources = [
                s
                for s in state["sources"]
                if s.is_mock
                or (s.is_accessible and s.relevance_score >= 0.2 and s.duplicate_of_source_id is None)
            ]
            bib_sources.sort(key=lambda s: (-s.relevance_score, -s.credibility_score))
            bib_sources = bib_sources[:3]

        bibliography = [
            BibliographyItem(
                source_id=s.source_id,
                title=s.title,
                publisher=s.publisher,
                author=s.author,
                published_date=s.published_date,
                retrieved_at=s.retrieved_at,
                url=s.url,
                language=s.language,
                provider=s.provider,
                is_mock=s.is_mock,
                evidence_level=s.evidence_level,
                source_type=s.source_type,
                retrieval_method=s.retrieval_method,
                relevance_assessment=s.relevance_assessment,
                credibility_score=s.credibility_score,
                is_accessible=s.is_accessible,
                duplicate_of_source_id=s.duplicate_of_source_id,
            )
            for s in bib_sources
        ]

        updated_sqs = [sq.model_copy() for sq in state["sub_questions"]]
        open_questions: list[str] = []
        clean_q = clean_question_title(state["question"])
        title = clean_q
        short_answer = ""
        overall_assessment = ""
        usage_delta = UsageDelta(model_calls=0)

        fatal_msg = state["fatal_error_message"]
        fatal_type = state["fatal_error_type"]
        has_usable_evidence = len(state["chunks"]) > 0

        if state["technical_search_failure"] and not has_usable_evidence:
            final_status = RunStatus.FAILED
            short_answer = f"{t('technical_error_prefix', lang)}: {fatal_msg}"
            open_questions = [sq.text for sq in updated_sqs] or [clean_q]
            overall_assessment = short_answer
        elif fatal_msg and not supported_findings and not conflicts and not has_usable_evidence:
            final_status = RunStatus.FAILED
            short_answer = f"{t('technical_error_prefix', lang)}: {fatal_msg}"
            open_questions = [sq.text for sq in updated_sqs] or [clean_q]
            overall_assessment = short_answer
        else:
            ok_budget, _ = _budget_allows_model(state, config)
            can_call_llm = ok_budget and (fatal_msg is None or has_usable_evidence) and not is_retry
            if can_call_llm and (len(state["chunks"]) > 0 or state["mode"] == "demo"):
                try:
                    syn_out, usage_delta = llm_provider.synthesize_report(
                        question=state["question"],
                        language=lang,
                        sub_questions=updated_sqs,
                        claims=state["claims"],
                        verifications=state["verifications"],
                        sources=state["sources"],
                    )
                    title = clean_question_title(syn_out.title) if syn_out.title else title
                    short_answer = syn_out.short_answer
                    overall_assessment = syn_out.overall_assessment
                    sq_status_map = {u.sq_id: u.status for u in syn_out.sub_question_updates}
                    for sq in updated_sqs:
                        if sq.sq_id in sq_status_map:
                            sq.status = sq_status_map[sq.sq_id]
                    open_questions = list(syn_out.open_questions)
                except ProviderError as exc:
                    fatal_type = fatal_type or exc.error_type
                    fatal_msg = fatal_msg or redact_secrets(exc.message, secrets)

            # Kalite geçidi düzeltmesi veya deterministik sentez yedeği
            if is_retry or not short_answer:
                if len(state["sources"]) == 0:
                    short_answer = (
                        f"{t('empty_search_notice', lang)} "
                        + (
                            "Destekli bir bulguya ulaşılamadı."
                            if lang == "tr"
                            else "No supported findings could be verified."
                        )
                    )
                    overall_assessment = short_answer
                elif conflicts:
                    c0 = conflicts[0]
                    short_answer = (
                        f"İki bağımsız kaynak çeliştiği için kesin hüküm verilmemiştir: [{c0.side_a.source_id}] ve [{c0.side_b.source_id}] farklı sonuçlar bildirmektedir."
                        if lang == "tr"
                        else f"No definitive conclusion due to conflicting independent sources [{c0.side_a.source_id}] and [{c0.side_b.source_id}]."
                    )
                    overall_assessment = "Kaynaklar arasındaki yöntem veya ölçüm farkı giderilmeden tekil hüküm kurulamaz."
                else:
                    short_answer, overall_assessment = _build_academic_synthesis(
                        question=state["question"],
                        language=lang,
                        sub_questions=updated_sqs,
                        claims=state["claims"],
                        verifications=state["verifications"],
                        sources=state["sources"],
                    )

            # Alt soruların findings, remaining_gaps ve durumlarını kanıta göre kesinleştir
            for sq in updated_sqs:
                sq_sup = [f for f in supported_findings if f.sq_id == sq.sq_id]
                sq_conf = [c for c in conflicts if c.sq_id == sq.sq_id]
                sq_lim = [lim for lim in limitations if lim.sq_id == sq.sq_id]
                sq.findings = [f"{f.statement} {' '.join(f'[{s}]' for s in f.citation_ids)}" for f in sq_sup]
                sq.remaining_gaps = [
                    lim.missing_info or lim.rationale for lim in sq_lim
                ]
                has_unanswered_gap = any(
                    lim.verdict in (VerdictStatus.INSUFFICIENT, VerdictStatus.IRRELEVANT)
                    for lim in sq_lim
                )
                has_contra = any(lim.verdict == VerdictStatus.CONTRADICTED for lim in sq_lim)

                if sq_conf:
                    sq.status = SubQuestionStatus.CONFLICTING
                elif not sq_sup and not has_contra:
                    # Kanıt yetersiz veya hiç iddia yoksa ASLA 'Yanıtlandı' olamaz
                    if sq.status != SubQuestionStatus.ANSWERED or has_unanswered_gap or not state["claims"]:
                        sq.status = SubQuestionStatus.UNANSWERED
                elif sq_sup and has_unanswered_gap:
                    sq.status = SubQuestionStatus.PARTIALLY_ANSWERED
                elif sq_sup or has_contra:
                    if sq.status == SubQuestionStatus.UNANSWERED:
                        sq.status = SubQuestionStatus.ANSWERED

            # Açık soruları ve kısa yanıt atıflarını tutarlı kıl
            open_questions = [
                sq.text
                for sq in updated_sqs
                if sq.status in (SubQuestionStatus.UNANSWERED, SubQuestionStatus.CONFLICTING)
            ]
            if not open_questions and len(state["sources"]) == 0:
                open_questions = [clean_q]

            # Geçersiz [S#] atıflarını kısa yanıttan temizle
            def _clean_cite(match: re.Match) -> str:
                sid = match.group(1)
                return f"[{sid}]" if sid in sources_by_id else ""

            short_answer = _deduplicate_sentences(_CITATION_TAG_RE.sub(_clean_cite, short_answer))
            if not overall_assessment:
                _, overall_assessment = _build_academic_synthesis(
                    question=state["question"],
                    language=lang,
                    sub_questions=updated_sqs,
                    claims=state["claims"],
                    verifications=state["verifications"],
                    sources=state["sources"],
                )

            if fatal_msg:
                final_status = RunStatus.PARTIAL
            elif len(state["sources"]) == 0 and state["technical_search_failure"]:
                final_status = RunStatus.FAILED
            else:
                final_status = RunStatus.COMPLETED

        method_scope = (
            f"Derinlik: {state['depth']} · Taranan kaynak: {len(state['sources'])} · "
            f"Arama turu: {max((sr.round_no for sr in state['search_rounds']), default=1)}. "
            f"{state['scope_notes']}"
        ).strip()

        report = FinalReport(
            title=title,
            language=RunLanguage(lang),
            short_answer=short_answer,
            research_purpose=state.get("research_purpose") or state["plan_summary"] or clean_q,
            methodology_and_scope=method_scope,
            sub_questions=updated_sqs,
            supported_findings=supported_findings,
            quantitative_comparisons=quantitative_comparisons,
            conflicts=conflicts,
            counter_evidence_summary=counter_evidence_summary,
            limitations=limitations,
            overall_assessment=overall_assessment,
            open_questions=open_questions,
            bibliography=bibliography,
            error_notice=fatal_msg,
        )

        dur_total = int((time.monotonic() - state["started_monotonic"]) * 1000)
        usage_update = _add_usage(state, usage_delta)
        preview_meta = RunMetadata(
            run_id=state["run_id"],
            idempotency_key=state["idempotency_key"],
            question=state["question"],
            language=RunLanguage(lang),
            depth=ResearchDepth(state["depth"]),
            mode=RunMode(state["mode"]),
            model_name=state["model_name"],
            started_at=state["started_at"],
            finished_at=_utc_now_iso(),
            duration_ms=dur_total,
            status=final_status,
            plan_summary=state["plan_summary"],
            scope_notes=state["scope_notes"],
            research_purpose=state.get("research_purpose", ""),
            timeframe=state.get("timeframe", ""),
            key_concepts=state.get("key_concepts", []),
            model_calls=usage_update["model_calls_used"],
            search_calls=state["search_calls_used"],
            prompt_tokens=usage_update["prompt_tokens"],
            completion_tokens=usage_update["completion_tokens"],
            total_tokens=usage_update["total_tokens"],
            token_measurement_type=usage_update["token_measurement_type"],
            error_type=fatal_type,
            error_message=fatal_msg,
        )
        preview_bundle = RunBundle(
            metadata=preview_meta,
            sub_questions=updated_sqs,
            search_rounds=state["search_rounds"],
            sources=state["sources"],
            chunks=state["chunks"],
            claims=state["claims"],
            verifications=state["verifications"],
            events=events,
            report=report,
        )
        from p04_research.export.markdown_export import render_report_markdown

        report.report_markdown = render_report_markdown(preview_bundle)

        dur_node = int((time.monotonic() - t0) * 1000)
        events = _append_event(
            events,
            "reporter",
            state["round_no"],
            "finish" if final_status != RunStatus.FAILED else "error",
            started_at,
            dur_node,
            "Son rapor derlendi." if lang == "tr" else "Final report compiled.",
            payload_summary=f"Durum: {final_status.value}, Destekli: {len(supported_findings)}, Çelişki: {len(conflicts)}, Sınırlılık: {len(limitations)}",
            on_event=on_event,
        )

        return {
            "sub_questions": updated_sqs,
            "report": report,
            "status": final_status.value,
            "fatal_error_type": fatal_type,
            "fatal_error_message": fatal_msg,
            "events": events,
            **usage_update,
        }

    def quality_gate_node(state: ResearchGraphState) -> dict:
        """Raportör çıktısını 10 kalite ölçütüne göre denetler ve teknik başarıdan ayrı puanlar."""
        t0 = time.monotonic()
        started_at = _utc_now_iso()
        lang = state["language"]
        rep = state["report"]
        events = list(state["events"])

        if rep is None or state["status"] == RunStatus.FAILED.value:
            events = _append_event(
                events,
                "quality_gate",
                state["round_no"],
                "finish",
                started_at,
                int((time.monotonic() - t0) * 1000),
                "Teknik hata nedeniyle kalite geçidi atlandı.",
                on_event=on_event,
            )
            return {
                "quality_passed": False,
                "research_quality": "insufficient_evidence",
                "quality_feedback": [],
                "events": events,
            }

        sources_by_id = {s.source_id: s for s in state["sources"]}
        final_vers = {v.claim_id: v for v in state["verifications"] if v.is_final}
        defects: list[str] = []

        # 1. Başlık kesme kontrolü
        if rep.title.rstrip().endswith("..."):
            defects.append("Rapor başlığı üç nokta (...) ile kesilmiş.")

        # 2. Kısa yanıt boş / çelişkili / geçersiz atıflı olmamalı
        if not rep.short_answer.strip():
            defects.append("Yönetici özeti / kısa yanıt boş.")
        for ref_sid in _CITATION_TAG_RE.findall(rep.short_answer):
            if ref_sid not in sources_by_id:
                defects.append(f"Kısa yanıtta geçersiz kaynak kimliği [{ref_sid}] bulundu.")

        # 3. Destekli bulgularda ilgisiz veya geçersiz kaynaklı iddia olmamalı
        for f in rep.supported_findings:
            if not f.citation_ids or any(sid not in sources_by_id for sid in f.citation_ids):
                defects.append(f"{f.claim_id} bulgusunun geçerli kaynak atıfı eksik.")
            v = final_vers.get(f.claim_id)
            if v is not None and v.verdict != VerdictStatus.SUPPORTED:
                defects.append(f"{f.claim_id} kesin destekli olmadığı halde bulgularda yer alıyor.")

        # 4. Çelişkili kaynaklar varken özet tek yanlı hüküm vermemeli
        if rep.conflicts and "çeliş" not in rep.short_answer.lower() and "conflict" not in rep.short_answer.lower():
            defects.append("Çelişen kaynaklar kısa yanıtta dengeli biçimde belirtilmedi.")

        # 5. Açık sorular / alt soru durumu tutarlılığı
        unanswered_texts = {sq.text for sq in rep.sub_questions if sq.status == SubQuestionStatus.UNANSWERED}
        if unanswered_texts and not any(t in rep.open_questions for t in unanswered_texts):
            defects.append("Yanıtsız alt sorular Açık Sorular listesine tam aktarılmadı.")

        # 6. Kısa yanıtta tekrar cümle kontrolü
        raw_sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", rep.short_answer) if s.strip()]
        if len(raw_sents) != len(set(raw_sents)):
            defects.append("Kısa yanıt içinde tekrarlanan cümleler tespit edildi.")

        retry_count = state.get("quality_retry_count", 0)
        if defects and retry_count < config.max_quality_retries:
            dur = int((time.monotonic() - t0) * 1000)
            events = _append_event(
                events,
                "quality_gate",
                state["round_no"],
                "decision",
                started_at,
                dur,
                "Kalite geçidi düzeltilebilir eksikler tespit etti; raportöre geri gönderiliyor.",
                payload_summary="; ".join(defects),
                on_event=on_event,
            )
            return {
                "quality_retry_count": retry_count + 1,
                "quality_feedback": defects,
                "quality_passed": False,
                "events": events,
            }

        if defects:
            research_quality = "needs_improvement"
            quality_passed = False
        elif not rep.supported_findings and not rep.conflicts:
            research_quality = "insufficient_evidence"
            quality_passed = True
        else:
            research_quality = "high"
            quality_passed = True

        rep.research_quality = research_quality
        rep.quality_notes = defects
        from p04_research.export.markdown_export import render_report_markdown

        meta_copy = RunMetadata(
            run_id=state["run_id"],
            idempotency_key=state["idempotency_key"],
            question=state["question"],
            language=RunLanguage(lang),
            depth=ResearchDepth(state["depth"]),
            mode=RunMode(state["mode"]),
            model_name=state["model_name"],
            started_at=state["started_at"],
            finished_at=_utc_now_iso(),
            duration_ms=int((time.monotonic() - state["started_monotonic"]) * 1000),
            status=RunStatus(state["status"]),
            plan_summary=state["plan_summary"],
            scope_notes=state["scope_notes"],
            research_purpose=state.get("research_purpose", ""),
            timeframe=state.get("timeframe", ""),
            key_concepts=state.get("key_concepts", []),
            model_calls=state["model_calls_used"],
            search_calls=state["search_calls_used"],
            prompt_tokens=state["prompt_tokens"],
            completion_tokens=state["completion_tokens"],
            total_tokens=state["total_tokens"],
            token_measurement_type=state.get("token_measurement_type", "unmeasured"),
            research_quality=research_quality,
            error_type=state["fatal_error_type"],
            error_message=state["fatal_error_message"],
        )
        preview_bundle = RunBundle(
            metadata=meta_copy,
            sub_questions=rep.sub_questions,
            search_rounds=state["search_rounds"],
            sources=state["sources"],
            chunks=state["chunks"],
            claims=state["claims"],
            verifications=state["verifications"],
            events=events,
            report=rep,
        )
        rep.report_markdown = render_report_markdown(preview_bundle)

        dur = int((time.monotonic() - t0) * 1000)
        events = _append_event(
            events,
            "quality_gate",
            state["round_no"],
            "finish",
            started_at,
            dur,
            f"Kalite geçidi tamamlandı: {research_quality}",
            payload_summary=f"Geçti: {quality_passed}, Notlar: {len(defects)}",
            on_event=on_event,
        )
        return {
            "report": rep,
            "quality_passed": quality_passed,
            "research_quality": research_quality,
            "quality_feedback": defects,
            "events": events,
        }

    return {
        "planner": planner_node,
        "researcher": researcher_node,
        "analyst": analyst_node,
        "verifier": verifier_node,
        "plan_followup": plan_followup_node,
        "reporter": reporter_node,
        "quality_gate": quality_gate_node,
    }
