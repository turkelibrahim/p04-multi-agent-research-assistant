"""Atomik SQLite veri deposu: kayıt, idempotency koruması, geçmiş listeleme ve kanıt zinciri."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from p04_research.config import redact_secrets
from p04_research.db.schema import get_connection, init_db
from p04_research.models import (
    ClaimProvenanceView,
    ClaimRecord,
    EvidenceChunk,
    EvidenceLevel,
    EvidenceRole,
    FinalReport,
    ResearchDepth,
    RunBundle,
    RunEvent,
    RunLanguage,
    RunMetadata,
    RunMode,
    RunStatus,
    SearchRoundRecord,
    SourceRecord,
    SourceType,
    SubQuestion,
    SubQuestionStatus,
    VerdictEvidenceItem,
    VerdictStatus,
    VerificationRecord,
)


class ResearchRepository:
    """Araştırma koşumlarını, kanıt zincirini ve onaylı raporları SQLite üzerinde saklar."""

    def __init__(self, db_path: Path, known_secrets: Iterable[str] = ()) -> None:
        self.db_path = db_path
        self.known_secrets = tuple(known_secrets)
        self._initialized = False

    def ensure_initialized(self) -> None:
        if not self._initialized:
            init_db(self.db_path)
            self._initialized = True

    def _scrub(self, text: str | None) -> str | None:
        if text is None:
            return None
        return redact_secrets(text, self.known_secrets)

    def get_run_id_by_idempotency_key(self, idempotency_key: str) -> str | None:
        self.ensure_initialized()
        with closing(get_connection(self.db_path)) as conn:
            row = conn.execute(
                "SELECT run_id FROM research_runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            return str(row["run_id"]) if row else None

    def save_run_bundle(self, bundle: RunBundle) -> tuple[bool, str | None]:
        """Tüm çalışma verisini tek atomik işlemde kaydeder; yarım yazmaya izin vermez."""
        try:
            self.ensure_initialized()
            with closing(get_connection(self.db_path)) as conn:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    existing = conn.execute(
                        "SELECT run_id FROM research_runs WHERE idempotency_key = ?",
                        (bundle.metadata.idempotency_key,),
                    ).fetchone()
                    if existing is not None:
                        conn.rollback()
                        return True, None

                    m = bundle.metadata
                    conn.execute(
                        """
                        INSERT INTO research_runs (
                            run_id, idempotency_key, question, language, depth, mode,
                            model_name, started_at, finished_at, duration_ms, status,
                            plan_summary, scope_notes, research_purpose, timeframe,
                            key_concepts_json, model_calls, search_calls,
                            prompt_tokens, completion_tokens, total_tokens,
                            token_measurement_type, estimated_cost_usd, research_quality,
                            error_type, error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            m.run_id,
                            m.idempotency_key,
                            self._scrub(m.question),
                            m.language.value,
                            m.depth.value,
                            m.mode.value,
                            m.model_name,
                            m.started_at,
                            m.finished_at,
                            m.duration_ms,
                            m.status.value,
                            self._scrub(m.plan_summary),
                            self._scrub(m.scope_notes),
                            self._scrub(m.research_purpose),
                            self._scrub(m.timeframe),
                            json.dumps(m.key_concepts, ensure_ascii=False),
                            m.model_calls,
                            m.search_calls,
                            m.prompt_tokens,
                            m.completion_tokens,
                            m.total_tokens,
                            m.token_measurement_type,
                            m.estimated_cost_usd,
                            m.research_quality,
                            m.error_type,
                            self._scrub(m.error_message),
                        ),
                    )

                    for sq in bundle.sub_questions:
                        conn.execute(
                            """
                            INSERT INTO sub_questions (
                                run_id, sq_id, order_index, text, scope_note,
                                search_intent, required_evidence, status,
                                findings_json, remaining_gaps_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m.run_id,
                                sq.sq_id,
                                sq.order_index,
                                self._scrub(sq.text),
                                self._scrub(sq.scope_note),
                                self._scrub(sq.search_intent),
                                self._scrub(sq.required_evidence),
                                sq.status.value,
                                json.dumps(sq.findings, ensure_ascii=False),
                                json.dumps(sq.remaining_gaps, ensure_ascii=False),
                            ),
                        )

                    for sr in bundle.search_rounds:
                        conn.execute(
                            """
                            INSERT INTO search_rounds (
                                run_id, round_no, query, result_count, source_ids_json,
                                triggered_by_claim_id, trigger_reason, missing_info, error_message
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m.run_id,
                                sr.round_no,
                                self._scrub(sr.query),
                                sr.result_count,
                                json.dumps(sr.source_ids, ensure_ascii=False),
                                sr.triggered_by_claim_id,
                                self._scrub(sr.trigger_reason),
                                self._scrub(sr.missing_info),
                                self._scrub(sr.error_message),
                            ),
                        )

                    # Önce kopya bağı olmayan kaynakları, sonra kopyaları ekle (FK güvenliği)
                    ordered_sources = sorted(
                        bundle.sources, key=lambda s: (1 if s.duplicate_of_source_id else 0, s.source_id)
                    )
                    for src in ordered_sources:
                        conn.execute(
                            """
                            INSERT INTO sources (
                                run_id, source_id, url, normalized_url, title, publisher,
                                author, published_date, retrieved_at, language, provider, is_mock,
                                evidence_level, source_type, retrieval_method, retrieved_content,
                                relevance_score, relevance_assessment, credibility_score, is_accessible,
                                content_fingerprint, duplicate_of_source_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m.run_id,
                                src.source_id,
                                src.url,
                                src.normalized_url,
                                self._scrub(src.title),
                                self._scrub(src.publisher),
                                self._scrub(src.author),
                                src.published_date,
                                src.retrieved_at,
                                src.language,
                                src.provider,
                                1 if src.is_mock else 0,
                                src.evidence_level.value,
                                src.source_type.value,
                                src.retrieval_method,
                                self._scrub(src.retrieved_content),
                                src.relevance_score,
                                self._scrub(src.relevance_assessment),
                                src.credibility_score,
                                1 if src.is_accessible else 0,
                                src.content_fingerprint,
                                src.duplicate_of_source_id,
                            ),
                        )

                    for chunk in bundle.chunks:
                        conn.execute(
                            """
                            INSERT INTO evidence_chunks (
                                run_id, chunk_id, source_id, round_no, excerpt_text,
                                char_count, is_truncated, query_used
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m.run_id,
                                chunk.chunk_id,
                                chunk.source_id,
                                chunk.round_no,
                                self._scrub(chunk.excerpt_text),
                                chunk.char_count,
                                1 if chunk.is_truncated else 0,
                                self._scrub(chunk.query_used),
                            ),
                        )

                    for claim in bundle.claims:
                        conn.execute(
                            """
                            INSERT INTO claims (
                                run_id, claim_id, sq_id, statement, initial_source_id,
                                initial_chunk_id, initial_quote, initial_quote_language,
                                explanatory_note, round_created
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m.run_id,
                                claim.claim_id,
                                claim.sq_id,
                                self._scrub(claim.statement),
                                claim.initial_source_id,
                                claim.initial_chunk_id,
                                self._scrub(claim.initial_quote),
                                claim.initial_quote_language,
                                self._scrub(claim.explanatory_note),
                                claim.round_created,
                            ),
                        )

                    for ver in bundle.verifications:
                        cur = conn.execute(
                            """
                            INSERT INTO verifications (
                                run_id, claim_id, round_no, verdict, rationale,
                                missing_info, follow_up_query, is_final,
                                causation_warning, guardrail_notes_json,
                                claim_nature, numeric_context, independent_support_count,
                                single_source_warning
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m.run_id,
                                ver.claim_id,
                                ver.round_no,
                                ver.verdict.value,
                                self._scrub(ver.rationale),
                                self._scrub(ver.missing_info),
                                self._scrub(ver.follow_up_query),
                                1 if ver.is_final else 0,
                                self._scrub(ver.causation_warning),
                                json.dumps(ver.guardrail_notes, ensure_ascii=False),
                                ver.claim_nature,
                                self._scrub(ver.numeric_context),
                                ver.independent_support_count,
                                self._scrub(ver.single_source_warning),
                            ),
                        )
                        ver_id = cur.lastrowid
                        for ev in ver.evidence_items:
                            conn.execute(
                                """
                                INSERT INTO verification_evidence (
                                    verification_id, run_id, source_id, chunk_id,
                                    quote, role, quote_language, explanatory_note
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    ver_id,
                                    m.run_id,
                                    ev.source_id,
                                    ev.chunk_id,
                                    self._scrub(ev.quote),
                                    ev.role.value,
                                    ev.quote_language,
                                    self._scrub(ev.explanatory_note),
                                ),
                            )

                    if bundle.report is not None:
                        used_sids = [b.source_id for b in bundle.report.bibliography]
                        conn.execute(
                            """
                            INSERT INTO reports (
                                run_id, language, title, short_answer,
                                report_markdown, report_json, used_source_ids_json, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m.run_id,
                                bundle.report.language.value,
                                self._scrub(bundle.report.title),
                                self._scrub(bundle.report.short_answer),
                                self._scrub(bundle.report.report_markdown),
                                self._scrub(bundle.report.model_dump_json()),
                                json.dumps(used_sids, ensure_ascii=False),
                                m.finished_at or datetime.now(UTC).isoformat(),
                            ),
                        )

                    for ev in bundle.events:
                        conn.execute(
                            """
                            INSERT INTO run_events (
                                run_id, seq, node_name, round_no, status,
                                started_at, finished_at, duration_ms, message, payload_summary
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m.run_id,
                                ev.seq,
                                ev.node_name,
                                ev.round_no,
                                ev.status,
                                ev.started_at,
                                ev.finished_at,
                                ev.duration_ms,
                                self._scrub(ev.message),
                                self._scrub(ev.payload_summary),
                            ),
                        )

                    conn.commit()
                    return True, None
                except Exception as exc:
                    conn.rollback()
                    return False, f"Veritabanı yazma hatası (geri alındı): {exc}"
        except (sqlite3.Error, OSError) as exc:
            return False, f"Veritabanı bağlantısı açılamadı: {exc}"

    def list_runs(self, limit: int = 100) -> list[dict]:
        """En son yapılan çalışmaları tarih sırasıyla (en çok 100) döndürür."""
        self.ensure_initialized()
        with closing(get_connection(self.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT
                    r.run_id,
                    r.question,
                    r.language,
                    r.depth,
                    r.mode,
                    r.model_name,
                    r.started_at,
                    r.finished_at,
                    r.duration_ms,
                    r.status,
                    r.model_calls,
                    r.search_calls,
                    (SELECT COUNT(*) FROM sources s WHERE s.run_id = r.run_id) AS source_count,
                    (SELECT COUNT(*) FROM claims c WHERE c.run_id = r.run_id) AS total_claim_count,
                    (
                        SELECT COUNT(*) FROM verifications v
                        WHERE v.run_id = r.run_id AND v.is_final = 1 AND v.verdict = 'supported'
                    ) AS supported_claim_count
                FROM research_runs r
                ORDER BY r.started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def load_run_bundle(self, run_id: str) -> RunBundle | None:
        """Kaydedilmiş çalışmayı tüm turları, kaynakları ve raporuyla birlikte yeniden yükler."""
        self.ensure_initialized()
        with closing(get_connection(self.db_path)) as conn:
            run_row = conn.execute(
                "SELECT * FROM research_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                return None

            metadata = RunMetadata(
                run_id=run_row["run_id"],
                idempotency_key=run_row["idempotency_key"],
                question=run_row["question"],
                language=RunLanguage(run_row["language"]),
                depth=ResearchDepth(run_row["depth"]),
                mode=RunMode(run_row["mode"]),
                model_name=run_row["model_name"],
                started_at=run_row["started_at"],
                finished_at=run_row["finished_at"],
                duration_ms=run_row["duration_ms"],
                status=RunStatus(run_row["status"]),
                plan_summary=run_row["plan_summary"],
                scope_notes=run_row["scope_notes"],
                research_purpose=run_row["research_purpose"] if "research_purpose" in run_row.keys() else "",
                timeframe=run_row["timeframe"] if "timeframe" in run_row.keys() else "",
                key_concepts=json.loads(run_row["key_concepts_json"]) if "key_concepts_json" in run_row.keys() else [],
                model_calls=run_row["model_calls"],
                search_calls=run_row["search_calls"],
                prompt_tokens=run_row["prompt_tokens"],
                completion_tokens=run_row["completion_tokens"],
                total_tokens=run_row["total_tokens"],
                token_measurement_type=run_row["token_measurement_type"] if "token_measurement_type" in run_row.keys() else "unmeasured",
                estimated_cost_usd=run_row["estimated_cost_usd"] if "estimated_cost_usd" in run_row.keys() else None,
                research_quality=run_row["research_quality"] if "research_quality" in run_row.keys() else "high",
                error_type=run_row["error_type"],
                error_message=run_row["error_message"],
            )

            sq_rows = conn.execute(
                "SELECT * FROM sub_questions WHERE run_id = ? ORDER BY order_index ASC",
                (run_id,),
            ).fetchall()
            sub_questions = [
                SubQuestion(
                    sq_id=r["sq_id"],
                    order_index=r["order_index"],
                    text=r["text"],
                    scope_note=r["scope_note"],
                    search_intent=r["search_intent"] if "search_intent" in r.keys() else "",
                    required_evidence=r["required_evidence"] if "required_evidence" in r.keys() else "",
                    status=SubQuestionStatus(r["status"]),
                    findings=json.loads(r["findings_json"]) if "findings_json" in r.keys() else [],
                    remaining_gaps=json.loads(r["remaining_gaps_json"]) if "remaining_gaps_json" in r.keys() else [],
                )
                for r in sq_rows
            ]

            sr_rows = conn.execute(
                "SELECT * FROM search_rounds WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
            search_rounds = [
                SearchRoundRecord(
                    round_no=r["round_no"],
                    query=r["query"],
                    result_count=r["result_count"],
                    source_ids=json.loads(r["source_ids_json"]),
                    triggered_by_claim_id=r["triggered_by_claim_id"],
                    trigger_reason=r["trigger_reason"],
                    missing_info=r["missing_info"],
                    error_message=r["error_message"],
                )
                for r in sr_rows
            ]

            src_rows = conn.execute(
                "SELECT * FROM sources WHERE run_id = ? ORDER BY source_id ASC",
                (run_id,),
            ).fetchall()
            sources = [
                SourceRecord(
                    source_id=r["source_id"],
                    url=r["url"],
                    normalized_url=r["normalized_url"],
                    title=r["title"],
                    publisher=r["publisher"],
                    author=r["author"] if "author" in r.keys() else None,
                    published_date=r["published_date"],
                    retrieved_at=r["retrieved_at"],
                    language=r["language"],
                    provider=r["provider"],
                    is_mock=bool(r["is_mock"]),
                    evidence_level=EvidenceLevel(r["evidence_level"]),
                    source_type=SourceType(r["source_type"]) if "source_type" in r.keys() else SourceType.TRUSTED_SECONDARY_ANALYSIS,
                    retrieval_method=r["retrieval_method"] if "retrieval_method" in r.keys() else "search_snippet",
                    retrieved_content=r["retrieved_content"] if "retrieved_content" in r.keys() else "",
                    relevance_score=r["relevance_score"] if "relevance_score" in r.keys() else 1.0,
                    relevance_assessment=r["relevance_assessment"] if "relevance_assessment" in r.keys() else "",
                    credibility_score=r["credibility_score"] if "credibility_score" in r.keys() else 0.7,
                    is_accessible=bool(r["is_accessible"]) if "is_accessible" in r.keys() else True,
                    content_fingerprint=r["content_fingerprint"],
                    duplicate_of_source_id=r["duplicate_of_source_id"],
                )
                for r in src_rows
            ]

            chunk_rows = conn.execute(
                "SELECT * FROM evidence_chunks WHERE run_id = ? ORDER BY round_no ASC, chunk_id ASC",
                (run_id,),
            ).fetchall()
            chunks = [
                EvidenceChunk(
                    chunk_id=r["chunk_id"],
                    source_id=r["source_id"],
                    round_no=r["round_no"],
                    excerpt_text=r["excerpt_text"],
                    char_count=r["char_count"],
                    is_truncated=bool(r["is_truncated"]),
                    query_used=r["query_used"],
                )
                for r in chunk_rows
            ]

            claim_rows = conn.execute(
                "SELECT * FROM claims WHERE run_id = ? ORDER BY claim_id ASC",
                (run_id,),
            ).fetchall()
            claims = [
                ClaimRecord(
                    claim_id=r["claim_id"],
                    sq_id=r["sq_id"],
                    statement=r["statement"],
                    initial_source_id=r["initial_source_id"],
                    initial_chunk_id=r["initial_chunk_id"],
                    initial_quote=r["initial_quote"],
                    initial_quote_language=r["initial_quote_language"],
                    explanatory_note=r["explanatory_note"],
                    round_created=r["round_created"],
                )
                for r in claim_rows
            ]

            ver_rows = conn.execute(
                "SELECT * FROM verifications WHERE run_id = ? ORDER BY round_no ASC, claim_id ASC",
                (run_id,),
            ).fetchall()
            verifications: list[VerificationRecord] = []
            for vr in ver_rows:
                ev_rows = conn.execute(
                    "SELECT * FROM verification_evidence WHERE verification_id = ? AND run_id = ? ORDER BY id ASC",
                    (vr["id"], run_id),
                ).fetchall()
                ev_items = [
                    VerdictEvidenceItem(
                        source_id=er["source_id"],
                        chunk_id=er["chunk_id"],
                        quote=er["quote"],
                        role=EvidenceRole(er["role"]),
                        quote_language=er["quote_language"],
                        explanatory_note=er["explanatory_note"],
                    )
                    for er in ev_rows
                ]
                verifications.append(
                    VerificationRecord(
                        claim_id=vr["claim_id"],
                        round_no=vr["round_no"],
                        verdict=VerdictStatus(vr["verdict"]),
                        rationale=vr["rationale"],
                        missing_info=vr["missing_info"],
                        follow_up_query=vr["follow_up_query"],
                        evidence_items=ev_items,
                        is_final=bool(vr["is_final"]),
                        causation_warning=vr["causation_warning"],
                        guardrail_notes=json.loads(vr["guardrail_notes_json"]),
                        claim_nature=vr["claim_nature"] if "claim_nature" in vr.keys() else "direct_quote_measurement",
                        numeric_context=vr["numeric_context"] if "numeric_context" in vr.keys() else "",
                        independent_support_count=vr["independent_support_count"] if "independent_support_count" in vr.keys() else 1,
                        single_source_warning=vr["single_source_warning"] if "single_source_warning" in vr.keys() else None,
                    )
                )

            event_rows = conn.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
            events = [
                RunEvent(
                    seq=r["seq"],
                    node_name=r["node_name"],
                    round_no=r["round_no"],
                    status=r["status"],
                    started_at=r["started_at"],
                    finished_at=r["finished_at"],
                    duration_ms=r["duration_ms"],
                    message=r["message"],
                    payload_summary=r["payload_summary"],
                )
                for r in event_rows
            ]

            rep_row = conn.execute(
                "SELECT * FROM reports WHERE run_id = ?", (run_id,)
            ).fetchone()
            report = FinalReport.model_validate_json(rep_row["report_json"]) if rep_row else None

            return RunBundle(
                metadata=metadata,
                sub_questions=sub_questions,
                search_rounds=search_rounds,
                sources=sources,
                chunks=chunks,
                claims=claims,
                verifications=verifications,
                events=events,
                report=report,
            )

    def get_claim_provenance(self, run_id: str, claim_id: str) -> ClaimProvenanceView | None:
        """Belirli bir iddianın 1. tur ilk dayanağını, 2. tur sorgusunu ve son kararını getirir."""
        bundle = self.load_run_bundle(run_id)
        if bundle is None:
            return None
        claim = next((c for c in bundle.claims if c.claim_id == claim_id), None)
        if claim is None:
            return None

        sub_q = next((sq for sq in bundle.sub_questions if sq.sq_id == claim.sq_id), None)
        init_src = next((s for s in bundle.sources if s.source_id == claim.initial_source_id), None)
        init_chunk = next((ch for ch in bundle.chunks if ch.chunk_id == claim.initial_chunk_id), None)

        all_vers = sorted(
            [v for v in bundle.verifications if v.claim_id == claim_id],
            key=lambda v: v.round_no,
        )
        round1_ver = next((v for v in all_vers if v.round_no == 1), None)
        final_ver = next((v for v in all_vers if v.is_final), all_vers[-1] if all_vers else None)

        follow_up = next(
            (
                sr
                for sr in bundle.search_rounds
                if sr.round_no > 1 and sr.triggered_by_claim_id == claim_id
            ),
            next((sr for sr in bundle.search_rounds if sr.round_no > 1), None),
        )

        return ClaimProvenanceView(
            claim=claim,
            sub_question=sub_q,
            initial_source=init_src,
            initial_chunk=init_chunk,
            round1_verification=round1_ver,
            follow_up_search_round=follow_up,
            final_verification=final_ver,
            all_verifications=all_vers,
        )

    def delete_run(self, run_id: str) -> bool:
        """Tek bir araştırma kaydını ve bağlı tüm alt tabloları atomik olarak siler."""
        self.ensure_initialized()
        with closing(get_connection(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute("DELETE FROM verification_evidence WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM verifications WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM claims WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM evidence_chunks WHERE run_id = ?", (run_id,))
                conn.execute("UPDATE sources SET duplicate_of_source_id = NULL WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM sources WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM search_rounds WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM sub_questions WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM reports WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
                cur = conn.execute("DELETE FROM research_runs WHERE run_id = ?", (run_id,))
                conn.commit()
                return cur.rowcount > 0
            except Exception:
                conn.rollback()
                raise

    def clear_all_runs(self) -> int:
        """Veritabanındaki tüm araştırma kayıtlarını güvenli yabancı anahtar sırasıyla temizler."""
        self.ensure_initialized()
        with closing(get_connection(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute("DELETE FROM verification_evidence")
                conn.execute("DELETE FROM verifications")
                conn.execute("DELETE FROM claims")
                conn.execute("DELETE FROM evidence_chunks")
                conn.execute("UPDATE sources SET duplicate_of_source_id = NULL")
                conn.execute("DELETE FROM sources")
                conn.execute("DELETE FROM search_rounds")
                conn.execute("DELETE FROM sub_questions")
                conn.execute("DELETE FROM reports")
                conn.execute("DELETE FROM run_events")
                cur = conn.execute("DELETE FROM research_runs")
                conn.commit()
                return int(cur.rowcount or 0)
            except Exception:
                conn.rollback()
                raise
