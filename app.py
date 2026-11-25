"""Çok Ajanlı Araştırma Asistanı — FastAPI Web Sunucusu ve REST / SSE API."""

from __future__ import annotations

import html
import json
import os
import queue
import re
import sys
import threading
import uuid
import webbrowser
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_ROOT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from p04_research.config import AppConfig, load_config, validate_question  # noqa: E402
from p04_research.db.repository import ResearchRepository  # noqa: E402
from p04_research.export.markdown_export import (  # noqa: E402
    clean_publisher_name,
    clean_report_title,
    format_source_display_name,
    is_external_web_url,
    is_statement_duplicate_of_quote,
    render_report_markdown,
)
from p04_research.export.pdf_export import render_report_pdf_bytes  # noqa: E402
from p04_research.models import RunBundle, RunEvent, RunStatus, SourceRecord  # noqa: E402
from p04_research.providers.mock_demo import DEMO_PRESETS  # noqa: E402
from p04_research.workflow.graph import execute_research_run  # noqa: E402
from p04_research.workflow.i18n import (  # noqa: E402
    run_status_label,
    source_type_label,
    sq_status_label,
    t,
    verdict_label,
)

_WEB_DIR = _ROOT_DIR / "web"
_CITE_TAG_RE = re.compile(r"\[(S\d+)\]")
_PRESET_PREFIX_RE = re.compile(r"^\d+\s*·\s*[^:]+:\s*")

_NODE_DISPLAY: dict[str, dict[str, str]] = {
    "tr": {
        "planner": "Planlayıcı",
        "researcher": "Web Araştırmacısı",
        "analyst": "Analist",
        "verifier": "Kanıt Denetçisi",
        "router": "Takip Yönlendiricisi",
        "reporter": "Raportör",
        "quality_gate": "Kalite Denetimi",
    },
    "en": {
        "planner": "Planner",
        "researcher": "Web Researcher",
        "analyst": "Analyst",
        "verifier": "Evidence Verifier",
        "router": "Follow-up Router",
        "reporter": "Reporter",
        "quality_gate": "Quality Gate",
    },
}


class ResearchRequest(BaseModel):
    question: str = ""
    mode: str = Field(default="demo")
    depth: str = Field(default="standard")
    language: str = Field(default="tr")
    idempotency_key: str | None = None


def _clean_preset_label(raw: str) -> str:
    return _PRESET_PREFIX_RE.sub("", raw).strip()


def _source_short_label(src: SourceRecord) -> str:
    pub = clean_publisher_name(src.publisher, title=src.title, url=src.url)
    if pub:
        return pub
    title, _ = format_source_display_name(src.title, src.publisher, src.url)
    return title[:32]


def _linkify_citations_html(text: str, src_map: dict[str, SourceRecord]) -> str:
    """[S#] atıflarını ham URL göstermeden güvenli tıklanabilir kaynak adına dönüştürür."""
    escaped = html.escape(text or "")

    def _repl(match: re.Match) -> str:
        sid = match.group(1)
        src = src_map.get(sid)
        if src is None:
            return match.group(0)
        label = html.escape(_source_short_label(src))
        if is_external_web_url(src.url, is_mock=src.is_mock):
            safe_url = html.escape(src.url.strip(), quote=True)
            return (
                f'<a class="inline-cite" href="{safe_url}" target="_blank" '
                f'rel="noopener noreferrer">{label}</a>'
            )
        return f'<span class="inline-cite-static">{label}</span>'

    return _CITE_TAG_RE.sub(_repl, escaped)


def _source_display_payload(
    source_id: str,
    title: str,
    publisher: str,
    url: str,
    is_mock: bool,
) -> dict[str, Any]:
    clean_t, clean_pub = format_source_display_name(title, publisher, url)
    if clean_pub and clean_pub.lower() not in clean_t.lower():
        display_label = f"{clean_t} — {clean_pub}"
    else:
        display_label = clean_t
    has_ext = is_external_web_url(url, is_mock=is_mock)
    return {
        "source_id": source_id,
        "clean_title": clean_t,
        "clean_publisher": clean_pub,
        "display_label": display_label,
        "short_label": clean_pub or clean_t[:28],
        "url": url.strip() if has_ext else None,
        "has_external_url": has_ext,
        "is_mock": is_mock,
    }


def _serialize_bundle(
    bundle: RunBundle,
    *,
    persisted: bool = True,
    persist_error: str | None = None,
) -> dict[str, Any]:
    m = bundle.metadata
    lang = m.language.value
    rep = bundle.report
    src_map = {s.source_id: s for s in bundle.sources}
    chunks_by_src: dict[str, list[dict[str, Any]]] = {}
    for ch in bundle.chunks:
        chunks_by_src.setdefault(ch.source_id, []).append(
            {
                "chunk_id": ch.chunk_id,
                "round_no": ch.round_no,
                "excerpt_text": ch.excerpt_text,
            }
        )

    sources_out: list[dict[str, Any]] = []
    for s in bundle.sources:
        disp = _source_display_payload(s.source_id, s.title, s.publisher, s.url, s.is_mock)
        sources_out.append(
            {
                **disp,
                "published_date": s.published_date,
                "published_date_label": s.published_date or t("unknown", lang),
                "source_type_label": source_type_label(s.source_type, lang),
                "relevance_score": round(s.relevance_score, 2),
                "credibility_score": round(s.credibility_score, 2),
                "is_accessible": s.is_accessible,
                "duplicate_of_source_id": s.duplicate_of_source_id,
                "excerpts": chunks_by_src.get(s.source_id, []),
            }
        )

    report_out: dict[str, Any] | None = None
    if rep is not None:
        findings_out: list[dict[str, Any]] = []
        for item in rep.supported_findings:
            primary_q = item.quotes[0] if item.quotes else None
            src_obj = src_map.get(primary_q.source_id) if primary_q else None
            is_dup = (
                primary_q is not None
                and not primary_q.explanatory_note
                and primary_q.quote_language == lang
                and is_statement_duplicate_of_quote(item.statement, primary_q.quote)
            )
            src_disp = (
                _source_display_payload(
                    src_obj.source_id, src_obj.title, src_obj.publisher, src_obj.url, src_obj.is_mock
                )
                if src_obj is not None
                else None
            )
            findings_out.append(
                {
                    "claim_id": item.claim_id,
                    "sq_id": item.sq_id,
                    "statement": item.statement,
                    "statement_html": _linkify_citations_html(item.statement, src_map),
                    "primary_quote": primary_q.quote if primary_q else None,
                    "explanatory_note": primary_q.explanatory_note if primary_q else None,
                    "is_duplicate_of_quote": is_dup,
                    "citation_ids": list(item.citation_ids),
                    "causation_warning": item.causation_warning,
                    "source": src_disp,
                }
            )

        conflicts_out: list[dict[str, Any]] = []
        for conf in rep.conflicts:
            sa = src_map.get(conf.side_a.source_id)
            sb = src_map.get(conf.side_b.source_id)
            conflicts_out.append(
                {
                    "claim_id": conf.claim_id,
                    "statement": conf.statement,
                    "explanation": conf.explanation,
                    "side_a": {
                        "source_id": conf.side_a.source_id,
                        "quote": conf.side_a.quote,
                        "source": _source_display_payload(
                            sa.source_id, sa.title, sa.publisher, sa.url, sa.is_mock
                        )
                        if sa
                        else None,
                    },
                    "side_b": {
                        "source_id": conf.side_b.source_id,
                        "quote": conf.side_b.quote,
                        "source": _source_display_payload(
                            sb.source_id, sb.title, sb.publisher, sb.url, sb.is_mock
                        )
                        if sb
                        else None,
                    },
                }
            )

        limitations_out = [
            {
                "claim_id": lim.claim_id,
                "statement": lim.statement,
                "verdict": lim.verdict.value,
                "verdict_label": verdict_label(lim.verdict, lang),
                "rationale": lim.rationale,
                "missing_info": lim.missing_info,
            }
            for lim in rep.limitations
        ]

        sub_questions_out = [
            {
                "sq_id": sq.sq_id,
                "text": sq.text,
                "status": sq.status.value,
                "status_label": sq_status_label(sq.status, lang),
            }
            for sq in rep.sub_questions
        ]

        bib_out: list[dict[str, Any]] = []
        for b in rep.bibliography:
            disp = _source_display_payload(b.source_id, b.title, b.publisher, b.url, b.is_mock)
            bib_out.append(
                {
                    **disp,
                    "published_date": b.published_date,
                    "duplicate_of_source_id": b.duplicate_of_source_id,
                }
            )

        report_out = {
            "title": rep.title,
            "display_title": clean_report_title(rep.title),
            "short_answer": rep.short_answer,
            "short_answer_html": _linkify_citations_html(rep.short_answer, src_map),
            "findings": findings_out,
            "conflicts": conflicts_out,
            "limitations": limitations_out,
            "sub_questions": sub_questions_out,
            "bibliography": bib_out,
            "error_notice": rep.error_notice,
        }

    # İddia denetim zinciri
    provenance_out: list[dict[str, Any]] = []
    sq_map = {sq.sq_id: sq.text for sq in bundle.sub_questions}
    for claim in bundle.claims:
        all_vers = sorted(
            [v for v in bundle.verifications if v.claim_id == claim.claim_id],
            key=lambda v: v.round_no,
        )
        r1 = next((v for v in all_vers if v.round_no == 1), None)
        fv = next((v for v in all_vers if v.is_final), all_vers[-1] if all_vers else None)
        fu = next(
            (sr for sr in bundle.search_rounds if sr.round_no > 1 and sr.triggered_by_claim_id == claim.claim_id),
            next((sr for sr in bundle.search_rounds if sr.round_no > 1), None),
        )
        provenance_out.append(
            {
                "claim_id": claim.claim_id,
                "statement": claim.statement,
                "sub_question": sq_map.get(claim.sq_id, ""),
                "initial_source_id": claim.initial_source_id,
                "initial_quote": claim.initial_quote,
                "round1_verdict_label": verdict_label(r1.verdict, lang) if r1 else None,
                "round1_rationale": r1.rationale if r1 else None,
                "followup_query": fu.query if fu else None,
                "final_round": fv.round_no if fv else None,
                "final_verdict": fv.verdict.value if fv else None,
                "final_verdict_label": verdict_label(fv.verdict, lang) if fv else None,
                "final_rationale": fv.rationale if fv else None,
                "evidence_items": [
                    {"source_id": ev.source_id, "quote": ev.quote, "role": ev.role.value}
                    for ev in (fv.evidence_items if fv else [])
                ],
            }
        )

    node_labels = _NODE_DISPLAY["en" if lang == "en" else "tr"]
    events_out = [
        {
            "seq": ev.seq,
            "node_name": ev.node_name,
            "node_label": node_labels.get(ev.node_name, ev.node_name),
            "round_no": ev.round_no,
            "status": ev.status,
            "message": ev.message,
            "duration_ms": ev.duration_ms,
        }
        for ev in bundle.events
        if ev.status in ("finish", "decision", "error")
    ]

    rounds_out = [
        {
            "round_no": sr.round_no,
            "query": sr.query,
            "result_count": sr.result_count,
            "triggered_by_claim_id": sr.triggered_by_claim_id,
            "trigger_reason": sr.trigger_reason,
        }
        for sr in bundle.search_rounds
    ]

    return {
        "persisted": persisted,
        "persist_error": persist_error,
        "metadata": {
            "run_id": m.run_id,
            "question": m.question,
            "display_title": clean_report_title(rep.title if rep else m.question),
            "language": lang,
            "depth": m.depth.value,
            "mode": m.mode.value,
            "started_at": m.started_at,
            "duration_ms": m.duration_ms,
            "status": m.status.value,
            "status_label": run_status_label(m.status, lang),
            "model_calls": m.model_calls,
            "search_calls": m.search_calls,
            "total_tokens": m.total_tokens,
            "token_measurement_type": m.token_measurement_type,
            "estimated_cost_usd": m.estimated_cost_usd,
            "source_count": len(bundle.sources),
            "used_local_fallback": bool(
                m.mode.value == "live" and m.token_measurement_type == "estimated"
            ),
        },
        "report": report_out,
        "sources": sources_out,
        "provenance": provenance_out,
        "events": events_out,
        "search_rounds": rounds_out,
    }


def _get_runtime(base_dir: Path) -> tuple[AppConfig, ResearchRepository]:
    config = load_config(base_dir=base_dir)
    repo = ResearchRepository(config.db_path, known_secrets=config.known_secrets())
    repo.ensure_initialized()
    return config, repo


def create_app(base_dir: Path | None = None) -> FastAPI:
    root = base_dir or _ROOT_DIR
    api = FastAPI(title="Araştırma Asistanı", docs_url=None, redoc_url=None)

    if _WEB_DIR.exists():
        api.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @api.get("/")
    def serve_index() -> FileResponse:
        return FileResponse(_WEB_DIR / "index.html")

    @api.get("/api/config")
    def get_app_config() -> dict[str, Any]:
        config, _ = _get_runtime(root)
        force_demo = os.environ.get("P04_FORCE_DEMO_DEFAULT") == "1"
        default_mode = "live" if (config.has_tavily_key() and not force_demo) else "demo"
        return {
            "default_mode": default_mode,
            "has_tavily_key": config.has_tavily_key(),
            "has_openai_key": config.has_openai_key(),
            "presets": [
                {
                    "key": p.key,
                    "label": _clean_preset_label(p.label_tr),
                    "question_tr": p.default_question_tr,
                    "question_en": p.default_question_en,
                    "description": p.description_tr,
                }
                for p in DEMO_PRESETS
            ],
        }

    @api.get("/api/runs")
    def list_runs(limit: int = 100) -> dict[str, Any]:
        _, repo = _get_runtime(root)
        raw_runs = repo.list_runs(limit=min(max(limit, 1), 200))
        items = []
        for r in raw_runs:
            lang = r["language"]
            st_enum = RunStatus(r["status"])
            items.append(
                {
                    "run_id": r["run_id"],
                    "question": r["question"],
                    "display_title": clean_report_title(r["question"]),
                    "language": lang,
                    "depth": r["depth"],
                    "mode": r["mode"],
                    "started_at": r["started_at"],
                    "status": r["status"],
                    "status_label": run_status_label(st_enum, lang),
                    "supported_claim_count": r["supported_claim_count"],
                    "source_count": r["source_count"],
                }
            )
        return {"runs": items}

    @api.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        _, repo = _get_runtime(root)
        bundle = repo.load_run_bundle(run_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Araştırma kaydı bulunamadı.")
        return _serialize_bundle(bundle)

    @api.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> dict[str, Any]:
        _, repo = _get_runtime(root)
        deleted = repo.delete_run(run_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Silinecek kayıt bulunamadı.")
        return {"deleted": True, "run_id": run_id}

    @api.delete("/api/runs")
    def clear_runs() -> dict[str, Any]:
        _, repo = _get_runtime(root)
        count = repo.clear_all_runs()
        return {"deleted_count": count}

    @api.post("/api/research")
    def run_research_sync(req: ResearchRequest) -> dict[str, Any]:
        lang = "en" if req.language == "en" else "tr"
        depth = "fast" if req.depth == "fast" else "standard"
        mode = "live" if req.mode == "live" else "demo"
        ok, cleaned_q, err_msg = validate_question(req.question, lang)
        if not ok:
            raise HTTPException(status_code=400, detail=err_msg)

        config, repo = _get_runtime(root)
        bundle, persisted_ok, persist_err = execute_research_run(
            question=cleaned_q,
            language=lang,
            depth=depth,
            mode=mode,
            config=config,
            repository=repo,
            idempotency_key=req.idempotency_key or f"web_{uuid.uuid4().hex}",
        )
        return _serialize_bundle(bundle, persisted=persisted_ok, persist_error=persist_err)

    @api.post("/api/research/stream")
    def run_research_stream(req: ResearchRequest) -> StreamingResponse:
        lang = "en" if req.language == "en" else "tr"
        depth = "fast" if req.depth == "fast" else "standard"
        mode = "live" if req.mode == "live" else "demo"
        ok, cleaned_q, err_msg = validate_question(req.question, lang)
        if not ok:
            raise HTTPException(status_code=400, detail=err_msg)

        config, repo = _get_runtime(root)
        idem = req.idempotency_key or f"web_{uuid.uuid4().hex}"
        node_labels = _NODE_DISPLAY["en" if lang == "en" else "tr"]
        ev_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def _on_event(ev: RunEvent) -> None:
            if ev.status in ("start", "finish", "decision", "error"):
                ev_queue.put(
                    {
                        "type": "event",
                        "node_name": ev.node_name,
                        "node_label": node_labels.get(ev.node_name, ev.node_name),
                        "round_no": ev.round_no,
                        "status": ev.status,
                        "message": ev.message,
                    }
                )

        def _worker() -> None:
            try:
                bundle, persisted_ok, persist_err = execute_research_run(
                    question=cleaned_q,
                    language=lang,
                    depth=depth,
                    mode=mode,
                    config=config,
                    repository=repo,
                    idempotency_key=idem,
                    on_event=_on_event,
                )
                ev_queue.put(
                    {
                        "type": "complete",
                        "data": _serialize_bundle(
                            bundle, persisted=persisted_ok, persist_error=persist_err
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - beklenmeyen çalışma hatası
                ev_queue.put({"type": "error", "message": str(exc)})
            finally:
                ev_queue.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        def _event_gen() -> Iterator[str]:
            while True:
                item = ev_queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            _event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.get("/api/runs/{run_id}/export/md")
    def export_markdown(run_id: str) -> Response:
        _, repo = _get_runtime(root)
        bundle = repo.load_run_bundle(run_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Araştırma kaydı bulunamadı.")
        md_text = render_report_markdown(bundle)
        return Response(
            content=md_text.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="arastirma_raporu_{run_id}.md"'},
        )

    @api.get("/api/runs/{run_id}/export/pdf")
    def export_pdf(run_id: str) -> Response:
        _, repo = _get_runtime(root)
        bundle = repo.load_run_bundle(run_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Araştırma kaydı bulunamadı.")
        pdf_bytes = render_report_pdf_bytes(bundle)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="arastirma_raporu_{run_id}.pdf"'},
        )

    return api


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("P04_HOST", "127.0.0.1")
    port = int(os.environ.get("P04_PORT", "8000"))
    if os.environ.get("P04_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
