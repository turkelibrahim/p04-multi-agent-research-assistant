"""Çok Ajanlı Araştırma Asistanı — Sade Editoryal Streamlit Arayüzü."""

from __future__ import annotations

import html
import os
import re
import sys
import uuid
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st  # noqa: E402

from p04_research.config import load_config, validate_question  # noqa: E402
from p04_research.db.repository import ResearchRepository  # noqa: E402
from p04_research.export.markdown_export import (  # noqa: E402
    clean_publisher_name,
    clean_report_title,
    format_source_display_name,
    format_source_markdown_link,
    is_external_web_url,
    is_statement_duplicate_of_quote,
    render_report_markdown,
)
from p04_research.export.pdf_export import render_report_pdf_bytes  # noqa: E402
from p04_research.models import (  # noqa: E402
    ClaimProvenanceView,
    RunBundle,
    RunMode,
    RunStatus,
    SourceRecord,
    SubQuestionStatus,
)
from p04_research.providers.mock_demo import DEMO_PRESETS  # noqa: E402
from p04_research.workflow.graph import execute_research_run  # noqa: E402
from p04_research.workflow.i18n import (  # noqa: E402
    run_status_label,
    source_type_label,
    sq_status_label,
    t,
    verdict_label,
)

NAV_VIEWS = [
    "Yeni Araştırma",
    "Rapor",
    "Kaynaklar",
    "Akış",
    "Geçmiş",
]

_NODE_DISPLAY = {
    "planner": "Planlayıcı",
    "researcher": "Web Araştırmacısı",
    "analyst": "Analist",
    "verifier": "Kanıt Denetçisi",
    "router": "Takip Yönlendiricisi",
    "reporter": "Raportör",
    "quality_gate": "Kalite Denetimi",
}

_CITE_TAG_RE = re.compile(r"\[(S\d+)\]")
_PRESET_PREFIX_RE = re.compile(r"^\d+\s*·\s*[^:]+:\s*")


def _apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer { visibility: hidden; }
        .block-container {
            max-width: 880px;
            padding-top: 1.9rem;
            padding-bottom: 3.5rem;
        }
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
            color: #18181B;
        }
        h1, h2, h3, h4 {
            letter-spacing: -0.02em;
            color: #18181B;
        }
        /* Radyo dairelerini gizleyip modern segmentli sekme çubuğu olarak biçimlendir */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            background: #EFEFED;
            padding: 3px;
            border-radius: 9px;
            border: 1px solid #E4E4E7;
            display: inline-flex !important;
            flex-wrap: wrap;
            gap: 2px;
            width: fit-content;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            background: transparent;
            padding: 5px 12px !important;
            border-radius: 6px;
            margin: 0 !important;
            cursor: pointer;
            transition: background 0.14s ease, color 0.14s ease, box-shadow 0.14s ease;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] p {
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            color: #52525B !important;
            margin: 0 !important;
            line-height: 1.35 !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            background: rgba(255, 255, 255, 0.55);
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            background: #FFFFFF !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
            color: #18181B !important;
            font-weight: 600 !important;
        }
        .ed-header-title {
            font-size: 1.3rem;
            font-weight: 650;
            letter-spacing: -0.025em;
            color: #18181B;
            margin: 0 0 0.65rem 0;
        }
        .ed-hairline {
            border: 0;
            border-top: 1px solid #E4E4E7;
            margin: 0.85rem 0 1.4rem 0;
        }
        .ed-report-title {
            font-size: 1.44rem;
            font-weight: 650;
            line-height: 1.34;
            letter-spacing: -0.025em;
            color: #18181B;
            margin: 0 0 0.45rem 0;
        }
        .ed-meta {
            font-size: 0.85rem;
            color: #71717A;
            padding-top: 6px;
        }
        .ed-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 500;
            background: #F4F4F5;
            color: #52525B;
            border: 1px solid #E4E4E7;
            margin-left: 6px;
            vertical-align: middle;
        }
        .ed-pill-demo {
            background: #FEF3C7;
            color: #92400E;
            border-color: #FDE68A;
        }
        .ed-sq-pill {
            display: inline-block;
            padding: 1px 8px;
            border-radius: 5px;
            font-size: 0.76rem;
            font-weight: 500;
            margin-right: 6px;
        }
        .ed-sq-answered { background: #ECFDF5; color: #065F46; border: 1px solid #D1FAE5; }
        .ed-sq-partial { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
        .ed-sq-conflict { background: #FFEDD5; color: #9A3412; border: 1px solid #FED7AA; }
        .ed-sq-unanswered { background: #F4F4F5; color: #52525B; border: 1px solid #E4E4E7; }
        .ed-summary {
            background: #FFFFFF;
            border: 1px solid #E4E4E7;
            border-left: 3px solid #18181B;
            border-radius: 8px;
            padding: 16px 20px;
            font-size: 0.98rem;
            line-height: 1.7;
            color: #18181B;
            margin: 0.45rem 0 1.45rem 0;
        }
        .ed-summary a {
            color: #1D4ED8;
            text-decoration: underline;
            text-underline-offset: 2px;
        }
        .ed-finding-text {
            font-size: 0.96rem;
            line-height: 1.64;
            color: #18181B;
            margin: 0 0 8px 0;
        }
        .ed-finding-source {
            font-size: 0.84rem;
            color: #71717A;
            margin: 0;
        }
        .ed-finding-source a {
            color: #1D4ED8;
            text-decoration: none;
        }
        .ed-finding-source a:hover {
            text-decoration: underline;
            text-underline-offset: 2px;
        }
        .ed-quote {
            border-left: 2px solid #D4D4D8;
            padding: 6px 12px;
            margin: 8px 0 6px 0;
            color: #52525B;
            font-size: 0.91rem;
            line-height: 1.56;
        }
        .ed-step {
            background: #FFFFFF;
            border: 1px solid #E4E4E7;
            border-radius: 8px;
            padding: 11px 15px;
            margin-bottom: 8px;
            font-size: 0.91rem;
        }
        .stButton > button[kind="primary"] {
            background-color: #18181B;
            color: #FFFFFF;
            border: 1px solid #18181B;
            border-radius: 7px;
            font-weight: 500;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #27272A;
            border-color: #27272A;
        }
        .stButton > button[kind="secondary"],
        .stDownloadButton > button {
            background-color: #FFFFFF;
            color: #27272A;
            border: 1px solid #E4E4E7;
            border-radius: 7px;
            font-weight: 500;
            font-size: 0.85rem;
        }
        .stButton > button[kind="secondary"]:hover,
        .stDownloadButton > button:hover {
            background-color: #F4F4F5;
            border-color: #D4D4D8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _human_date(iso_str: str | None) -> str:
    if not iso_str:
        return "-"
    clean = iso_str.replace("T", " ")
    return clean[:16]


def _clean_preset_label(raw_label: str) -> str:
    return _PRESET_PREFIX_RE.sub("", raw_label).strip()


def _source_short_label(src: SourceRecord) -> str:
    pub = clean_publisher_name(src.publisher, title=src.title, url=src.url)
    if pub:
        return pub
    title, _ = format_source_display_name(src.title, src.publisher, src.url)
    return title[:32]


def _format_source_html_link(src: SourceRecord) -> str:
    clean_t, clean_pub = format_source_display_name(src.title, src.publisher, src.url)
    label = f"{clean_t} — {clean_pub}" if clean_pub and clean_pub.lower() not in clean_t.lower() else clean_t
    safe_label = html.escape(label)
    if is_external_web_url(src.url, is_mock=src.is_mock):
        safe_url = html.escape(src.url.strip(), quote=True)
        return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_label}</a>'
    return f"<strong>{safe_label}</strong>"


def _linkify_citations_md(text: str, src_map: dict[str, SourceRecord]) -> str:
    """Metindeki [S#] atıflarını ham URL göstermeden tıklanabilir kaynak adına dönüştürür."""
    def _repl(match: re.Match) -> str:
        sid = match.group(1)
        src = src_map.get(sid)
        if src is None:
            return match.group(0)
        label = _source_short_label(src).replace("[", "(").replace("]", ")")
        if is_external_web_url(src.url, is_mock=src.is_mock):
            safe_url = src.url.strip().replace(" ", "%20").replace(")", "%29")
            return f"([{label}]({safe_url}))"
        return f"({label})"

    return _CITE_TAG_RE.sub(_repl, text or "")


def _linkify_citations_html(text: str, src_map: dict[str, SourceRecord]) -> str:
    """HTML kutusu içinde güvenli kaçışla [S#] atıflarını parantez içi kaynak adına bağlar."""
    escaped = html.escape(text or "")

    def _repl(match: re.Match) -> str:
        sid = match.group(1)
        src = src_map.get(sid)
        if src is None:
            return match.group(0)
        label = html.escape(_source_short_label(src))
        if is_external_web_url(src.url, is_mock=src.is_mock):
            safe_url = html.escape(src.url.strip(), quote=True)
            return f'(<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{label}</a>)'
        return f"(<strong>{label}</strong>)"

    return _CITE_TAG_RE.sub(_repl, escaped)


def _sq_pill_class(status: SubQuestionStatus) -> str:
    return {
        SubQuestionStatus.ANSWERED: "ed-sq-answered",
        SubQuestionStatus.PARTIALLY_ANSWERED: "ed-sq-partial",
        SubQuestionStatus.CONFLICTING: "ed-sq-conflict",
        SubQuestionStatus.UNANSWERED: "ed-sq-unanswered",
    }.get(status, "ed-sq-unanswered")


def _init_session_state() -> None:
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = NAV_VIEWS[0]
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = DEMO_PRESETS[0].default_question_tr
    if "active_bundle" not in st.session_state:
        st.session_state["active_bundle"] = None
    if "active_persist_error" not in st.session_state:
        st.session_state["active_persist_error"] = None
    if "selected_source_id" not in st.session_state:
        st.session_state["selected_source_id"] = None
    if "selected_claim_id" not in st.session_state:
        st.session_state["selected_claim_id"] = None
    if "form_nonce" not in st.session_state:
        st.session_state["form_nonce"] = uuid.uuid4().hex

    if "pending_view" in st.session_state:
        st.session_state["active_view"] = st.session_state.pop("pending_view")
    if "pending_question" in st.session_state:
        st.session_state["question_input"] = st.session_state.pop("pending_question")
    if "pending_source_id" in st.session_state:
        st.session_state["selected_source_id"] = st.session_state.pop("pending_source_id")
    if "pending_claim_id" in st.session_state:
        st.session_state["selected_claim_id"] = st.session_state.pop("pending_claim_id")


def _jump_to_source(source_id: str) -> None:
    st.session_state["pending_view"] = "Kaynaklar"
    st.session_state["pending_source_id"] = source_id
    st.rerun()


def _render_download_buttons(bundle: RunBundle, key_prefix: str) -> None:
    lang = bundle.metadata.language.value
    md_text = render_report_markdown(bundle)
    pdf_bytes = render_report_pdf_bytes(bundle)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Markdown" if lang == "tr" else "Markdown",
            data=md_text.encode("utf-8"),
            file_name=f"arastirma_raporu_{bundle.metadata.run_id}.md",
            mime="text/markdown; charset=utf-8",
            key=f"{key_prefix}_dl_md",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="PDF",
            data=pdf_bytes,
            file_name=f"arastirma_raporu_{bundle.metadata.run_id}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_pdf",
            use_container_width=True,
        )


def _render_new_research_screen(config, repo: ResearchRepository) -> None:
    mode_options = ["Canlı Web", "Çevrimdışı Demo"]
    if "pending_mode_radio" in st.session_state:
        st.session_state["mode_radio"] = st.session_state.pop("pending_mode_radio")
    elif "mode_radio" not in st.session_state:
        force_demo = os.environ.get("P04_FORCE_DEMO_DEFAULT") == "1"
        st.session_state["mode_radio"] = (
            mode_options[0] if (config.has_tavily_key() and not force_demo) else mode_options[1]
        )

    with st.container(border=True):
        with st.form("research_form", clear_on_submit=False, border=False):
            question_val = st.text_area(
                "Araştırma Sorusu",
                key="question_input",
                height=110,
                placeholder="Araştırmak istediğiniz soruyu veya konuyu yazın...",
            )
            c_mode, c_depth, c_lang = st.columns([1.45, 1, 1])
            with c_mode:
                mode_label = st.radio(
                    "Çalışma Modu",
                    options=mode_options,
                    horizontal=True,
                    key="mode_radio",
                )
            with c_depth:
                depth_choice = st.radio(
                    "Derinlik",
                    options=["Standart", "Hızlı"],
                    horizontal=True,
                    key="depth_radio",
                )
            with c_lang:
                lang_choice = st.radio(
                    "Dil",
                    options=["Türkçe", "İngilizce"],
                    horizontal=True,
                    key="lang_radio",
                )
            submitted = st.form_submit_button(
                "Araştırmayı Başlat",
                type="primary",
                use_container_width=True,
            )

    selected_mode = "live" if mode_label == "Canlı Web" else "demo"
    if selected_mode == "live":
        if config.has_tavily_key():
            st.caption("Canlı web araması aktif — güncel web kaynakları üzerinden doğrulama yapılır.")
        else:
            st.caption("Canlı arama için .env dosyasında Tavily API anahtarı tanımlı olmalıdır.")
    else:
        st.caption("Çevrimdışı demo aktif — internet bağlantısı olmadan örnek veriyle çalışır.")

    if submitted:
        lang_code = "en" if lang_choice == "İngilizce" else "tr"
        depth_code = "fast" if depth_choice == "Hızlı" else "standard"
        ok, cleaned_q, err_msg = validate_question(question_val, lang_code)
        if not ok:
            st.error(err_msg)
        else:
            with st.status("Araştırma yürütülüyor...", expanded=True) as status_box:
                def _on_ev(ev):
                    if ev.status in ("finish", "decision", "error"):
                        lbl = _NODE_DISPLAY.get(ev.node_name, ev.node_name)
                        st.write(f"**{lbl}** — {ev.message}")

                bundle, persisted_ok, persist_err = execute_research_run(
                    question=cleaned_q,
                    language=lang_code,
                    depth=depth_code,
                    mode=selected_mode,
                    config=config,
                    repository=repo,
                    idempotency_key=f"ui_{st.session_state['form_nonce']}",
                    on_event=_on_ev,
                )
                status_box.update(
                    label=f"Tamamlandı ({run_status_label(bundle.metadata.status, lang_code)})",
                    state="complete" if bundle.metadata.status != RunStatus.FAILED else "error",
                )

            st.session_state["active_bundle"] = bundle
            st.session_state["active_persist_error"] = None if persisted_ok else persist_err
            st.session_state["form_nonce"] = uuid.uuid4().hex
            st.session_state["selected_claim_id"] = None
            st.session_state["selected_source_id"] = None
            st.session_state["pending_view"] = "Rapor"
            st.rerun()

    preset_map = {_clean_preset_label(p.label_tr): p for p in DEMO_PRESETS}
    with st.expander("Örnek sorular", expanded=False):
        col_pre, col_btn = st.columns([4, 1])
        with col_pre:
            chosen_label = st.selectbox(
                "Örnek senaryo",
                options=list(preset_map.keys()),
                key="preset_select",
                label_visibility="collapsed",
            )
        chosen_preset = preset_map[chosen_label]
        with col_btn:
            if st.button("Örneği Doldur", key="fill_preset_btn", use_container_width=True):
                st.session_state["pending_question"] = chosen_preset.default_question_tr
                st.session_state["pending_mode_radio"] = mode_options[1]
                st.rerun()
        st.caption(chosen_preset.description_tr)

    recent_runs = repo.list_runs(limit=3)
    if recent_runs:
        st.write("")
        st.markdown("##### Son Araştırmalar")
        for r in recent_runs:
            c_txt, c_open = st.columns([5.2, 1])
            with c_txt:
                mode_lbl = "Demo" if r["mode"] == "demo" else "Web"
                short_q = clean_report_title(r["question"])
                st.markdown(f"**{short_q}**")
                st.caption(
                    f"{_human_date(r['started_at'])} · {mode_lbl} · "
                    f"{r['supported_claim_count']} bulgu · {r['source_count']} kaynak"
                )
            with c_open:
                if st.button("Aç", key=f"quick_open_{r['run_id']}", use_container_width=True):
                    loaded = repo.load_run_bundle(r["run_id"])
                    if loaded is not None:
                        st.session_state["active_bundle"] = loaded
                        st.session_state["active_persist_error"] = None
                        st.session_state["selected_source_id"] = None
                        st.session_state["selected_claim_id"] = None
                        st.session_state["pending_view"] = "Rapor"
                        st.rerun()


def _render_report_screen(bundle: RunBundle | None) -> None:
    if bundle is None or bundle.report is None:
        st.info("Görüntülenecek bir rapor yok. 'Yeni Araştırma' sekmesinden bir soru başlatın.")
        return

    persist_err = st.session_state.get("active_persist_error")
    if persist_err:
        st.error(f"Rapor hazırlandı ancak veritabanına kaydedilemedi: {persist_err}")

    m = bundle.metadata
    rep = bundle.report
    lang = m.language.value
    src_map = {s.source_id: s for s in bundle.sources}

    display_title = clean_report_title(rep.title)
    st.markdown(f"<div class='ed-report-title'>{html.escape(display_title)}</div>", unsafe_allow_html=True)

    meta_col, act_col = st.columns([3.2, 1.8])
    with meta_col:
        demo_pill = (
            f"<span class='ed-pill ed-pill-demo'>{html.escape(t('mock_badge', lang))}</span>"
            if m.mode == RunMode.DEMO
            else ""
        )
        st.markdown(
            f"<div class='ed-meta'>"
            f"{len(rep.supported_findings)} bulgu · {len(rep.bibliography)} kaynak · "
            f"{_human_date(m.started_at)}{demo_pill}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with act_col:
        _render_download_buttons(bundle, key_prefix=f"rep_{m.run_id}")

    if rep.error_notice:
        st.error(rep.error_notice)

    # 1. Kısa Yanıt
    st.markdown("#### Kısa Yanıt" if lang == "tr" else "#### Short Answer")
    linked_summary = _linkify_citations_html(rep.short_answer, src_map)
    st.markdown(f"<div class='ed-summary'>{linked_summary}</div>", unsafe_allow_html=True)

    # 2. Temel Bulgular (tekrarsız, sade kartlar)
    st.markdown(f"#### {t('report_heading_findings', lang)}")
    if not rep.supported_findings:
        st.info(t("no_supported_findings", lang))
    else:
        for item in rep.supported_findings:
            with st.container(border=True):
                primary_q = item.quotes[0] if item.quotes else None
                src_obj = src_map.get(primary_q.source_id) if primary_q else None
                is_dup = (
                    primary_q is not None
                    and not primary_q.explanatory_note
                    and primary_q.quote_language == lang
                    and is_statement_duplicate_of_quote(item.statement, primary_q.quote)
                )
                if is_dup and primary_q is not None:
                    st.markdown(
                        f"<div class='ed-finding-text'>{html.escape(primary_q.quote)}</div>",
                        unsafe_allow_html=True,
                    )
                    if src_obj is not None:
                        st.markdown(
                            f"<div class='ed-finding-source'>Kaynak: {_format_source_html_link(src_obj)}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    stmt_md = _linkify_citations_md(item.statement, src_map)
                    st.markdown(f"**{stmt_md}**")
                    if primary_q is not None:
                        st.markdown(
                            f"<div class='ed-quote'>\"{html.escape(primary_q.quote)}\"</div>",
                            unsafe_allow_html=True,
                        )
                        if primary_q.explanatory_note:
                            st.caption(primary_q.explanatory_note)
                        if src_obj is not None:
                            st.markdown(
                                f"<div class='ed-finding-source'>Kaynak: {_format_source_html_link(src_obj)}</div>",
                                unsafe_allow_html=True,
                            )

    # 3. Çelişen Bulgular (yalnızca varsa)
    if rep.conflicts:
        st.markdown(f"#### {t('report_heading_conflicts', lang)}")
        for conf in rep.conflicts:
            with st.container(border=True):
                st.markdown(f"**{conf.statement}**")
                st.caption(conf.explanation)
                ca, cb = st.columns(2)
                for col, side in ((ca, conf.side_a), (cb, conf.side_b)):
                    with col:
                        s_obj = src_map.get(side.source_id)
                        st.markdown(
                            f"<div class='ed-quote'>\"{html.escape(side.quote)}\"</div>",
                            unsafe_allow_html=True,
                        )
                        if s_obj is not None:
                            st.markdown(
                                f"<div class='ed-finding-source'>{_format_source_html_link(s_obj)}</div>",
                                unsafe_allow_html=True,
                            )

    # 4. Kaynaklar (yalnızca temiz kaynak adı ve gizli bağlantı; tarihsizlerde 'Bilinmiyor' gürültüsü yok)
    st.markdown("#### Kaynaklar" if lang == "tr" else "#### Sources")
    if not rep.bibliography:
        st.caption("Kaynak kaydedilmedi.")
    else:
        for idx, b in enumerate(rep.bibliography, start=1):
            src_link = format_source_markdown_link(b.title, b.publisher, b.url, is_mock=b.is_mock)
            date_suffix = f" · {b.published_date}" if b.published_date else ""
            st.markdown(f"{idx}. {src_link}{date_suffix}")

    # 5. Daraltılmış Denetim Detayları
    st.write("")
    with st.expander("Denetim Detayları ve Alt Sorular", expanded=False):
        if m.question.strip() != display_title.strip():
            st.markdown(f"**Tam Araştırma Sorusu:** {m.question}")
        if rep.sub_questions:
            st.markdown("**Alt Sorular**")
            for sq in rep.sub_questions:
                cls = _sq_pill_class(sq.status)
                lbl = html.escape(sq_status_label(sq.status, lang))
                st.markdown(
                    f"<div style='margin-bottom:6px;font-size:0.9rem;'>"
                    f"<span class='ed-sq-pill {cls}'>{lbl}</span>{html.escape(sq.text)}</div>",
                    unsafe_allow_html=True,
                )
        if rep.limitations:
            st.markdown("**Sınırlılıklar ve Elenen İddialar**")
            for lim in rep.limitations:
                v_lbl = verdict_label(lim.verdict, lang)
                st.markdown(f"- **{v_lbl}:** {lim.statement} — {lim.rationale}")
        if rep.bibliography:
            st.markdown("**Kaynak Kartlarını İncele**")
            cols = st.columns(min(3, len(rep.bibliography)))
            for idx, b in enumerate(rep.bibliography):
                short_pub = clean_publisher_name(b.publisher, title=b.title, url=b.url) or b.title[:22]
                with cols[idx % len(cols)]:
                    if st.button(
                        f"[{b.source_id}] {short_pub}",
                        key=f"jump_src_det_{b.source_id}",
                        use_container_width=True,
                    ):
                        _jump_to_source(b.source_id)


def _build_local_provenance(bundle: RunBundle, claim_id: str) -> ClaimProvenanceView | None:
    claim = next((c for c in bundle.claims if c.claim_id == claim_id), None)
    if claim is None:
        return None
    sq = next((s for s in bundle.sub_questions if s.sq_id == claim.sq_id), None)
    src = next((s for s in bundle.sources if s.source_id == claim.initial_source_id), None)
    ch = next((c for c in bundle.chunks if c.chunk_id == claim.initial_chunk_id), None)
    all_vers = sorted([v for v in bundle.verifications if v.claim_id == claim_id], key=lambda v: v.round_no)
    r1 = next((v for v in all_vers if v.round_no == 1), None)
    final_v = next((v for v in all_vers if v.is_final), all_vers[-1] if all_vers else None)
    fu = next(
        (sr for sr in bundle.search_rounds if sr.round_no > 1 and sr.triggered_by_claim_id == claim_id),
        next((sr for sr in bundle.search_rounds if sr.round_no > 1), None),
    )
    return ClaimProvenanceView(
        claim=claim,
        sub_question=sq,
        initial_source=src,
        initial_chunk=ch,
        round1_verification=r1,
        follow_up_search_round=fu,
        final_verification=final_v,
        all_verifications=all_vers,
    )


def _render_provenance_screen(bundle: RunBundle | None) -> None:
    st.subheader("Kaynaklar ve Kanıt Detayları")
    if bundle is None:
        st.info("İncelenecek çalışma yok. Önce bir araştırma başlatın veya Geçmişten açın.")
        return

    lang = bundle.metadata.language.value

    if not bundle.sources:
        st.info("Bu çalışmada kaydedilmiş kaynak bulunmuyor.")
    else:
        selected_sid = st.session_state.get("selected_source_id")
        ordered_sources = list(bundle.sources)
        if selected_sid:
            ordered_sources.sort(key=lambda s: (0 if s.source_id == selected_sid else 1, s.source_id))

        for src in ordered_sources:
            is_selected = src.source_id == selected_sid
            clean_t, clean_pub = format_source_display_name(src.title, src.publisher, src.url)
            pub_date = src.published_date or t("unknown", lang)
            header = f"[{src.source_id}] {clean_t} — {clean_pub}" if clean_pub else f"[{src.source_id}] {clean_t}"

            with st.expander(header, expanded=is_selected or len(ordered_sources) <= 2):
                st.caption(
                    f"{source_type_label(src.source_type, lang)} · Yayın: {pub_date} · "
                    f"Uygunluk: {src.relevance_score:.2f} · Güvenilirlik: {src.credibility_score:.2f}"
                )
                if is_external_web_url(src.url, is_mock=src.is_mock):
                    safe_url = src.url.strip().replace(" ", "%20").replace(")", "%29")
                    st.markdown(f"[Kaynağı Aç — {clean_pub or clean_t}]({safe_url})")
                else:
                    st.caption("Kurmaca eğitim kaynağı (harici web bağlantısı yok)")

                if not src.is_accessible:
                    st.warning("Bu kaynağın içeriğine erişilemediği için kanıt olarak kullanılmadı.")
                if src.duplicate_of_source_id:
                    st.info(f"Bu kayıt {src.duplicate_of_source_id} ile aynı içeriğe sahiptir.")

                src_chunks = [ch for ch in bundle.chunks if ch.source_id == src.source_id]
                for ch in src_chunks:
                    st.markdown(
                        f"<div class='ed-quote'>{html.escape(ch.excerpt_text)}</div>",
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.markdown("#### İddia Denetim Zinciri")
    if not bundle.claims:
        st.caption("Bu çalışmada çıkarılmış iddia bulunmuyor.")
        return

    claim_ids = [c.claim_id for c in bundle.claims]
    preferred_cid = st.session_state.get("selected_claim_id")
    if preferred_cid not in claim_ids:
        preferred_cid = "C2" if "C2" in claim_ids else claim_ids[0]
    chosen_cid = st.selectbox(
        "İncelenecek İddia",
        options=claim_ids,
        index=claim_ids.index(preferred_cid),
        format_func=lambda cid: f"{cid}: {next(c.statement for c in bundle.claims if c.claim_id == cid)}",
        key="provenance_claim_select",
    )
    prov = _build_local_provenance(bundle, chosen_cid)
    if prov is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**1. Tur İlk Dayanak**")
            st.caption(prov.sub_question.text if prov.sub_question else "-")
            st.markdown(
                f"<div class='ed-quote'>\"{html.escape(prov.claim.initial_quote)}\"</div>",
                unsafe_allow_html=True,
            )
            if prov.round1_verification:
                st.write(
                    f"**Karar:** {verdict_label(prov.round1_verification.verdict, lang)} — "
                    f"{prov.round1_verification.rationale}"
                )
        with col_b:
            st.markdown("**Sonuç ve Doğrulama**")
            if prov.follow_up_search_round:
                st.caption(f"2. Tur sorgusu: {prov.follow_up_search_round.query}")
            if prov.final_verification:
                fv = prov.final_verification
                st.write(f"**Son Karar (Tur {fv.round_no}):** {verdict_label(fv.verdict, lang)}")
                st.write(fv.rationale)
                for ev in fv.evidence_items:
                    st.markdown(
                        f"<div class='ed-quote'>\"{html.escape(ev.quote)}\"</div>",
                        unsafe_allow_html=True,
                    )


def _render_progress_screen(bundle: RunBundle | None) -> None:
    st.subheader("Araştırma Akışı")
    if bundle is None:
        st.info("Henüz aktif bir çalışma yok.")
        return

    m = bundle.metadata
    lang = m.language.value
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Durum", run_status_label(m.status, lang))
    c2.metric("Kaynak", len(bundle.sources))
    c3.metric("Arama", m.search_calls)
    c4.metric("Model", m.model_calls)

    follow_ups = [sr for sr in bundle.search_rounds if sr.round_no > 1]
    for fu in follow_ups:
        st.info(f"2. Tur Takip Araması: {fu.query} ({fu.trigger_reason or '-'})")

    st.write("")
    finish_events = [e for e in bundle.events if e.status in ("finish", "decision", "error")]
    for ev in finish_events:
        lbl = _NODE_DISPLAY.get(ev.node_name, ev.node_name)
        st.markdown(
            f"<div class='ed-step'><strong>{html.escape(lbl)}</strong> "
            f"<span style='color:#71717A'>· Tur {ev.round_no}</span><br/>"
            f"{html.escape(ev.message)}</div>",
            unsafe_allow_html=True,
        )


def _render_history_screen(repo: ResearchRepository) -> None:
    hdr_col, clr_col = st.columns([4, 1.3])
    with hdr_col:
        st.subheader("Geçmiş Araştırmalar")
    runs = repo.list_runs(limit=100)
    with clr_col:
        if runs and st.button("Tümünü Temizle", key="clear_all_history_btn", use_container_width=True):
            repo.clear_all_runs()
            st.session_state["active_bundle"] = None
            st.session_state["selected_source_id"] = None
            st.session_state["selected_claim_id"] = None
            st.rerun()

    if not runs:
        st.info("Henüz kayıtlı bir araştırma bulunmuyor.")
        return

    active_bundle: RunBundle | None = st.session_state.get("active_bundle")
    for r in runs:
        lang = r["language"]
        status_enum = RunStatus(r["status"])
        mode_lbl = "Demo" if r["mode"] == "demo" else "Web"
        col_info, col_open, col_del = st.columns([6, 1.4, 1])
        with col_info:
            st.markdown(f"**{clean_report_title(r['question'])}**")
            st.caption(
                f"{_human_date(r['started_at'])} · {mode_lbl} · "
                f"{run_status_label(status_enum, lang)} · "
                f"{r['supported_claim_count']} bulgu · {r['source_count']} kaynak"
            )
        with col_open:
            if st.button("Raporu Aç", key=f"open_hist_{r['run_id']}", use_container_width=True):
                loaded = repo.load_run_bundle(r["run_id"])
                if loaded is not None:
                    st.session_state["active_bundle"] = loaded
                    st.session_state["active_persist_error"] = None
                    st.session_state["selected_source_id"] = None
                    st.session_state["selected_claim_id"] = None
                    st.session_state["pending_view"] = "Rapor"
                    st.rerun()
        with col_del:
            if st.button("Sil", key=f"del_hist_{r['run_id']}", use_container_width=True):
                repo.delete_run(r["run_id"])
                if active_bundle is not None and active_bundle.metadata.run_id == r["run_id"]:
                    st.session_state["active_bundle"] = None
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Araştırma Asistanı",
        layout="centered",
    )
    _apply_custom_css()
    _init_session_state()

    config = load_config(base_dir=_ROOT_DIR)
    repo = ResearchRepository(config.db_path, known_secrets=config.known_secrets())
    repo.ensure_initialized()

    if st.session_state.get("active_bundle") is None:
        recent = repo.list_runs(limit=1)
        if recent:
            st.session_state["active_bundle"] = repo.load_run_bundle(recent[0]["run_id"])

    st.markdown("<div class='ed-header-title'>Araştırma Asistanı</div>", unsafe_allow_html=True)
    st.radio(
        "Görünüm",
        options=NAV_VIEWS,
        horizontal=True,
        key="active_view",
        label_visibility="collapsed",
    )
    st.markdown("<hr class='ed-hairline'/>", unsafe_allow_html=True)

    view = st.session_state["active_view"]
    bundle: RunBundle | None = st.session_state.get("active_bundle")

    if view == "Yeni Araştırma":
        _render_new_research_screen(config, repo)
    elif view == "Rapor":
        _render_report_screen(bundle)
    elif view == "Kaynaklar":
        _render_provenance_screen(bundle)
    elif view == "Akış":
        _render_progress_screen(bundle)
    else:
        _render_history_screen(repo)


if __name__ == "__main__":
    main()
