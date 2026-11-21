"""P04 · Çok Ajanlı Araştırma Asistanı — Sadeleştirilmiş Streamlit Arayüzü."""

from __future__ import annotations

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
from p04_research.export.markdown_export import render_report_markdown  # noqa: E402
from p04_research.export.pdf_export import render_report_pdf_bytes  # noqa: E402
from p04_research.models import (  # noqa: E402
    ClaimProvenanceView,
    RunBundle,
    RunMode,
    RunStatus,
    VerdictStatus,
)
from p04_research.providers.mock_demo import DEMO_PRESETS  # noqa: E402
from p04_research.workflow.graph import execute_research_run  # noqa: E402
from p04_research.workflow.i18n import (  # noqa: E402
    evidence_level_label,
    quality_label,
    run_status_label,
    source_type_label,
    sq_status_label,
    t,
    verdict_label,
)

NAV_VIEWS = [
    "Yeni Araştırma",
    "Rapor",
    "Kanıt ve Kaynak Ayrıntısı",
    "İlerleme ve Ajan Akışı",
    "Geçmiş ve Dışa Aktarım",
]

_NODE_DISPLAY = {
    "planner": "1 · Planlayıcı",
    "researcher": "2 · Web Araştırmacısı",
    "analyst": "3 · Analist",
    "verifier": "4 · Kanıt Denetçisi",
    "router": "Yönlendirici (2. Tur Kararı)",
    "reporter": "5 · Raportör",
    "quality_gate": "6 · Kalite Geçidi (Quality Gate)",
}


def _apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        .p04-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.84rem;
            font-weight: 600;
            border: 1px solid #cbd5e1;
            margin-right: 6px;
        }
        .p04-badge-mock { background-color: #fef3c7; color: #92400e; border-color: #f59e0b; }
        .p04-badge-supported { background-color: #dcfce7; color: #166534; border-color: #22c55e; }
        .p04-badge-contradicted { background-color: #fee2e2; color: #991b1b; border-color: #ef4444; }
        .p04-badge-conflicting { background-color: #ffedd5; color: #9a3412; border-color: #f97316; }
        .p04-badge-insufficient { background-color: #e2e8f0; color: #1e293b; border-color: #64748b; }
        .p04-badge-irrelevant { background-color: #f3e8ff; color: #6b21a8; border-color: #a855f7; }
        .p04-card {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .p04-summary-box {
            background: #f0fdf4;
            border: 1px solid #86efac;
            border-left: 5px solid #16a34a;
            border-radius: 10px;
            padding: 16px 18px;
            font-size: 1.04rem;
            line-height: 1.55;
            margin-bottom: 16px;
        }
        .p04-quote {
            border-left: 4px solid #2563eb;
            background: #f8fafc;
            padding: 10px 14px;
            margin: 8px 0;
            border-radius: 0 6px 6px 0;
            font-style: italic;
        }
        *:focus-visible {
            outline: 3px solid #1d4ed8 !important;
            outline-offset: 2px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _verdict_css_class(verdict: VerdictStatus) -> str:
    return {
        VerdictStatus.SUPPORTED: "p04-badge-supported",
        VerdictStatus.CONTRADICTED: "p04-badge-contradicted",
        VerdictStatus.CONFLICTING: "p04-badge-conflicting",
        VerdictStatus.INSUFFICIENT: "p04-badge-insufficient",
        VerdictStatus.IRRELEVANT: "p04-badge-irrelevant",
    }.get(verdict, "p04-badge-insufficient")


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
    st.session_state["pending_view"] = "Kanıt ve Kaynak Ayrıntısı"
    st.session_state["pending_source_id"] = source_id
    st.rerun()


def _jump_to_claim(claim_id: str) -> None:
    st.session_state["pending_view"] = "Kanıt ve Kaynak Ayrıntısı"
    st.session_state["pending_claim_id"] = claim_id
    st.rerun()


def _render_download_buttons(bundle: RunBundle, key_prefix: str) -> None:
    lang = bundle.metadata.language.value
    md_text = (
        bundle.report.report_markdown
        if bundle.report and bundle.report.report_markdown
        else render_report_markdown(bundle)
    )
    pdf_bytes = render_report_pdf_bytes(bundle)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Markdown (.md) İndir" if lang == "tr" else "Download Markdown (.md)",
            data=md_text.encode("utf-8"),
            file_name=f"p04_rapor_{bundle.metadata.run_id}.md",
            mime="text/markdown; charset=utf-8",
            key=f"{key_prefix}_dl_md",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="PDF (.pdf) İndir" if lang == "tr" else "Download PDF (.pdf)",
            data=pdf_bytes,
            file_name=f"p04_rapor_{bundle.metadata.run_id}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_pdf",
            use_container_width=True,
        )


def _render_new_research_screen(config, repo: ResearchRepository) -> None:
    import os

    st.header("Yeni Araştırma Başlat")

    mode_options = [
        "Çevrimdışı Demo (Kurmaca kanıt akışı — internet ve anahtar gerekmez)",
        "Canlı Araştırma (Gerçek Web / Tavily + Dil Modeli)",
    ]
    if "pending_mode_radio" in st.session_state:
        st.session_state["mode_radio"] = st.session_state.pop("pending_mode_radio")
    elif "mode_radio" not in st.session_state:
        force_demo = os.environ.get("P04_FORCE_DEMO_DEFAULT") == "1"
        st.session_state["mode_radio"] = (
            mode_options[1] if (config.has_tavily_key() and not force_demo) else mode_options[0]
        )

    preset_labels = {p.label_tr: p for p in DEMO_PRESETS}
    with st.expander("Okul Sunumu İçin Hazır Demo Senaryoları (5 Senaryo)", expanded=False):
        col_pre, col_btn = st.columns([4, 1])
        with col_pre:
            chosen_label = st.selectbox(
                "Hazır Demo Senaryosu",
                options=list(preset_labels.keys()),
                key="preset_select",
            )
        chosen_preset = preset_labels[chosen_label]
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Örneği Forma Doldur", use_container_width=True):
                st.session_state["pending_question"] = chosen_preset.default_question_tr
                st.session_state["pending_mode_radio"] = mode_options[0]
                st.rerun()
        st.caption(chosen_preset.description_tr)

    mode_label = st.radio(
        "Çalışma Modu",
        options=mode_options,
        horizontal=True,
        key="mode_radio",
    )
    selected_mode = "live" if mode_label.startswith("Canlı") else "demo"

    if selected_mode == "live":
        openai_ok = config.has_openai_key()
        tavily_ok = config.has_tavily_key()
        st.info(
            f"**Canlı Web Araştırması Seçili:** Arama sorguları Tavily Search API ile gerçek web üzerinde taranır "
            f"(Tavily Anahtarı: **{'Hazır' if tavily_ok else 'Eksik'}**, OpenAI Anahtarı: **{'Hazır' if openai_ok else 'Eksik'}**). "
            f"OpenAI hesap bakiyesi/kotası yetersiz olsa bile Tavily'den gelen gerçek web kaynakları yerel kanıt çıkarıcı ile birebir alıntılanarak raporlanır."
        )
    else:
        st.markdown(
            "<span class='p04-badge p04-badge-mock'>Kurmaca eğitim verisi</span> "
            "Çevrimdışı demo modunda yazdığınız her soru için 2 turlu örnek kanıt ve karşı kanıt zinciri üretilir.",
            unsafe_allow_html=True,
        )

    with st.form("research_form", clear_on_submit=False):
        question_val = st.text_area(
            "Araştırma Sorunuz (10–2.000 karakter)",
            key="question_input",
            height=95,
            placeholder="Örn: 2026 Yılında Yapay Zekâ Ajanlarının İş Dünyasına Etkisi: İnsan Çalışanların Yerini Alabilirler mi?",
        )
        col_lang, col_depth = st.columns(2)
        with col_lang:
            lang_choice = st.radio(
                "Rapor Dili",
                options=["Türkçe (tr)", "English (en)"],
                horizontal=True,
                key="lang_radio",
            )
        with col_depth:
            depth_choice = st.radio(
                "Araştırma Derinliği",
                options=[
                    "Standart (Eksik kanıtta 2. tur takip araması yapar)",
                    "Hızlı (Tek tur, 1 arama)",
                ],
                horizontal=True,
                key="depth_radio",
            )
        submitted = st.form_submit_button("Araştırmayı Başlat", type="primary", use_container_width=True)

    if submitted:
        lang_code = "en" if lang_choice.startswith("English") else "tr"
        depth_code = "fast" if depth_choice.startswith("Hızlı") else "standard"
        ok, cleaned_q, err_msg = validate_question(question_val, lang_code)
        if not ok:
            st.error(err_msg)
            return

        with st.status("5 düğümlü araştırma akışı yürütülüyor...", expanded=True) as status_box:
            def _on_ev(ev):
                if ev.status in ("finish", "decision", "error"):
                    lbl = _NODE_DISPLAY.get(ev.node_name, ev.node_name)
                    st.write(f"**{lbl}** (Tur {ev.round_no}): {ev.message}")

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
                label=f"Tamamlandı: {run_status_label(bundle.metadata.status, lang_code)}",
                state="complete" if bundle.metadata.status != RunStatus.FAILED else "error",
            )

        st.session_state["active_bundle"] = bundle
        st.session_state["active_persist_error"] = None if persisted_ok else persist_err
        st.session_state["form_nonce"] = uuid.uuid4().hex
        st.session_state["selected_claim_id"] = None
        st.session_state["selected_source_id"] = None
        st.session_state["pending_view"] = "Rapor"
        st.rerun()


def _render_report_screen(bundle: RunBundle | None) -> None:
    if bundle is None or bundle.report is None:
        st.info("Görüntülenecek bir rapor yok. 'Yeni Araştırma' sekmesinden bir soru başlatın.")
        return

    persist_err = st.session_state.get("active_persist_error")
    if persist_err:
        st.error(f"Rapor bu oturumda hazırlandı ancak veritabanına yazılamadı: {persist_err}")

    m = bundle.metadata
    rep = bundle.report
    lang = m.language.value

    # Başlık ve Durum Rozetleri
    badge_html = (
        f"<span class='p04-badge p04-badge-mock'>{t('mock_badge', lang)}</span>"
        if m.mode == RunMode.DEMO
        else f"<span class='p04-badge p04-badge-supported'>{t('live_tag', lang)}</span>"
    )
    qual_val = (rep.research_quality or m.research_quality or "high")
    st.markdown(
        f"{badge_html} "
        f"<span class='p04-badge p04-badge-insufficient'>Teknik: {run_status_label(m.status, lang)}</span> "
        f"<span class='p04-badge p04-badge-supported'>Kalite: {quality_label(qual_val, lang)}</span>",
        unsafe_allow_html=True,
    )
    st.subheader(rep.title)

    # Sade Özet Metrikleri
    rounds_count = max((sr.round_no for sr in bundle.search_rounds), default=1)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Doğrulanan Bulgu", len(rep.supported_findings))
    k2.metric("İncelenen Kaynak", len(bundle.sources))
    k3.metric("Arama Turu", f"{rounds_count} Tur")
    k4.metric("Çürütülen / Eksik İddia", len(rep.limitations) + len(rep.conflicts))

    if rep.error_notice:
        st.error(rep.error_notice)

    _render_download_buttons(bundle, key_prefix=f"rep_{m.run_id}")
    st.divider()

    # 1. Kısa Yanıt (En Üstte Net Özet)
    st.markdown(f"### 1. {t('report_heading_short', lang)}")
    st.markdown(f"<div class='p04-summary-box'>{rep.short_answer}</div>", unsafe_allow_html=True)

    # 2. Doğrulanan Bulgular ve Kanıt Alıntıları
    st.markdown(f"### 2. {t('report_heading_findings', lang)}")
    if not rep.supported_findings:
        st.warning(t("no_supported_findings", lang))
    else:
        src_map = {s.source_id: s for s in bundle.sources}
        for idx, item in enumerate(rep.supported_findings):
            cites_str = " ".join(f"[{sid}]" for sid in item.citation_ids)
            st.markdown(
                f"<div class='p04-card'>"
                f"<span class='p04-badge p04-badge-supported'>Destekleniyor</span> "
                f"<small style='color:#475569'>İddia {item.claim_id} · Alt Soru {item.sq_id}</small><br/>"
                f"<strong style='font-size:1.05rem'>{item.statement} {cites_str}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )
            for q in item.quotes:
                src_obj = src_map.get(q.source_id)
                src_title = src_obj.title if src_obj else q.source_id
                pub_date = (src_obj.published_date or "Tarih Bilinmiyor") if src_obj else ""
                st.markdown(
                    f"<div class='p04-quote'><strong>[{q.source_id}] {src_title}</strong> ({pub_date}):<br/>"
                    f"\"{q.quote}\"</div>",
                    unsafe_allow_html=True,
                )
                if q.explanatory_note:
                    st.caption(q.explanatory_note)

            btn_cols = st.columns(len(item.citation_ids) + 1)
            for c_idx, sid in enumerate(item.citation_ids):
                with btn_cols[c_idx]:
                    if st.button(f"[{sid}] Kaynağını Aç", key=f"jump_src_{idx}_{sid}"):
                        _jump_to_source(sid)
            with btn_cols[-1]:
                if st.button(f"{item.claim_id} Kanıt Zinciri", key=f"jump_clm_sup_{item.claim_id}"):
                    _jump_to_claim(item.claim_id)
            st.write("")

    # 3. Çelişen Kaynaklar (Varsa)
    if rep.conflicts:
        st.markdown(f"### 3. {t('report_heading_conflicts', lang)}")
        for conf in rep.conflicts:
            st.warning(f"**{conf.claim_id}**: {conf.statement} — {conf.explanation}")
            ca, cb = st.columns(2)
            with ca:
                st.markdown(
                    f"<div class='p04-quote'><strong>[{conf.side_a.source_id}]</strong>: \"{conf.side_a.quote}\"</div>",
                    unsafe_allow_html=True,
                )
                if st.button(f"[{conf.side_a.source_id}] Kartını Aç", key=f"conf_a_{conf.claim_id}_{conf.side_a.source_id}"):
                    _jump_to_source(conf.side_a.source_id)
            with cb:
                st.markdown(
                    f"<div class='p04-quote'><strong>[{conf.side_b.source_id}]</strong>: \"{conf.side_b.quote}\"</div>",
                    unsafe_allow_html=True,
                )
                if st.button(f"[{conf.side_b.source_id}] Kartını Aç", key=f"conf_b_{conf.claim_id}_{conf.side_b.source_id}"):
                    _jump_to_source(conf.side_b.source_id)

    # 4. Karşı Kanıt ve Sınırlılıklar (Denetçinin Elendiği İddialar)
    if rep.limitations:
        st.markdown(f"### 3. {t('report_heading_limitations', lang)} (Denetçi Kararları)")
        for lim in rep.limitations:
            v_lbl = verdict_label(lim.verdict, lang)
            css_cls = _verdict_css_class(lim.verdict)
            transition_html = ""
            if lim.initial_verdict is not None:
                init_lbl = verdict_label(lim.initial_verdict, lang)
                transition_html = (
                    f"<div style='margin-top:8px;padding:8px 12px;background:#fff7ed;border-left:4px solid #f97316;'>"
                    f"<strong>1. Tur İlk Karar ({init_lbl}):</strong> {lim.initial_rationale or ''}</div>"
                )
            st.markdown(
                f"<div class='p04-card'>"
                f"<span class='p04-badge {css_cls}'>{v_lbl}</span> "
                f"<strong>{lim.claim_id}: {lim.statement}</strong>"
                f"{transition_html}"
                f"<div style='margin-top:8px'><strong>Son Gerekçe:</strong> {lim.rationale}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if lim.missing_info:
                st.info(f"Eksik bilgi: {lim.missing_info}")
            for ce in lim.counter_evidence:
                st.markdown(
                    f"<div class='p04-quote'><strong>Karşı Kanıt [{ce.source_id}]</strong>: \"{ce.quote}\"</div>",
                    unsafe_allow_html=True,
                )
            cols = st.columns(max(2, len(lim.counter_evidence) + 1))
            with cols[0]:
                if st.button(f"{lim.claim_id} Kanıt Zincirini İncele", key=f"lim_clm_{lim.claim_id}"):
                    _jump_to_claim(lim.claim_id)
            for idx_ce, ce in enumerate(lim.counter_evidence):
                with cols[idx_ce + 1]:
                    if st.button(f"[{ce.source_id}] Kaynağını Aç", key=f"lim_src_{lim.claim_id}_{ce.source_id}_{idx_ce}"):
                        _jump_to_source(ce.source_id)

    # 5. Kaynakça ve Araştırma Planı
    st.divider()
    col_plan, col_bib = st.columns([1, 1])
    with col_plan:
        st.markdown(f"#### {t('report_heading_plan', lang)}")
        for sq in rep.sub_questions:
            st.markdown(f"- **{sq.sq_id}** `[{sq_status_label(sq.status, lang)}]` — {sq.text}")
            if sq.findings:
                st.caption(f"Bulgular: {len(sq.findings)} · Eksikler: {len(sq.remaining_gaps)}")
        if rep.open_questions:
            st.markdown(f"**{t('report_heading_open', lang)}:**")
            for oq in rep.open_questions:
                st.markdown(f"- {oq}")

    with col_bib:
        st.markdown(f"#### {t('report_heading_bibliography', lang)}")
        if not rep.bibliography:
            st.write("Kaynak kaydedilmedi.")
        else:
            for b in rep.bibliography:
                pub_date = b.published_date or t("unknown", lang)
                tag = t("mock_tag", lang) if b.is_mock else t("live_tag", lang)
                s_type = source_type_label(b.source_type, lang)
                c_txt, c_btn = st.columns([4, 1])
                with c_txt:
                    st.markdown(
                        f"**[{b.source_id}] {b.title}** (`{tag}`)  \n"
                        f"<small>{b.publisher} · {s_type} ({b.retrieval_method}) · Yayın: {pub_date}</small>",
                        unsafe_allow_html=True,
                    )
                with c_btn:
                    if st.button(f"{b.source_id} Aç", key=f"bib_open_{b.source_id}"):
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
    st.header("Kanıt ve Kaynak Ayrıntısı")
    if bundle is None:
        st.info("İncelenecek çalışma yok. Önce bir araştırma çalıştırın veya Geçmişten yükleyin.")
        return

    lang = bundle.metadata.language.value

    st.subheader("İddia Kanıt Zinciri (1. Tur İlk Dayanak -> 2. Tur Sorgusu -> Son Karar)")
    if not bundle.claims:
        st.info("Bu çalışmada çıkarılmış iddia bulunmuyor.")
    else:
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
                st.markdown("#### 1. Analistin İlk Dayanağı (1. Tur)")
                st.write(f"**Alt Soru (`{prov.claim.sq_id}`):** {prov.sub_question.text if prov.sub_question else '-'}")
                st.write(f"**İlk Kaynak:** `[{prov.claim.initial_source_id}]` · Dil: `{prov.claim.initial_quote_language}`")
                st.markdown(
                    f"<div class='p04-quote'>\"{prov.claim.initial_quote}\"</div>",
                    unsafe_allow_html=True,
                )
                if prov.claim.explanatory_note:
                    st.caption(prov.claim.explanatory_note)
                if prov.round1_verification:
                    st.markdown(
                        f"**1. Tur Denetçi Kararı:** `{verdict_label(prov.round1_verification.verdict, lang)}`  \n"
                        f"{prov.round1_verification.rationale}"
                    )
                    if prov.round1_verification.missing_info:
                        st.info(f"1. Tur Eksik Bilgi: {prov.round1_verification.missing_info}")

            with col_b:
                st.markdown("#### 2. Takip Araması ve Son Karar")
                if prov.follow_up_search_round:
                    st.warning(
                        f"**2. Tur Takip Sorgusu:** `{prov.follow_up_search_round.query}`  \n"
                        f"Neden: {prov.follow_up_search_round.trigger_reason or '-'}"
                    )
                if prov.final_verification:
                    fv = prov.final_verification
                    st.markdown(
                        f"<span class='p04-badge {_verdict_css_class(fv.verdict)}'>{verdict_label(fv.verdict, lang)}</span> "
                        f"(Tur {fv.round_no})",
                        unsafe_allow_html=True,
                    )
                    st.write(fv.rationale)
                    if fv.causation_warning:
                        st.warning(fv.causation_warning)
                    for ev in fv.evidence_items:
                        st.markdown(
                            f"<div class='p04-quote'><strong>[{ev.source_id}]</strong> ({ev.role.value}): \"{ev.quote}\"</div>",
                            unsafe_allow_html=True,
                        )

    st.divider()
    st.subheader("Kayıtlı Kaynak Kartları")
    if not bundle.sources:
        st.info("Bu çalışmada kaydedilmiş kaynak bulunmuyor.")
        return

    selected_sid = st.session_state.get("selected_source_id")
    ordered_sources = list(bundle.sources)
    if selected_sid:
        ordered_sources.sort(key=lambda s: (0 if s.source_id == selected_sid else 1, s.source_id))

    for src in ordered_sources:
        is_selected = src.source_id == selected_sid
        pub_date = src.published_date or t("unknown", lang)
        tag = t("mock_tag", lang) if src.is_mock else t("live_tag", lang)
        header = f"[{src.source_id}] {src.title} · ({tag})"
        with st.expander(header, expanded=is_selected or len(ordered_sources) <= 3):
            author_line = f" · **Yazar:** {src.author}" if src.author else ""
            st.markdown(
                f"- **Yayıncı:** {src.publisher}{author_line} · **Yayın Tarihi:** {pub_date} · **Erişim:** {src.retrieved_at}\n"
                f"- **Kaynak Türü:** {source_type_label(src.source_type, lang)} · **Alıntı Yöntemi:** `{src.retrieval_method}`\n"
                f"- **Güvenilirlik:** `{src.credibility_score:.2f}` · **Konu Uygunluğu:** `{src.relevance_score:.2f}` ({src.relevance_assessment})\n"
                f"- **Dil:** `{src.language}` · **Kanıt Düzeyi:** {evidence_level_label(src.evidence_level, lang)}"
            )
            if not src.is_accessible:
                st.error("Bu kaynağın sayfasına erişilemedi veya içerik okunamadı; kanıt olarak kullanılmadı.")
            if src.duplicate_of_source_id:
                st.warning(
                    f"Bu kaynak içerik parmak izi açısından `{src.duplicate_of_source_id}` ile kopyadır; bağımsız ikinci kanıt sayılmaz."
                )
            if src.is_mock:
                st.caption(f"Kurmaca eğitim kaynağı adresi: `{src.url}`")
            else:
                st.markdown(f"**Özgün Web Sayfası:** [{src.url}]({src.url})")

            src_chunks = [ch for ch in bundle.chunks if ch.source_id == src.source_id]
            for ch in src_chunks:
                st.markdown(f"**Alıntıya Dayanak Metin (`{ch.chunk_id}`, Tur {ch.round_no}):**")
                st.markdown(f"<div class='p04-quote'>{ch.excerpt_text}</div>", unsafe_allow_html=True)


def _render_progress_screen(bundle: RunBundle | None) -> None:
    st.header("İlerleme ve Ajan Akışı")
    if bundle is None:
        st.info("Henüz aktif bir çalışma yok.")
        return

    m = bundle.metadata
    lang = m.language.value
    if m.total_tokens is None:
        tokens_str = t("not_measured", lang)
    elif m.token_measurement_type == "estimated":
        tokens_str = f"{m.total_tokens} (Tahmini)"
    else:
        tokens_str = str(m.total_tokens)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Durum", run_status_label(m.status, lang))
    c2.metric("Benzersiz Kaynak", len(bundle.sources))
    c3.metric("Arama Çağrısı", m.search_calls)
    c4.metric("Model Çağrısı", m.model_calls)
    c5.metric("Token", tokens_str)

    follow_ups = [sr for sr in bundle.search_rounds if sr.round_no > 1]
    if follow_ups:
        for fu in follow_ups:
            st.warning(
                f"**2. Tur Takip Araması Tetiklendi** · İddia: `{fu.triggered_by_claim_id or '-'}` | "
                f"Neden: {fu.trigger_reason or '-'} | Sorgu: `{fu.query}` | Yeni Kaynak: {', '.join(fu.source_ids) or '0'}"
            )

    st.subheader("Beş Düğümün Adım Özeti")
    finish_events = [e for e in bundle.events if e.status in ("finish", "decision", "error")]
    for ev in finish_events:
        lbl = _NODE_DISPLAY.get(ev.node_name, ev.node_name)
        st.markdown(
            f"<div class='p04-card'><strong>{lbl}</strong> · Tur {ev.round_no} · "
            f"{ev.message}<br/><small style='color:#475569'>{ev.payload_summary}</small></div>",
            unsafe_allow_html=True,
        )


def _render_history_screen(repo: ResearchRepository) -> None:
    st.header("Geçmiş Çalışmalar ve Dışa Aktarım")
    runs = repo.list_runs(limit=100)
    if not runs:
        st.info("Veritabanında henüz kayıtlı çalışma bulunmuyor.")
        return

    active_bundle: RunBundle | None = st.session_state.get("active_bundle")
    if active_bundle is not None:
        st.subheader(f"Aktif Raporu İndir ({active_bundle.metadata.run_id})")
        _render_download_buttons(active_bundle, key_prefix=f"hist_active_{active_bundle.metadata.run_id}")
        st.divider()

    for r in runs:
        lang = r["language"]
        status_enum = RunStatus(r["status"])
        mode_lbl = t("mode_demo", lang) if r["mode"] == "demo" else t("mode_live", lang)
        depth_lbl = t("depth_fast", lang) if r["depth"] == "fast" else t("depth_standard", lang)
        col_info, col_act = st.columns([5, 1])
        with col_info:
            st.markdown(
                f"**{r['question']}**  \n"
                f"<small>{r['started_at']} · {mode_lbl} / {depth_lbl} · "
                f"**{run_status_label(status_enum, lang)}** · "
                f"Destekli bulgu: {r['supported_claim_count']} · Kaynak: {r['source_count']}</small>",
                unsafe_allow_html=True,
            )
        with col_act:
            if st.button("Raporu Aç", key=f"open_hist_{r['run_id']}", use_container_width=True):
                loaded = repo.load_run_bundle(r["run_id"])
                if loaded is not None:
                    st.session_state["active_bundle"] = loaded
                    st.session_state["active_persist_error"] = None
                    st.session_state["selected_source_id"] = None
                    st.session_state["selected_claim_id"] = None
                    st.session_state["pending_view"] = "Rapor"
                    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="P04 · Çok Ajanlı Araştırma Asistanı",
        page_icon="🔎",
        layout="wide",
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

    st.title("P04 · Çok Ajanlı Araştırma Asistanı")
    st.radio(
        "Ekran Seçimi",
        options=NAV_VIEWS,
        horizontal=True,
        key="active_view",
        label_visibility="collapsed",
    )
    st.divider()

    view = st.session_state["active_view"]
    bundle: RunBundle | None = st.session_state.get("active_bundle")

    if view == "Yeni Araştırma":
        _render_new_research_screen(config, repo)
    elif view == "Rapor":
        _render_report_screen(bundle)
    elif view == "Kanıt ve Kaynak Ayrıntısı":
        _render_provenance_screen(bundle)
    elif view == "İlerleme ve Ajan Akışı":
        _render_progress_screen(bundle)
    else:
        _render_history_screen(repo)


if __name__ == "__main__":
    main()
