"""Kayıtlı FinalReport nesnesinden standart UTF-8 Markdown raporu üretimi (12 bölümlü akademik yapı)."""

from __future__ import annotations

from p04_research.models import RunBundle, RunMode
from p04_research.workflow.i18n import (
    evidence_level_label,
    quality_label,
    run_status_label,
    source_type_label,
    sq_status_label,
    t,
    verdict_label,
)


def _format_token_summary(bundle: RunBundle) -> str:
    m = bundle.metadata
    lang = m.language.value
    if m.total_tokens is None:
        return t("not_measured", lang)
    if m.token_measurement_type == "estimated":
        cost_part = (
            f" (~${m.estimated_cost_usd:.5f})"
            if m.estimated_cost_usd is not None
            else ""
        )
        return f"{m.total_tokens} ({t('estimated_tag', lang)}{cost_part})"
    cost_part = (
        f" (${m.estimated_cost_usd:.5f})" if m.estimated_cost_usd is not None else ""
    )
    return f"{m.total_tokens}{cost_part}"


def render_report_markdown(bundle: RunBundle) -> str:
    """Saklanmış RunBundle ve FinalReport verisinden 12 bölümlü UTF-8 Markdown raporu üretir."""
    m = bundle.metadata
    lang = m.language.value
    rep = bundle.report
    if rep is None:
        return f"# {m.question}\n\n{t('technical_error_prefix', lang)}: {m.error_message or '-'}\n"

    lines: list[str] = []
    lines.append(f"# {rep.title}")
    lines.append("")

    if m.mode == RunMode.DEMO:
        lines.append(f"> **{t('mock_badge', lang)}** ({t('mode_demo', lang)})")
        lines.append("")

    mode_str = t("mode_demo", lang) if m.mode == RunMode.DEMO else t("mode_live", lang)
    depth_str = t("depth_fast", lang) if m.depth.value == "fast" else t("depth_standard", lang)
    status_str = run_status_label(m.status, lang)
    qual_str = quality_label(rep.research_quality or m.research_quality, lang)
    tokens_str = _format_token_summary(bundle)

    if lang == "tr":
        lines.append(f"- **Soru:** {m.question}")
        lines.append(f"- **Çalışma Kimliği:** `{m.run_id}`")
        lines.append(f"- **Mod / Derinlik / Teknik Durum:** {mode_str} / {depth_str} / **{status_str}**")
        lines.append(f"- **Araştırma Kalitesi (Quality Gate):** **{qual_str}**")
        lines.append(f"- **Tarih:** {m.started_at}")
        lines.append(
            f"- **Kullanım:** Model çağrısı: {m.model_calls} · Arama çağrısı: {m.search_calls} · "
            f"Süre: {m.duration_ms} ms · Token: {tokens_str}"
        )
    else:
        lines.append(f"- **Question:** {m.question}")
        lines.append(f"- **Run ID:** `{m.run_id}`")
        lines.append(f"- **Mode / Depth / Technical Status:** {mode_str} / {depth_str} / **{status_str}**")
        lines.append(f"- **Research Quality (Quality Gate):** **{qual_str}**")
        lines.append(f"- **Date:** {m.started_at}")
        lines.append(
            f"- **Usage:** Model calls: {m.model_calls} · Search calls: {m.search_calls} · "
            f"Duration: {m.duration_ms} ms · Tokens: {tokens_str}"
        )
    lines.append("")

    if rep.error_notice:
        lines.append(f"> **!** {rep.error_notice}")
        lines.append("")

    # 1. Yönetici Özeti / Kısa Yanıt (test uyumluluğu için 'Kısa Yanıt' / 'Short Answer' başlığı korunur)
    lines.append(f"## {'Kısa Yanıt — Yönetici Özeti' if lang == 'tr' else 'Short Answer — Executive Summary'}")
    lines.append("")
    lines.append(rep.short_answer)
    lines.append("")

    # 2. Araştırma Amacı
    lines.append(f"## {t('report_heading_purpose', lang)}")
    lines.append("")
    lines.append(rep.research_purpose or m.plan_summary or m.question)
    lines.append("")

    # 3. Yöntem ve Kapsam
    lines.append(f"## {t('report_heading_method', lang)}")
    lines.append("")
    lines.append(rep.methodology_and_scope or m.scope_notes or "-")
    lines.append("")

    # 4. Alt Araştırma Soruları
    lines.append(f"## {t('report_heading_plan', lang)}")
    lines.append("")
    if m.plan_summary:
        lines.append(m.plan_summary)
        lines.append("")
    for sq in rep.sub_questions:
        st_lbl = sq_status_label(sq.status, lang)
        lines.append(f"- **{sq.sq_id}** [{st_lbl}] — {sq.text}")
        if sq.scope_note:
            lines.append(f"  - _{sq.scope_note}_")
        if sq.search_intent:
            lines.append(f"  - {'Arama amacı' if lang == 'tr' else 'Search intent'}: {sq.search_intent}")
    lines.append("")

    # 5. Temel Bulgular
    lines.append(f"## {t('report_heading_findings', lang)}")
    lines.append("")
    if not rep.supported_findings:
        lines.append(t("no_supported_findings", lang))
        lines.append("")
    else:
        for item in rep.supported_findings:
            cites = " ".join(f"[{sid}]" for sid in item.citation_ids)
            lines.append(f"### {item.claim_id} ({item.sq_id}) — {item.statement} {cites}")
            if item.rationale:
                lines.append(f"- **Gerekçe:** {item.rationale}" if lang == "tr" else f"- **Rationale:** {item.rationale}")
            if item.numeric_context:
                lines.append(f"- **Sayısal Bağlam:** {item.numeric_context}" if lang == "tr" else f"- **Numeric Context:** {item.numeric_context}")
            for q in item.quotes:
                lines.append(f"  - [{q.source_id}] ({q.quote_language}): \"{q.quote}\"")
                if q.explanatory_note:
                    lines.append(f"    - _{q.explanatory_note}_")
            if item.single_source_warning:
                lines.append(f"- *Not:* {item.single_source_warning}")
            if item.causation_warning:
                lines.append(f"- **!** {item.causation_warning}")
            lines.append("")

    # 6. Sayısal Veriler ve Karşılaştırmalar
    lines.append(f"## {t('report_heading_quant', lang)}")
    lines.append("")
    if not rep.quantitative_comparisons:
        lines.append(
            "Doğrulanmış sayısal karşılaştırma kaydı bulunmamaktadır."
            if lang == "tr"
            else "No verified quantitative comparisons recorded."
        )
    else:
        for q_item in rep.quantitative_comparisons:
            lines.append(f"- {q_item}")
    lines.append("")

    # 7. Çelişen Kaynaklar ve Karşıt Kanıtlar
    if rep.conflicts:
        lines.append(f"## {t('report_heading_conflicts', lang)}")
        lines.append("")
        for conf in rep.conflicts:
            lines.append(f"### {conf.claim_id} ({conf.sq_id}) — {conf.statement}")
            lines.append(conf.explanation)
            lines.append(f"- **[{conf.side_a.source_id}]:** \"{conf.side_a.quote}\"")
            lines.append(f"- **[{conf.side_b.source_id}]:** \"{conf.side_b.quote}\"")
            lines.append("")

    lines.append(f"## {t('report_heading_counter', lang)}")
    lines.append("")
    if not rep.counter_evidence_summary:
        lines.append(
            "Doğrudan karşıt kanıt tespit edilmemiştir."
            if lang == "tr"
            else "No direct counter-evidence detected."
        )
    else:
        for ce_line in rep.counter_evidence_summary:
            lines.append(f"- {ce_line}")
    lines.append("")

    # 8. Sınırlılıklar
    lines.append(f"## {t('report_heading_limitations', lang)}")
    lines.append("")
    if not rep.limitations:
        lines.append(
            "Önemli bir kanıt boşluğu veya çürütülen iddia bulunmamaktadır."
            if lang == "tr"
            else "No major evidence gaps or contradicted claims remain."
        )
        lines.append("")
    else:
        for lim in rep.limitations:
            v_lbl = verdict_label(lim.verdict, lang)
            lines.append(f"### {lim.claim_id} ({lim.sq_id}) · **{v_lbl}** — {lim.statement}")
            if lim.initial_verdict is not None:
                init_lbl = verdict_label(lim.initial_verdict, lang)
                lines.append(
                    f"- **1. Tur İlk Karar:** {init_lbl} — {lim.initial_rationale or ''}"
                    if lang == "tr"
                    else f"- **Round 1 Initial Verdict:** {init_lbl} — {lim.initial_rationale or ''}"
                )
            lines.append(
                f"- **Son Gerekçe:** {lim.rationale}"
                if lang == "tr"
                else f"- **Final Rationale:** {lim.rationale}"
            )
            if lim.missing_info:
                lines.append(
                    f"- **Eksik Bilgi:** {lim.missing_info}"
                    if lang == "tr"
                    else f"- **Missing Info:** {lim.missing_info}"
                )
            for ce in lim.counter_evidence:
                lines.append(f"  - [{ce.source_id}]: \"{ce.quote}\"")
            lines.append("")

    # 9. Genel Değerlendirme
    lines.append(f"## {t('report_heading_assessment', lang)}")
    lines.append("")
    lines.append(rep.overall_assessment or rep.short_answer)
    if rep.quality_notes:
        lines.append("")
        for qn in rep.quality_notes:
            lines.append(f"- **Kalite Notu:** {qn}")
    lines.append("")

    # 10. Açık Sorular
    lines.append(f"## {t('report_heading_open', lang)}")
    lines.append("")
    if not rep.open_questions:
        lines.append(t("no_open_questions", lang))
    else:
        for oq in rep.open_questions:
            lines.append(f"- {oq}")
    lines.append("")

    # 11. Kaynakça
    lines.append(f"## {t('report_heading_bibliography', lang)}")
    lines.append("")
    if not rep.bibliography:
        lines.append("Kaynak kaydedilmedi." if lang == "tr" else "No sources recorded.")
    else:
        for b in rep.bibliography:
            pub_date = b.published_date or t("unknown", lang)
            ev_lvl = evidence_level_label(b.evidence_level, lang)
            s_type = source_type_label(b.source_type, lang)
            mock_str = f" · [{t('mock_tag', lang)}]" if b.is_mock else ""
            dup_str = (
                f" · (Kopya -> {b.duplicate_of_source_id})"
                if b.duplicate_of_source_id and lang == "tr"
                else (f" · (Duplicate of {b.duplicate_of_source_id})" if b.duplicate_of_source_id else "")
            )
            acc_str = "" if b.is_accessible else (" · [Erişilemedi]" if lang == "tr" else " · [Inaccessible]")
            author_str = f" · Yazar: {b.author}" if b.author else ""
            lines.append(
                f"- **[{b.source_id}] {b.title}**{mock_str}{dup_str}{acc_str} — {b.publisher}{author_str} | "
                f"{s_type} ({ev_lvl}, {b.retrieval_method}) | "
                f"{'Yayın Tarihi' if lang == 'tr' else 'Published'}: {pub_date} | "
                f"{'Erişim Tarihi' if lang == 'tr' else 'Accessed'}: {b.retrieved_at} | {b.url}"
            )
    lines.append("")
    return "\n".join(lines)
