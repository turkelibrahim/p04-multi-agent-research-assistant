"""ReportLab ile Türkçe karakterleri, tıklanabilir URL'leri, iç bağlantıları ve sayfa numaralarını destekleyen PDF dışa aktarımı."""

from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from p04_research.export.markdown_export import _format_token_summary
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

_FONT_REGISTERED_FAMILY: str | None = None
_CITE_LINK_RE = re.compile(r"\[(S\d+)\]")


def _find_font_candidates() -> list[tuple[str, Path, Path | None, Path | None]]:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    return [
        ("P04SegoeUI", windir / "segoeui.ttf", windir / "segoeuib.ttf", windir / "segoeuii.ttf"),
        ("P04Arial", windir / "arial.ttf", windir / "arialbd.ttf", windir / "ariali.ttf"),
        ("P04Calibri", windir / "calibri.ttf", windir / "calibrib.ttf", windir / "calibrii.ttf"),
        (
            "P04DejaVu",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            None,
        ),
    ]


def register_turkish_font_family() -> str:
    """Türkçe karakterleri (ç, ğ, ı, İ, ö, ş, ü) destekleyen TrueType yazı tipi ailesini kaydeder."""
    global _FONT_REGISTERED_FAMILY
    if _FONT_REGISTERED_FAMILY is not None:
        return _FONT_REGISTERED_FAMILY

    for family_name, regular_path, bold_path, italic_path in _find_font_candidates():
        if not regular_path.exists():
            continue
        reg_name = family_name
        bold_name = f"{family_name}-Bold"
        italic_name = f"{family_name}-Italic"

        pdfmetrics.registerFont(TTFont(reg_name, str(regular_path)))
        if bold_path and bold_path.exists():
            pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
        else:
            bold_name = reg_name
        if italic_path and italic_path.exists():
            pdfmetrics.registerFont(TTFont(italic_name, str(italic_path)))
        else:
            italic_name = reg_name

        registerFontFamily(
            family_name,
            normal=reg_name,
            bold=bold_name,
            italic=italic_name,
            boldItalic=bold_name,
        )
        _FONT_REGISTERED_FAMILY = family_name
        return family_name

    raise RuntimeError(
        "Türkçe karakterleri destekleyen TrueType yazı tipi (Segoe UI / Arial / Calibri / DejaVuSans) bulunamadı."
    )


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    safe = escape(text or "").replace("\n", "<br/>")
    return Paragraph(safe, style)


def _p_with_links(text: str, style: ParagraphStyle, valid_sids: set[str]) -> Paragraph:
    """Metindeki [S#] atıflarını PDF içi kaynakça çapalarına bağlar."""
    safe = escape(text or "").replace("\n", "<br/>")

    def _repl(match: re.Match) -> str:
        sid = match.group(1)
        if sid in valid_sids:
            return f'<a href="#src_{sid}" color="#1D4ED8"><b>[{sid}]</b></a>'
        return f"[{sid}]"

    linked = _CITE_LINK_RE.sub(_repl, safe)
    return Paragraph(linked, style)


def _p_url(url: str, style: ParagraphStyle) -> Paragraph:
    safe = escape(url or "")
    if url.startswith(("http://", "https://")):
        return Paragraph(f'<a href="{safe}" color="#1D4ED8">{safe}</a>', style)
    return Paragraph(safe, style)


def render_report_pdf_bytes(bundle: RunBundle) -> bytes:
    """Kayıtlı RunBundle ve FinalReport verisinden A4 akademik Türkçe PDF üretir."""
    family = register_turkish_font_family()
    bold_font = f"{family}-Bold" if f"{family}-Bold" in pdfmetrics.getRegisteredFontNames() else family
    italic_font = f"{family}-Italic" if f"{family}-Italic" in pdfmetrics.getRegisteredFontNames() else family

    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "P04Title",
        parent=base_styles["Title"],
        fontName=bold_font,
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "P04H2",
        parent=base_styles["Heading2"],
        fontName=bold_font,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "P04Body",
        parent=base_styles["BodyText"],
        fontName=family,
        fontSize=9.2,
        leading=13.2,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=4,
    )
    url_style = ParagraphStyle(
        "P04URL",
        parent=body_style,
        fontSize=8.0,
        leading=10.5,
        wordWrap="CJK",
    )
    quote_style = ParagraphStyle(
        "P04Quote",
        parent=body_style,
        fontName=italic_font,
        fontSize=8.8,
        leading=12.2,
        leftIndent=10,
        textColor=colors.HexColor("#374151"),
        spaceAfter=4,
    )
    badge_style = ParagraphStyle(
        "P04Badge",
        parent=body_style,
        fontName=bold_font,
        fontSize=9,
        textColor=colors.HexColor("#92400E"),
        backColor=colors.HexColor("#FEF3C7"),
        borderPadding=6,
        spaceAfter=6,
    )

    m = bundle.metadata
    lang = m.language.value
    rep = bundle.report
    buffer = BytesIO()
    title_text = rep.title if rep else m.question
    short_hdr = title_text if len(title_text) <= 72 else f"{title_text[:69]}..."

    def _draw_chrome(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(family, 7.5)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        # Üst bilgi
        canvas.drawString(16 * mm, A4[1] - 10 * mm, f"P04 · {short_hdr}")
        # Alt bilgi: rapor kimliği, tarih ve sayfa numarası
        page_lbl = f"Sayfa {doc_obj.page}" if lang == "tr" else f"Page {doc_obj.page}"
        canvas.drawString(16 * mm, 9 * mm, f"ID: {m.run_id} · {m.started_at}")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, page_lbl)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title=title_text,
    )

    story: list = []
    story.append(_p(title_text, title_style))

    if m.mode == RunMode.DEMO:
        story.append(_p(f"{t('mock_badge', lang)} ({t('mode_demo', lang)})", badge_style))

    mode_str = t("mode_demo", lang) if m.mode == RunMode.DEMO else t("mode_live", lang)
    depth_str = t("depth_fast", lang) if m.depth.value == "fast" else t("depth_standard", lang)
    status_str = run_status_label(m.status, lang)
    qual_str = quality_label(
        (rep.research_quality if rep else m.research_quality) or "high", lang
    )
    tokens_str = _format_token_summary(bundle)

    meta_rows = [
        [_p("Soru" if lang == "tr" else "Question", body_style), _p(m.question, body_style)],
        [_p("Kimlik / Tarih" if lang == "tr" else "Run ID / Date", body_style), _p(f"{m.run_id} · {m.started_at}", body_style)],
        [_p("Mod / Derinlik / Durum" if lang == "tr" else "Mode / Depth / Status", body_style), _p(f"{mode_str} / {depth_str} / {status_str}", body_style)],
        [_p("Araştırma Kalitesi" if lang == "tr" else "Research Quality", body_style), _p(qual_str, body_style)],
        [_p("Kullanım" if lang == "tr" else "Usage", body_style), _p(f"Model: {m.model_calls} · Arama: {m.search_calls} · Süre: {m.duration_ms} ms · Token: {tokens_str}", body_style)],
    ]
    meta_table = Table(meta_rows, colWidths=[45 * mm, 133 * mm])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 6))

    if rep is None:
        story.append(_p(f"{t('technical_error_prefix', lang)}: {m.error_message or '-'}", body_style))
        doc.build(story, onFirstPage=_draw_chrome, onLaterPages=_draw_chrome)
        return buffer.getvalue()

    if rep.error_notice:
        story.append(_p(rep.error_notice, badge_style))

    valid_sids = {b.source_id for b in rep.bibliography}

    # 1. Kısa Yanıt
    story.append(_p("Kısa Yanıt — Yönetici Özeti" if lang == "tr" else "Short Answer — Executive Summary", h2_style))
    story.append(_p_with_links(rep.short_answer, body_style, valid_sids))

    # 2. Araştırma Amacı ve Yöntem/Kapsam
    story.append(_p(t("report_heading_purpose", lang), h2_style))
    story.append(_p(rep.research_purpose or m.plan_summary or m.question, body_style))

    story.append(_p(t("report_heading_method", lang), h2_style))
    story.append(_p(rep.methodology_and_scope or m.scope_notes or "-", body_style))

    # 3. Alt Araştırma Soruları
    story.append(_p(t("report_heading_plan", lang), h2_style))
    if m.plan_summary:
        story.append(_p(m.plan_summary, body_style))
    sq_rows = [[_p("ID", body_style), _p("Durum" if lang == "tr" else "Status", body_style), _p("Alt Soru" if lang == "tr" else "Sub-question", body_style)]]
    for sq in rep.sub_questions:
        sq_rows.append([_p(sq.sq_id, body_style), _p(sq_status_label(sq.status, lang), body_style), _p(sq.text, body_style)])
    sq_table = Table(sq_rows, colWidths=[16 * mm, 36 * mm, 126 * mm], repeatRows=1, splitByRow=1)
    sq_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(sq_table)

    # 4. Temel Bulgular
    story.append(_p(t("report_heading_findings", lang), h2_style))
    if not rep.supported_findings:
        story.append(_p(t("no_supported_findings", lang), body_style))
    else:
        for item in rep.supported_findings:
            cites = " ".join(f"[{sid}]" for sid in item.citation_ids)
            story.append(
                _p_with_links(
                    f"{item.claim_id} ({item.sq_id}): {item.statement} {cites}",
                    body_style,
                    valid_sids,
                )
            )
            if item.rationale:
                story.append(_p(item.rationale, body_style))
            for q in item.quotes:
                story.append(_p_with_links(f"[{q.source_id}] ({q.quote_language}): \"{q.quote}\"", quote_style, valid_sids))
                if q.explanatory_note:
                    story.append(_p(q.explanatory_note, body_style))
            if item.single_source_warning:
                story.append(_p(f"Not: {item.single_source_warning}", body_style))

    # 5. Sayısal Veriler ve Karşılaştırmalar
    story.append(_p(t("report_heading_quant", lang), h2_style))
    if not rep.quantitative_comparisons:
        story.append(_p("Doğrulanmış sayısal karşılaştırma kaydı yoktur." if lang == "tr" else "No quantitative comparisons.", body_style))
    else:
        for q_item in rep.quantitative_comparisons:
            story.append(_p_with_links(f"• {q_item.replace('**', '')}", body_style, valid_sids))

    # 6. Çelişen Kaynaklar & Karşıt Kanıtlar
    if rep.conflicts:
        story.append(_p(t("report_heading_conflicts", lang), h2_style))
        for conf in rep.conflicts:
            story.append(Paragraph(f"<b>{escape(conf.claim_id)}</b>: {escape(conf.statement)}", body_style))
            story.append(_p(conf.explanation, body_style))
            conf_table = Table(
                [
                    [_p_with_links(f"[{conf.side_a.source_id}]", body_style, valid_sids), _p_with_links(f"[{conf.side_b.source_id}]", body_style, valid_sids)],
                    [_p(f"\"{conf.side_a.quote}\"", quote_style), _p(f"\"{conf.side_b.quote}\"", quote_style)],
                ],
                colWidths=[89 * mm, 89 * mm],
                splitByRow=1,
            )
            conf_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEF3C7")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(conf_table)

    story.append(_p(t("report_heading_counter", lang), h2_style))
    if not rep.counter_evidence_summary:
        story.append(_p("Doğrudan karşıt kanıt tespit edilmemiştir." if lang == "tr" else "No counter-evidence.", body_style))
    else:
        for ce_line in rep.counter_evidence_summary:
            story.append(_p_with_links(f"• {ce_line}", body_style, valid_sids))

    # 7. Sınırlılıklar
    story.append(_p(t("report_heading_limitations", lang), h2_style))
    if not rep.limitations:
        story.append(_p("Önemli bir kanıt boşluğu yoktur." if lang == "tr" else "No major limitations.", body_style))
    else:
        for lim in rep.limitations:
            v_lbl = verdict_label(lim.verdict, lang)
            story.append(Paragraph(f"<b>{escape(lim.claim_id)} · {escape(v_lbl)}</b>: {escape(lim.statement)}", body_style))
            if lim.initial_verdict is not None:
                init_lbl = verdict_label(lim.initial_verdict, lang)
                story.append(_p(f"1. Tur: {init_lbl} — {lim.initial_rationale or ''}", body_style))
            story.append(_p(f"Son Gerekçe: {lim.rationale}" if lang == "tr" else f"Final Rationale: {lim.rationale}", body_style))
            if lim.missing_info:
                story.append(_p(f"Eksik Bilgi: {lim.missing_info}" if lang == "tr" else f"Missing Info: {lim.missing_info}", body_style))
            for ce in lim.counter_evidence:
                story.append(_p_with_links(f"[{ce.source_id}]: \"{ce.quote}\"", quote_style, valid_sids))

    # 8. Genel Değerlendirme & Açık Sorular
    story.append(_p(t("report_heading_assessment", lang), h2_style))
    story.append(_p(rep.overall_assessment or rep.short_answer, body_style))

    story.append(_p(t("report_heading_open", lang), h2_style))
    if not rep.open_questions:
        story.append(_p(t("no_open_questions", lang), body_style))
    else:
        for oq in rep.open_questions:
            story.append(_p(f"• {oq}", body_style))

    # 9. Kaynakça (sayfa taşmasız, tıklanabilir URL ve iç çapa bağlantılı)
    story.append(_p(t("report_heading_bibliography", lang), h2_style))
    if not rep.bibliography:
        story.append(_p("Kaynak yok." if lang == "tr" else "No sources.", body_style))
    else:
        bib_rows = [[
            _p("ID", body_style),
            _p("Başlık / Yayıncı / Tür" if lang == "tr" else "Title / Publisher / Type", body_style),
            _p("Yayın / Erişim Tarihi" if lang == "tr" else "Published / Accessed", body_style),
            _p("URL", body_style),
        ]]
        for b in rep.bibliography:
            pub_date = b.published_date or t("unknown", lang)
            tag = f" [{t('mock_tag', lang)}]" if b.is_mock else ""
            ev_lbl = evidence_level_label(b.evidence_level, lang)
            s_type = source_type_label(b.source_type, lang)
            id_cell = Paragraph(f'<a name="src_{escape(b.source_id)}"/><b>{escape(b.source_id)}</b>', body_style)
            date_cell = _p(
                f"{'Yayın' if lang == 'tr' else 'Pub'}: {pub_date}\n"
                f"{'Erişim' if lang == 'tr' else 'Acc'}: {b.retrieved_at}",
                body_style,
            )
            bib_rows.append([
                id_cell,
                _p(f"{b.title}{tag}\n{b.publisher}\n{s_type} ({ev_lbl}, {b.retrieval_method})", body_style),
                date_cell,
                _p_url(b.url, url_style),
            ])
        bib_table = Table(
            bib_rows,
            colWidths=[13 * mm, 67 * mm, 44 * mm, 54 * mm],
            repeatRows=1,
            splitByRow=1,
        )
        bib_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(bib_table)

    doc.build(story, onFirstPage=_draw_chrome, onLaterPages=_draw_chrome)
    return buffer.getvalue()
