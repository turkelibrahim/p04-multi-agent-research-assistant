"""ReportLab ile Türkçe karakterleri, gizli kaynak bağlantılarını ve sayfa numaralarını destekleyen sade PDF dışa aktarımı."""

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

from p04_research.export.markdown_export import (
    _format_token_summary,
    clean_report_title,
    format_source_display_name,
    is_external_web_url,
    is_statement_duplicate_of_quote,
)
from p04_research.models import BibliographyItem, RunBundle, RunMode
from p04_research.workflow.i18n import (
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


def _p_source_cell(item: BibliographyItem, style: ParagraphStyle, lang: str) -> Paragraph:
    """Kaynak başlığının içine URL'yi gömer; açıkta URL göstermez."""
    clean_t, clean_pub = format_source_display_name(item.title, item.publisher, item.url)
    safe_t = escape(clean_t)
    safe_pub = escape(clean_pub)
    tag = f" [{escape(t('mock_tag', lang))}]" if item.is_mock else ""

    if is_external_web_url(item.url, is_mock=item.is_mock):
        safe_url = escape(item.url.strip(), {'"': "&quot;"})
        title_part = f'<a href="{safe_url}" color="#1D4ED8"><b>{safe_t}</b></a>'
    else:
        title_part = f"<b>{safe_t}</b>"

    pub_part = f" — {safe_pub}" if safe_pub and safe_pub.lower() not in clean_t.lower() else ""
    return Paragraph(f"{title_part}{pub_part}{tag}", style)


def render_report_pdf_bytes(bundle: RunBundle) -> bytes:
    """Kayıtlı RunBundle ve FinalReport verisinden sade A4 Türkçe PDF üretir."""
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
        textColor=colors.HexColor("#18181B"),
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "P04Meta",
        parent=base_styles["BodyText"],
        fontName=family,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#71717A"),
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "P04H2",
        parent=base_styles["Heading2"],
        fontName=bold_font,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#18181B"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "P04Body",
        parent=base_styles["BodyText"],
        fontName=family,
        fontSize=9.4,
        leading=13.8,
        textColor=colors.HexColor("#27272A"),
        spaceAfter=5,
    )
    quote_style = ParagraphStyle(
        "P04Quote",
        parent=body_style,
        fontName=italic_font,
        fontSize=8.9,
        leading=12.6,
        leftIndent=10,
        textColor=colors.HexColor("#52525B"),
        spaceAfter=5,
    )
    badge_style = ParagraphStyle(
        "P04Badge",
        parent=body_style,
        fontName=bold_font,
        fontSize=8.5,
        textColor=colors.HexColor("#92400E"),
        backColor=colors.HexColor("#FEF3C7"),
        borderPadding=5,
        spaceAfter=8,
    )

    m = bundle.metadata
    lang = m.language.value
    rep = bundle.report
    buffer = BytesIO()
    raw_title = rep.title if rep else m.question
    title_text = clean_report_title(raw_title)
    short_hdr = title_text if len(title_text) <= 72 else f"{title_text[:69]}..."
    date_short = (m.started_at or "")[:10]

    def _draw_chrome(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(family, 7.5)
        canvas.setFillColor(colors.HexColor("#71717A"))
        canvas.drawString(16 * mm, A4[1] - 10 * mm, short_hdr)
        page_lbl = f"Sayfa {doc_obj.page}" if lang == "tr" else f"Page {doc_obj.page}"
        canvas.drawString(16 * mm, 9 * mm, date_short)
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
    tokens_str = _format_token_summary(bundle)
    meta_line = (
        f"Tarih: {date_short} · Mod: {mode_str} ({depth_str}) · "
        f"İncelenen kaynak: {len(bundle.sources)} · Token: {tokens_str}"
        if lang == "tr"
        else f"Date: {date_short} · Mode: {mode_str} ({depth_str}) · "
        f"Sources inspected: {len(bundle.sources)} · Tokens: {tokens_str}"
    )
    story.append(_p(meta_line, meta_style))

    if rep is None:
        story.append(_p(f"{t('technical_error_prefix', lang)}: {m.error_message or '-'}", body_style))
        doc.build(story, onFirstPage=_draw_chrome, onLaterPages=_draw_chrome)
        return buffer.getvalue()

    if rep.error_notice:
        story.append(_p(rep.error_notice, badge_style))

    valid_sids = {b.source_id for b in rep.bibliography}

    # 1. Kısa Yanıt
    story.append(_p("Kısa Yanıt" if lang == "tr" else "Short Answer", h2_style))
    story.append(_p_with_links(rep.short_answer, body_style, valid_sids))

    # 2. Temel Bulgular
    story.append(_p(t("report_heading_findings", lang), h2_style))
    if not rep.supported_findings:
        story.append(_p(t("no_supported_findings", lang), body_style))
    else:
        for item in rep.supported_findings:
            cites = " ".join(f"[{sid}]" for sid in item.citation_ids)
            cite_suffix = f" {cites}" if cites else ""
            primary_q = item.quotes[0] if item.quotes else None
            is_dup = (
                primary_q is not None
                and not primary_q.explanatory_note
                and primary_q.quote_language == lang
                and is_statement_duplicate_of_quote(item.statement, primary_q.quote)
            )
            if is_dup and primary_q is not None:
                story.append(_p_with_links(f"• {primary_q.quote}{cite_suffix}", body_style, valid_sids))
            else:
                story.append(
                    _p_with_links(
                        f"• {item.statement}{cite_suffix}",
                        body_style,
                        valid_sids,
                    )
                )
                if primary_q is not None:
                    story.append(_p(f'"{primary_q.quote}"', quote_style))
                    if primary_q.explanatory_note:
                        story.append(_p(primary_q.explanatory_note, quote_style))
            if item.causation_warning:
                note_lbl = "Not" if lang == "tr" else "Note"
                story.append(_p(f"{note_lbl}: {item.causation_warning}", body_style))

    # 3. Sınırlılıklar ve Çelişen Bulgular (yalnızca varsa)
    if rep.conflicts or rep.limitations:
        sec_title = (
            "Sınırlılıklar ve Çelişen Bulgular"
            if lang == "tr"
            else "Limitations & Conflicting Evidence"
        )
        story.append(_p(sec_title, h2_style))
        for conf in rep.conflicts:
            story.append(Paragraph(f"<b>{escape(conf.statement)}</b>", body_style))
            story.append(_p(conf.explanation, body_style))
            conf_table = Table(
                [
                    [
                        _p_with_links(f"[{conf.side_a.source_id}]", body_style, valid_sids),
                        _p_with_links(f"[{conf.side_b.source_id}]", body_style, valid_sids),
                    ],
                    [
                        _p(f'"{conf.side_a.quote}"', quote_style),
                        _p(f'"{conf.side_b.quote}"', quote_style),
                    ],
                ],
                colWidths=[89 * mm, 89 * mm],
                splitByRow=1,
            )
            conf_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F4F5")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E4E4E7")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("PADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(conf_table)
            story.append(Spacer(1, 4))

        for lim in rep.limitations:
            v_lbl = verdict_label(lim.verdict, lang)
            story.append(
                Paragraph(
                    f"<b>{escape(v_lbl)}:</b> {escape(lim.statement)} — {escape(lim.rationale)}",
                    body_style,
                )
            )
            for ce in lim.counter_evidence[:1]:
                story.append(_p_with_links(f'[{ce.source_id}]: "{ce.quote}"', quote_style, valid_sids))

    # 4. Kaynakça (URL sütunu olmadan, bağlantılar kaynak adının içine gömülü)
    story.append(_p(t("report_heading_bibliography", lang), h2_style))
    if not rep.bibliography:
        story.append(_p("Kaynak kaydedilmedi." if lang == "tr" else "No sources recorded.", body_style))
    else:
        bib_rows = [[
            _p("No", body_style),
            _p("Kaynak" if lang == "tr" else "Source", body_style),
            _p("Tarih" if lang == "tr" else "Date", body_style),
        ]]
        for b in rep.bibliography:
            pub_date = b.published_date or t("unknown", lang)
            id_cell = Paragraph(f'<a name="src_{escape(b.source_id)}"/><b>{escape(b.source_id)}</b>', body_style)
            bib_rows.append([
                id_cell,
                _p_source_cell(b, body_style, lang),
                _p(pub_date, body_style),
            ])
        bib_table = Table(
            bib_rows,
            colWidths=[15 * mm, 128 * mm, 35 * mm],
            repeatRows=1,
            splitByRow=1,
        )
        bib_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F4F5")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E4E4E7")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(bib_table)

    doc.build(story, onFirstPage=_draw_chrome, onLaterPages=_draw_chrome)
    return buffer.getvalue()
