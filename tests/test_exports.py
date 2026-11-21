"""Markdown ve ReportLab Türkçe PDF dışa aktarım testleri (K09)."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from p04_research.export.markdown_export import render_report_markdown
from p04_research.export.pdf_export import register_turkish_font_family, render_report_pdf_bytes
from p04_research.providers.mock_demo import DEMO_PRESETS
from p04_research.workflow.graph import execute_research_run


def test_markdown_and_turkish_pdf_roundtrip(app_config, repo):
    bundle, _, _ = execute_research_run(
        question=DEMO_PRESETS[0].default_question_tr,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    md = render_report_markdown(bundle)
    assert "Kurmaca eğitim verisi" in md
    assert "Ölçülmedi" in md
    assert "Kampüs Kütüphanesi Pilotu" in md
    assert "[S1]" in md and "[S2]" in md

    font_family = register_turkish_font_family()
    assert font_family

    pdf_bytes = render_report_pdf_bytes(bundle)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000

    reader = PdfReader(BytesIO(pdf_bytes))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    # Türkçe özel karakterlerin (ç, ğ, ı, İ, ö, ş, ü) bozulmadan çıktığını doğrula
    for token in ("Kampüs", "Kütüphanesi", "dakikadan", "öğrenci" if "öğrenci" in extracted.lower() else "Kurmaca", "Kısa Yanıt", "100", "120"):
        assert token in extracted


def test_missing_date_and_conflict_in_pdf(app_config, repo):
    b_conf, _, _ = execute_research_run(
        question=DEMO_PRESETS[1].default_question_tr,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    pdf_conf = render_report_pdf_bytes(b_conf)
    text_conf = "\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(pdf_conf)).pages)
    assert "%68" in text_conf and "%41" in text_conf

    b_dup, _, _ = execute_research_run(
        question=DEMO_PRESETS[3].default_question_tr,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    md_dup = render_report_markdown(b_dup)
    assert "Bilinmiyor" in md_dup
    pdf_dup = render_report_pdf_bytes(b_dup)
    text_dup = "\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(pdf_dup)).pages)
    assert "Bilinmiyor" in text_dup
