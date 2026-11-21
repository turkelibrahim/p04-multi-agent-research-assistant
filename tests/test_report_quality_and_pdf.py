"""Rapor sentezi, Quality Gate döngü sınırı, 6 senaryo ve gelişmiş PDF testleri."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from p04_research.export.markdown_export import render_report_markdown
from p04_research.export.pdf_export import render_report_pdf_bytes
from p04_research.models import (
    EvidenceLevel,
    FinalReport,
    ReporterSynthesisOutput,
    ResearchDepth,
    RunBundle,
    RunLanguage,
    RunMetadata,
    RunMode,
    RunStatus,
    SubQuestionStatus,
    SubQuestionUpdateLLM,
)
from p04_research.providers.base import RawSearchItem, SearchResponse, UsageDelta
from p04_research.providers.live_extractive import ExtractiveLiveLLMProvider
from p04_research.providers.mock_demo import DEMO_PRESETS, MockDemoProvider
from p04_research.workflow.graph import execute_research_run


def test_quality_gate_detects_and_fixes_defects_within_retry_limit(app_config):
    app_config.max_quality_retries = 1

    class FlawedReporterProvider(MockDemoProvider):
        def synthesize_report(self, question, language, sub_questions, claims, verifications, sources):
            # Bilerek geçersiz [S99] atıfı ve tekrarlanan cümle üretir
            return (
                ReporterSynthesisOutput(
                    title="Kötü Kesilmiş Başlık...",
                    short_answer="Bu cümle tekrar eder. Bu cümle tekrar eder. Geçersiz atıf [S99].",
                    sub_question_updates=[
                        SubQuestionUpdateLLM(sq_id=sq.sq_id, status=SubQuestionStatus.ANSWERED)
                        for sq in sub_questions
                    ],
                    open_questions=[],
                ),
                UsageDelta(model_calls=1),
            )

    prov = FlawedReporterProvider()
    bundle, _, _ = execute_research_run(
        question=DEMO_PRESETS[0].default_question_tr,
        language="tr",
        depth="fast",
        mode="demo",
        config=app_config,
        llm_provider=prov,
        search_provider=prov,
    )
    qg_events = [e for e in bundle.events if e.node_name == "quality_gate"]
    assert any(e.status == "decision" for e in qg_events)
    assert any(e.status == "finish" for e in qg_events)
    assert bundle.report is not None
    assert not bundle.report.title.endswith("...")
    assert "[S99]" not in bundle.report.short_answer
    assert bundle.report.research_quality in ("high", "needs_improvement", "insufficient_evidence")


def test_all_twelve_report_sections_and_coherent_synthesis(app_config):
    class RichLiveSearch:
        def search(self, query, max_results, depth):
            return SearchResponse(
                query=query,
                items=[
                    RawSearchItem(
                        url="https://research.org.tr/rapor-2026",
                        title="2026 Yapay Zekâ Ajanları Kurumsal Verimlilik Raporu",
                        publisher="Araştırma Enstitüsü",
                        author="Dr. Emre Yılmaz",
                        content=(
                            "2026 yılında yapılan saha araştırmasında (n=800 şirket) yapay zekâ ajanları kullanan finans ekiplerinde aylık raporlama süresi %42 azaldı. "
                            "Aynı raporda çalışanların %58'inin yeni ajan denetim rollerine geçtiği kaydedildi."
                        ),
                        published_date="2026-08-12",
                        provider="mock_live",
                        is_mock=True,
                        evidence_level=EvidenceLevel.OFFICIAL_DOC,
                    ),
                ],
                search_calls=1,
            )

    bundle, _, _ = execute_research_run(
        question="2026 yılında yapay zekâ ajanlarının iş dünyasına etkisi: İnsan çalışanların yerini alabilirler mi?",
        language="tr",
        depth="standard",
        mode="live",
        config=app_config,
        llm_provider=ExtractiveLiveLLMProvider(),
        search_provider=RichLiveSearch(),
    )
    assert bundle.metadata.token_measurement_type == "estimated"
    assert (bundle.metadata.total_tokens or 0) > 0
    assert bundle.metadata.estimated_cost_usd is not None
    md = bundle.report.report_markdown
    for heading in (
        "## Kısa Yanıt",
        "## Araştırma Amacı",
        "## Yöntem ve Kapsam",
        "## Alt Araştırma Soruları",
        "## Temel Bulgular",
        "## Sayısal Veriler ve Karşılaştırmalar",
        "## Karşıt Kanıtlar",
        "## Sınırlılıklar",
        "## Genel Değerlendirme",
        "## Açık Sorular",
        "## Kaynakça",
    ):
        assert heading in md


def test_pdf_long_title_long_url_empty_table_and_multipage(app_config):
    # 1) Boş tablo / sıfır kaynaklı uzun Türkçe başlıklı PDF
    empty_bundle = RunBundle(
        metadata=RunMetadata(
            run_id="run_empty_pdf",
            idempotency_key="idem_empty_pdf",
            question="Çok Uzun Türkçe Araştırma Sorusu: Çeşitli Sektörlerdeki İşgücü Dönüşümü ve Ölçüm Yöntemlerinin Karşılaştırmalı Analizi Üzerine Kapsamlı İnceleme",
            language=RunLanguage.TR,
            depth=ResearchDepth.FAST,
            mode=RunMode.DEMO,
            model_name="gpt-4.1-mini",
            started_at="2026-11-20T10:00:00Z",
            status=RunStatus.COMPLETED,
        ),
        report=FinalReport(
            title="Araştırma Raporu: Çok Uzun Türkçe Araştırma Sorusu: Çeşitli Sektörlerdeki İşgücü Dönüşümü ve Ölçüm Yöntemlerinin Karşılaştırmalı Analizi Üzerine Kapsamlı İnceleme",
            language=RunLanguage.TR,
            short_answer="Bu testte kaynak bulunmadığı için boş kaynakça tablosuyla güvenli PDF üretimi doğrulanmaktadır.",
            sub_questions=[],
            bibliography=[],
        ),
    )
    pdf_empty = render_report_pdf_bytes(empty_bundle)
    txt_empty = "\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(pdf_empty)).pages)
    assert "Kapsamlı İnceleme" in txt_empty
    assert "Sayfa 1" in txt_empty

    # 2) Çok sayfalı ve çok uzun URL'li rapor (14 kaynak)
    class ManySourcesSearch:
        def search(self, query, max_results, depth):
            items = []
            for i in range(1, 15):
                long_url = f"https://akademik.kurum{i}.edu.tr/yayinlar/2026/yapay-zeka-ajanlari-ve-kurumsal-verimlilik-arastirmasi-bolum-{i}-ek-dosyalar-cok-uzun-adres"
                items.append(
                    RawSearchItem(
                        url=long_url,
                        title=f"2026 Yapay Zekâ Ajanları Kurumsal İnceleme {i}",
                        publisher=f"Üniversite Enstitüsü {i}",
                        author=f"Dr. Araştırmacı {i}",
                        content=f"2026 yılı {i}. saha çalışmasında (n={100 + i * 10} şirket) yapay zekâ ajanları operasyonel süreç süresini %{20 + i} iyileştirdi.",
                        published_date=None if i % 2 == 0 else f"2026-0{(i % 9) + 1}-15",
                        provider="mock_many",
                        is_mock=False,
                        evidence_level=EvidenceLevel.OFFICIAL_DOC,
                    )
                )
            return SearchResponse(query=query, items=items[:max_results], search_calls=1)

    app_config.max_results_per_query = 14
    bundle_many, _, _ = execute_research_run(
        question="2026 yılında yapay zekâ ajanlarının kurumsal süreç süresine etkisi nedir?",
        language="tr",
        depth="fast",
        mode="live",
        config=app_config,
        llm_provider=ExtractiveLiveLLMProvider(),
        search_provider=ManySourcesSearch(),
    )
    pdf_many = render_report_pdf_bytes(bundle_many)
    reader = PdfReader(BytesIO(pdf_many))
    assert len(reader.pages) >= 2
    full_txt = "\n".join(p.extract_text() or "" for p in reader.pages)
    assert "Sayfa 1" in full_txt and "Sayfa 2" in full_txt
    assert "Bilinmiyor" in full_txt
    md_many = render_report_markdown(bundle_many)
    assert "Yayın Tarihi" in md_many and "Erişim Tarihi" in md_many


def test_six_diverse_research_scenarios(app_config):
    # 1. Güncel teknoloji araştırması
    b_tech, _, _ = execute_research_run(
        "2026 yılında yapay zekâ ajanlarının iş dünyasına etkisi: İnsan çalışanların yerini alabilirler mi?",
        "tr", "standard", "demo", app_config,
    )
    # 2. Akademik araştırma (iki dilli kohort)
    b_acad, _, _ = execute_research_run(DEMO_PRESETS[2].default_question_tr, "tr", "standard", "demo", app_config)
    # 3. Karşılaştırmalı araştırma (kütüphane pilotu önce vs sonra)
    b_comp, _, _ = execute_research_run(DEMO_PRESETS[0].default_question_tr, "tr", "standard", "demo", app_config)
    # 4. Kaynakları çelişen konu
    b_conf, _, _ = execute_research_run(DEMO_PRESETS[1].default_question_tr, "tr", "standard", "demo", app_config)
    # 5. Yetersiz kaynak bulunan konu
    b_zero, _, _ = execute_research_run(DEMO_PRESETS[4].default_question_tr, "tr", "standard", "demo", app_config)
    # 6. Uzun ve karmaşık kullanıcı sorusu
    long_complex = (
        "2026 yılında orta ve büyük ölçekli teknoloji şirketlerinde otonom yapay zekâ ajanlarının "
        "yazılım mühendisliği, müşteri operasyonları ve finansal raporlama süreçlerindeki verimlilik "
        "artışları ile insan çalışanların görev değişimi arasındaki ölçülebilir farklar nelerdir?"
    )
    b_long, _, _ = execute_research_run(long_complex, "tr", "standard", "demo", app_config)

    assert b_tech.metadata.status == RunStatus.COMPLETED
    assert len(b_acad.report.supported_findings) == 1
    assert len(b_comp.search_rounds) == 2
    assert len(b_conf.report.conflicts) == 1
    assert b_zero.report.research_quality == "insufficient_evidence"
    assert not b_long.report.title.endswith("...")
