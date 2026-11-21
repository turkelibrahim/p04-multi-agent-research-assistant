"""Planlayıcı (Planner) ve Kaynak Değerlendirme (Web Research & Source Evaluation) testleri."""

from __future__ import annotations

from p04_research.models import (
    EvidenceLevel,
    PlannedSubQuestionLLM,
    PlannerOutput,
    SourceType,
    SubQuestionStatus,
    VerdictStatus,
)
from p04_research.providers.base import RawSearchItem, SearchResponse, UsageDelta
from p04_research.providers.live_extractive import ExtractiveLiveLLMProvider
from p04_research.providers.mock_demo import MockDemoProvider
from p04_research.verification_rules import (
    classify_source_type,
    evaluate_source_quality_and_relevance,
)
from p04_research.workflow.graph import execute_research_run


def test_long_turkish_title_preserved_without_ellipsis(app_config, repo):
    long_q = (
        "2026 Yılında Kurumsal İşletmelerde Çok Ajanlı Yapay Zekâ Sistemlerinin Yazılım Geliştirme, "
        "Veri Analizi ve İnsan Çalışanların İstihdam Yapısı Üzerindeki Uzun Dönemli Etkileri Nelerdir?"
    )
    bundle, _, _ = execute_research_run(
        question=long_q,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    assert bundle.report is not None
    assert not bundle.report.title.endswith("...")
    assert "Uzun Dönemli Etkileri Nelerdir?" in bundle.report.title
    assert bundle.metadata.question == long_q


def test_planner_deduplicates_subquestions_and_preserves_bilingual_meaning(app_config):
    class DuplicateSQPlanner(MockDemoProvider):
        def plan(self, question, language, depth, max_initial_queries):
            return (
                PlannerOutput(
                    plan_summary="Tekrarlı alt soru testi",
                    research_purpose="Alt soru tekilleştirmeyi doğrulamak",
                    timeframe="2026",
                    key_concepts=["yapay zeka", "istihdam"],
                    sub_questions=[
                        PlannedSubQuestionLLM(
                            sq_id="SQ1",
                            text="2026 yılında yapay zekâ ajanları hangi iş görevlerini otomatikleştiriyor?",
                            search_intent="Görev otomasyonu",
                            required_evidence="Saha araştırması",
                        ),
                        PlannedSubQuestionLLM(
                            sq_id="SQ2",
                            text="2026 yılında yapay zekâ ajanları hangi iş görevlerini otomatikleştiriyor?",
                            search_intent="Tekrar",
                            required_evidence="Tekrar",
                        ),
                        PlannedSubQuestionLLM(
                            sq_id="SQ3",
                            text="Ölçülmüş istihdam ve kadro etkileri nelerdir?",
                            search_intent="İstihdam verisi",
                            required_evidence="Bordro kaydı",
                        ),
                        PlannedSubQuestionLLM(
                            sq_id="SQ4",
                            text="Mevcut kanıtların sınırlılıkları nelerdir?",
                            search_intent="Sınırlılıklar",
                            required_evidence="Kapsam notları",
                        ),
                    ],
                    initial_queries=["kuzey kampusu 2029 isinlanma istasyonu"],
                ),
                UsageDelta(model_calls=1),
            )

    prov = DuplicateSQPlanner()
    bundle, _, _ = execute_research_run(
        question="Kuzey Kampüsü 2029 ışınlanma istasyonu günlük yolcu sayısı nedir?",
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        llm_provider=prov,
        search_provider=prov,
    )
    # Tekrarlanan SQ2 elenmiş olmalı (3 benzersiz alt soru kalır)
    assert len(bundle.sub_questions) == 3
    assert all(sq.search_intent for sq in bundle.sub_questions)
    # Kanıt bulunamayan alt sorular ASLA ANSWERED olmamalı
    assert all(sq.status == SubQuestionStatus.UNANSWERED for sq in bundle.sub_questions)

    # Türkçe ve İngilizce planlamada aynı 5 boyut üretilmeli
    ext = ExtractiveLiveLLMProvider()
    tr_plan, _ = ext.plan("2026 yılında yapay zekâ ajanlarının iş dünyasına etkisi?", "tr", "standard", 2)
    en_plan, _ = ext.plan("Impact of AI agents on business and workforce in 2026?", "en", "standard", 2)
    assert len(tr_plan.sub_questions) == len(en_plan.sub_questions) == 5


def test_seven_source_types_and_multifactor_evaluation():
    q = "2026 yılında yapay zekâ ajanlarının istihdam etkisi"
    cases = [
        ("https://journals.edu.tr/ai-workforce-2026", "Journal of AI Studies", "Üniversite Dergisi", "Dr. Ayşe Kaya", "2026-05-01", "2026 randomized cohort study (n=400) AI agents workforce.", SourceType.ACADEMIC_PEER_REVIEWED),
        ("https://tuik.gov.tr/rapor/istihdam-2026", "Resmî İstihdam İstatistikleri 2026", "TÜİK", None, "2026-06-10", "Resmî istatistik kurumu 2026 yapay zeka istihdam anketi.", SourceType.OFFICIAL_INSTITUTION),
        ("https://pwc.com.tr/rapor/ai-2026", "PwC Yapay Zekâ Ajanları 2026 Raporu", "PwC Türkiye", None, "2026-03-15", "2026 saha araştırması yapay zeka ajanları n=1200 şirket.", SourceType.RESEARCH_ORG_REPORT),
        ("https://corp.example.com/bordro-2026", "Kurumsal Bordro Verisi 2026", "ABC A.Ş.", None, "2026-07-01", "Kurumsal bordro ve turnike ölçüm kaydı n=500 çalışan.", SourceType.CORPORATE_PRIMARY_DATA),
        ("https://analiz.example.com/ai-2026", "Teknoloji Sektör Analizi", "TechReview", "Mehmet Demir", "2026-04-10", "2026 yapay zeka ajanlarının şirketlere etkisi.", SourceType.TRUSTED_SECONDARY_ANALYSIS),
        ("https://medium.com/@yazar/ai-gorus", "Bence Yapay Zekâ Ajanları", "Medium Blog", "Ali Can", None, "Kişisel blog görüş yazısı yapay zeka ajanları.", SourceType.NEWS_BLOG_OPINION),
        ("https://www.youtube.com/watch?v=abc123", "YouTube Teknoloji Sohbeti", "Video Kanalı", None, "2026-02-01", "Video sohbetinde konuşmacı yorumları.", SourceType.VIDEO_SOCIAL_MEDIA),
    ]
    for url, title, pub, author, pdate, content, expected in cases:
        s_type = classify_source_type(url, title, pub, content, author=author)
        assert s_type == expected
        ev = evaluate_source_quality_and_relevance(
            url=url,
            title=title,
            publisher=pub,
            content=content,
            question=q,
            author=author,
            published_date=pdate,
        )
        assert ev["source_type"] == expected
        assert ev["is_accessible"] is True
        assert 0.0 <= ev["credibility_score"] <= 1.0

    # Yayın tarihi ve metodolojisi olan akademik kaynak, tarihsiz blogdan daha yüksek güvenilirlik almalı
    acad = evaluate_source_quality_and_relevance(
        url=cases[0][0], title=cases[0][1], publisher=cases[0][2], content=cases[0][5],
        question=q, author=cases[0][3], published_date=cases[0][4],
    )
    blog = evaluate_source_quality_and_relevance(
        url=cases[5][0], title=cases[5][1], publisher=cases[5][2], content=cases[5][5],
        question=q, author=cases[5][3], published_date=cases[5][4],
    )
    assert acad["credibility_score"] > blog["credibility_score"]


def test_inaccessible_url_and_snippet_vs_fulltext_and_missing_date(app_config):
    class MixedSourcesSearch:
        def search(self, query, max_results, depth):
            return SearchResponse(
                query=query,
                items=[
                    RawSearchItem(
                        url="https://locked.example.com/report-2026",
                        title="Kilitli Rapor",
                        publisher="locked.example.com",
                        content="403 Forbidden - Access Denied",
                        published_date=None,
                        provider="mock_test",
                        is_mock=True,
                    ),
                    RawSearchItem(
                        url="https://research.example.edu.tr/full-study-2026",
                        title="2026 Yapay Zekâ Ajanları Tam Metin Araştırması",
                        publisher="Üniversite Araştırma Enstitüsü",
                        author="Prof. Dr. Zeynep Ak",
                        content=(
                            "2026 yılında yapılan bu kapsamlı saha araştırmasında (n=640 şirket) yapay zekâ ajanlarını kullanan işletmelerde veri işleme süresi %31 kısaldı. "
                            + ("Metodoloji ve örneklem ayrıntıları tam metinde sunulmuştur. " * 20)
                        ),
                        published_date=None,  # Tarih bilinmiyor!
                        provider="mock_test",
                        is_mock=True,
                        evidence_level=EvidenceLevel.OFFICIAL_DOC,
                    ),
                ],
                search_calls=1,
            )

    bundle, _, _ = execute_research_run(
        question="2026 yılında yapay zekâ ajanları işletmelerde veri işleme süresini nasıl etkiledi?",
        language="tr",
        depth="fast",
        mode="live",
        config=app_config,
        llm_provider=ExtractiveLiveLLMProvider(),
        search_provider=MixedSourcesSearch(),
    )
    s1 = next(s for s in bundle.sources if s.source_id == "S1")
    s2 = next(s for s in bundle.sources if s.source_id == "S2")

    # S1 erişilemeyen sayfa olarak işaretlenmeli ve okunmuş parça üretmemeli
    assert s1.is_accessible is False
    assert s1.retrieval_method == "inaccessible"
    assert not any(ch.source_id == "S1" for ch in bundle.chunks)

    # S2 tam metin ayrımı yapmalı ve eksik yayın tarihini erişim tarihiyle doldurmamalı
    assert s2.is_accessible is True
    assert s2.retrieval_method == "full_text"
    assert s2.published_date is None
    assert s2.published_date != s2.retrieved_at
    assert "Bilinmiyor" in bundle.report.report_markdown

    # Tek kaynak desteği varsa bunu tek-kaynak uyarısıyla veya doğrulamada belirtmeli
    c1_ver = next((v for v in bundle.verifications if v.is_final and v.verdict == VerdictStatus.SUPPORTED), None)
    if c1_ver is not None:
        assert c1_ver.independent_support_count == 1
        assert c1_ver.single_source_warning is not None
