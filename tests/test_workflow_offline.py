"""LangGraph 5 düğümlü iş akışı ve K01-K07 çevrimdışı demo testleri."""

from __future__ import annotations

from p04_research.models import (
    RunStatus,
    VerdictStatus,
)
from p04_research.providers.base import ProviderError
from p04_research.providers.mock_demo import DEMO_PRESETS, MockDemoProvider
from p04_research.workflow.graph import execute_research_run


def test_k01_k03_k04_k05_classroom_standard_demo_two_rounds(app_config, repo):
    q = DEMO_PRESETS[0].default_question_tr
    bundle, persisted, err = execute_research_run(
        question=q,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    assert persisted is True and err is None
    assert bundle.metadata.status == RunStatus.COMPLETED
    assert bundle.metadata.search_calls == 2
    assert len(bundle.search_rounds) == 2
    assert bundle.search_rounds[1].round_no == 2
    assert bundle.search_rounds[1].triggered_by_claim_id == "C2"

    # Tüm 5 düğüm + yönlendirici olay kaydında görünür
    node_names = {e.node_name for e in bundle.events}
    assert {"planner", "researcher", "analyst", "verifier", "router", "reporter"}.issubset(node_names)

    # S1 ve S2 kurmaca etiketli ve doğru verili
    sids = {s.source_id: s for s in bundle.sources}
    assert "S1" in sids and "S2" in sids
    assert sids["S1"].is_mock is True and sids["S2"].is_mock is True

    # C2 1. turda insufficient, 2. turda contradicted olmalıdır
    c2_vers = sorted([v for v in bundle.verifications if v.claim_id == "C2"], key=lambda v: v.round_no)
    assert len(c2_vers) == 2
    assert c2_vers[0].verdict == VerdictStatus.INSUFFICIENT
    assert c2_vers[0].is_final is False
    assert c2_vers[1].verdict == VerdictStatus.CONTRADICTED
    assert c2_vers[1].is_final is True
    assert c2_vers[1].evidence_items[0].source_id == "S2"

    # Rapor kısa yanıtı ve bulguları doğru değerleri taşır
    assert bundle.report is not None
    assert "8 dakikadan 6 dakikaya" in bundle.report.short_answer
    assert "100" in bundle.report.short_answer and "120" in bundle.report.short_answer
    supported_ids = {f.claim_id for f in bundle.report.supported_findings}
    assert "C1" in supported_ids and "C3" in supported_ids
    assert "C2" not in supported_ids


def test_k03_fast_mode_stops_after_one_round(app_config, repo):
    q = DEMO_PRESETS[0].default_question_tr
    bundle, _, _ = execute_research_run(
        question=q,
        language="tr",
        depth="fast",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    assert bundle.metadata.status == RunStatus.COMPLETED
    assert bundle.metadata.search_calls == 1
    assert len(bundle.search_rounds) == 1
    assert [s.source_id for s in bundle.sources] == ["S1"]
    c2_final = next(v for v in bundle.verifications if v.claim_id == "C2" and v.is_final)
    assert c2_final.verdict == VerdictStatus.INSUFFICIENT


def test_k05_conflicting_dining_no_definitive_ruling(app_config, repo):
    q = DEMO_PRESETS[1].default_question_tr
    bundle, _, _ = execute_research_run(
        question=q,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    assert bundle.report is not None
    assert len(bundle.report.conflicts) == 1
    conf = bundle.report.conflicts[0]
    assert {conf.side_a.source_id, conf.side_b.source_id} == {"S1", "S2"}
    assert "%68" in bundle.report.short_answer and "%41" in bundle.report.short_answer
    assert "çelişkili" in bundle.report.short_answer.lower() or "kesin hüküm" in bundle.report.short_answer.lower()


def test_k02_bilingual_english_quote_preserved_in_turkish_report(app_config, repo):
    q = DEMO_PRESETS[2].default_question_tr
    bundle, _, _ = execute_research_run(
        question=q,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    assert bundle.report is not None
    assert "## Kısa Yanıt" in bundle.report.report_markdown
    f0 = bundle.report.supported_findings[0]
    assert f0.quotes[0].quote_language == "en"
    assert "In the fall 2025 randomized cohort (n=310)" in f0.quotes[0].quote
    assert f0.quotes[0].explanatory_note is not None


def test_k02_english_report_headings(app_config, repo):
    q = DEMO_PRESETS[0].default_question_en
    bundle, _, _ = execute_research_run(
        question=q,
        language="en",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    assert bundle.report is not None
    assert "## Short Answer" in bundle.report.report_markdown
    assert "## Bibliography" in bundle.report.report_markdown


def test_k06_unknown_date_duplicate_copy_and_causation_warning(app_config, repo):
    q = DEMO_PRESETS[3].default_question_tr
    bundle, _, _ = execute_research_run(
        question=q,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    sids = {s.source_id: s for s in bundle.sources}
    assert sids["S1"].published_date is None
    assert sids["S2"].duplicate_of_source_id == "S1"
    assert "Bilinmiyor" in bundle.report.report_markdown
    c2_ver = next(v for v in bundle.verifications if v.claim_id == "C2" and v.is_final)
    assert c2_ver.verdict == VerdictStatus.INSUFFICIENT
    assert c2_ver.causation_warning is not None


def test_k06_zero_results_distinct_from_technical_error(app_config, repo):
    q = DEMO_PRESETS[4].default_question_tr
    bundle, _, _ = execute_research_run(
        question=q,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    assert bundle.metadata.status == RunStatus.COMPLETED
    assert len(bundle.sources) == 0
    assert bundle.metadata.error_type is None
    assert len(bundle.report.supported_findings) == 0


def test_k07_technical_error_redacts_secret_and_marks_failed(app_config, repo):
    app_config.openai_api_key = "sk-secretkey1234567890"

    class FailingLLM(MockDemoProvider):
        def plan(self, question, language, depth, max_initial_queries):
            raise ProviderError(
                "auth_error",
                "API hatası: sk-secretkey1234567890 geçersiz.",
                retryable=False,
            )

    bundle, persisted, _ = execute_research_run(
        question="Kampüs kütüphanesi bekleme süresi nasıldı?",
        language="tr",
        depth="fast",
        mode="live",
        config=app_config,
        repository=repo,
        llm_provider=FailingLLM(),
        search_provider=MockDemoProvider(),
    )
    assert persisted is True
    assert bundle.metadata.status == RunStatus.FAILED
    assert bundle.metadata.error_type == "auth_error"
    assert "sk-secret" not in (bundle.metadata.error_message or "")
    assert "[GIZLI_ANAHTAR]" in (bundle.metadata.error_message or "")


def test_k07_budget_exhaustion_stops_safely(app_config, repo):
    app_config.max_model_calls = 2

    bundle, _, _ = execute_research_run(
        question=DEMO_PRESETS[0].default_question_tr,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
    )
    assert bundle.metadata.model_calls <= 2
    assert bundle.metadata.status in (RunStatus.PARTIAL, RunStatus.FAILED)
    assert bundle.metadata.error_type == "budget_exhausted"


def test_k07_technical_search_failure_distinct_from_empty_search(app_config, repo):
    class FailingSearch:
        def search(self, query, max_results, depth):
            raise ProviderError("timeout", "Tavily araması 20 sn içinde yanıt vermedi.", retryable=True)

    bundle, _, _ = execute_research_run(
        question="Kampüs kütüphanesi pilotunda bekleme süresi nasıl değişti?",
        language="tr",
        depth="fast",
        mode="live",
        config=app_config,
        repository=repo,
        llm_provider=MockDemoProvider(),
        search_provider=FailingSearch(),
    )
    assert bundle.metadata.status == RunStatus.FAILED
    assert bundle.metadata.error_type == "timeout"
    assert bundle.report is not None and bundle.report.error_notice is not None
