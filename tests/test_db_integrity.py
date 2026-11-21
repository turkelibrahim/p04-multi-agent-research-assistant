"""SQLite bütünlüğü, yabancı anahtarlar, idempotency, atomik geri alma ve yeniden yükleme testleri (K08)."""

from __future__ import annotations

from contextlib import closing

from p04_research.db.repository import ResearchRepository
from p04_research.db.schema import CURRENT_SCHEMA_VERSION, get_connection, verify_migration_on_copy
from p04_research.models import VerdictStatus
from p04_research.providers.mock_demo import DEMO_PRESETS
from p04_research.workflow.graph import execute_research_run


def test_foreign_keys_enabled_and_schema_migration_version(repo, tmp_db_path):
    with closing(get_connection(tmp_db_path)) as conn:
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk == 1
        v = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        assert v == CURRENT_SCHEMA_VERSION

    assert verify_migration_on_copy(tmp_db_path) is True


def test_cross_run_s1_isolation_and_provenance_reload(app_config, tmp_db_path):
    repo1 = ResearchRepository(tmp_db_path)
    b1, ok1, _ = execute_research_run(
        question=DEMO_PRESETS[0].default_question_tr,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo1,
    )
    b2, ok2, _ = execute_research_run(
        question=DEMO_PRESETS[1].default_question_tr,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo1,
    )
    assert ok1 and ok2

    # Yeni bir repository örneği (uygulama yeniden başlatma simülasyonu)
    repo_restarted = ResearchRepository(tmp_db_path)
    loaded1 = repo_restarted.load_run_bundle(b1.metadata.run_id)
    loaded2 = repo_restarted.load_run_bundle(b2.metadata.run_id)
    assert loaded1 is not None and loaded2 is not None

    s1_run1 = next(s for s in loaded1.sources if s.source_id == "S1")
    s1_run2 = next(s for s in loaded2.sources if s.source_id == "S1")
    assert "Kütüphane" in s1_run1.title
    assert "Yemekhane" in s1_run2.title
    assert loaded1.report.report_markdown == b1.report.report_markdown

    # Bölüm 7 veri kabul örneği: C2 iddiasının 1. tur ilk dayanağı + 2. tur sorgusu + S2 son kararı
    prov = repo_restarted.get_claim_provenance(b1.metadata.run_id, "C2")
    assert prov is not None
    assert prov.claim.initial_source_id == "S1"
    assert "iki katına çıkmış olabilir" in prov.claim.initial_quote
    assert prov.round1_verification is not None
    assert prov.round1_verification.verdict == VerdictStatus.INSUFFICIENT
    assert prov.follow_up_search_round is not None
    assert prov.follow_up_search_round.round_no == 2
    assert prov.final_verification is not None
    assert prov.final_verification.verdict == VerdictStatus.CONTRADICTED
    assert prov.final_verification.evidence_items[0].source_id == "S2"
    assert "100" in prov.final_verification.evidence_items[0].quote


def test_double_submit_idempotency_creates_single_report(app_config, repo):
    q = DEMO_PRESETS[0].default_question_tr
    b1, ok1, _ = execute_research_run(
        question=q,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
        idempotency_key="same_submit_key_123",
    )
    b2, ok2, _ = execute_research_run(
        question=q,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=repo,
        idempotency_key="same_submit_key_123",
    )
    assert ok1 and ok2
    assert b1.metadata.run_id == b2.metadata.run_id
    runs = repo.list_runs()
    assert len(runs) == 1


def test_atomic_rollback_prevents_partial_writes(app_config, repo, tmp_db_path):
    bundle, _, _ = execute_research_run(
        question=DEMO_PRESETS[0].default_question_tr,
        language="tr",
        depth="standard",
        mode="demo",
        config=app_config,
        repository=None,
    )
    # Bilerek geçersiz FK (olmayan bir source_id'ye bağlanan iddia) yerleştir
    bundle.claims[0].initial_source_id = "S999_INVALID"
    ok, err = repo.save_run_bundle(bundle)
    assert ok is False
    assert err is not None
    # Hiçbir yarım kayıt kalmamalı
    assert repo.list_runs() == []
    with closing(get_connection(tmp_db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0


def test_simulated_db_failure_keeps_in_session_report(app_config, tmp_path):
    # Klasör yerine dosya yolu vererek DB bağlantı/yazma hatasını simüle et
    bad_path = tmp_path / "not_a_dir"
    bad_path.write_text("blocker", encoding="utf-8")
    failing_repo = ResearchRepository(bad_path / "db.sqlite")

    bundle, persisted, persist_err = execute_research_run(
        question=DEMO_PRESETS[0].default_question_tr,
        language="tr",
        depth="fast",
        mode="demo",
        config=app_config,
        repository=failing_repo,
    )
    assert persisted is False
    assert persist_err is not None
    assert bundle.report is not None
    assert "8 dakikadan 6 dakikaya" in bundle.report.short_answer
