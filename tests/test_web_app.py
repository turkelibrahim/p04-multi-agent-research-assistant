"""FastAPI web uygulaması, SSE akışı, doğrulama, gizli kaynak linkleri ve geçmiş yönetimi testleri."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import create_app  # noqa: E402

from p04_research.db.repository import ResearchRepository  # noqa: E402
from p04_research.providers.mock_demo import DEMO_PRESETS  # noqa: E402


def test_web_app_full_lifecycle_validation_stream_export_and_delete(tmp_path, monkeypatch):
    db_path = tmp_path / "web_test.db"
    monkeypatch.setenv("P04_DB_PATH", str(db_path))
    monkeypatch.setenv("P04_FORCE_DEMO_DEFAULT", "1")

    client = TestClient(create_app(base_dir=ROOT_DIR))

    # 1) Ana sayfa HTML ve statik dosyalar sorunsuz sunulur
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "Araştırma Asistanı" in r_index.text
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200

    # 2) Yapılandırma ve hazır örnekler döner
    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg_data = r_cfg.json()
    assert cfg_data["default_mode"] == "demo"
    assert len(cfg_data["presets"]) == len(DEMO_PRESETS)

    # 3) Kısa soru reddedilir ve DB'ye kayıt oluşmaz
    r_bad = client.post("/api/research", json={"question": "kısa", "mode": "demo"})
    assert r_bad.status_code == 400
    repo = ResearchRepository(db_path)
    assert repo.list_runs() == []

    # 4) SSE akışı demo araştırmasını canlı olaylarla yürütür ve kaydeder
    r_stream = client.post(
        "/api/research/stream",
        json={
            "question": DEMO_PRESETS[0].default_question_tr,
            "mode": "demo",
            "depth": "standard",
            "language": "tr",
        },
    )
    assert r_stream.status_code == 200
    events = [
        json.loads(line[5:].strip())
        for line in r_stream.text.splitlines()
        if line.startswith("data:")
    ]
    assert any(ev["type"] == "event" and ev["node_name"] == "planner" for ev in events)
    complete_ev = next(ev for ev in events if ev["type"] == "complete")
    bundle_data = complete_ev["data"]
    run_id = bundle_data["metadata"]["run_id"]
    assert len(bundle_data["report"]["findings"]) == 2
    assert len(bundle_data["sources"]) == 2
    # Kurmaca demo kaynakları harici URL sızdırmamalı
    assert all(s["url"] is None for s in bundle_data["sources"])

    # 5) Geçmişten yeniden yükleme yeni koşum oluşturmaz
    r_list = client.get("/api/runs")
    assert r_list.status_code == 200
    assert len(r_list.json()["runs"]) == 1

    r_get = client.get(f"/api/runs/{run_id}")
    assert r_get.status_code == 200
    assert r_get.json()["metadata"]["run_id"] == run_id
    assert len(repo.list_runs()) == 1

    # 6) Markdown ve PDF dışa aktarımları açık URL sızdırmadan çalışır
    r_md = client.get(f"/api/runs/{run_id}/export/md")
    assert r_md.status_code == 200
    md_text = r_md.text
    assert "## Kısa Yanıt" in md_text
    md_without_targets = re.sub(r"\[[^\]]+\]\(https?://[^)]+\)", "", md_text)
    assert "http://" not in md_without_targets and "https://" not in md_without_targets

    r_pdf = client.get(f"/api/runs/{run_id}/export/pdf")
    assert r_pdf.status_code == 200
    assert r_pdf.content.startswith(b"%PDF-")

    # 7) Kayıt silme işlemi veritabanından temizler
    r_del = client.delete(f"/api/runs/{run_id}")
    assert r_del.status_code == 200
    assert repo.list_runs() == []


def test_app_cold_import_has_no_circular_import():
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    res = subprocess.run(
        [sys.executable, "-c", "import app; assert app.app is not None"],
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
