"""Streamlit AppTest ile uçtan uca arayüz, doğrulama, [S2] atlama ve geçmiş testleri."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from p04_research.db.repository import ResearchRepository

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def test_ui_validation_offline_demo_jump_and_history(tmp_path, monkeypatch):
    db_path = tmp_path / "ui_test.db"
    monkeypatch.setenv("P04_DB_PATH", str(db_path))
    monkeypatch.setenv("P04_FORCE_DEMO_DEFAULT", "1")

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # 1) Kısa giriş reddedilir ve veritabanına kayıt oluşmaz (K02)
    at.text_area(key="question_input").set_value("kısa").run()
    start_btn = next(b for b in at.button if b.label == "Araştırmayı Başlat")
    start_btn.click().run()
    assert not at.exception
    assert len(at.error) >= 1
    repo = ResearchRepository(db_path)
    assert repo.list_runs() == []

    # 2) Sınıf demosu standart modda çalıştırılır (K01 / K03 / K04 / K05)
    at.button[0].click().run()  # "Örneği Forma Doldur" -> varsayılan Kütüphane Pilotu
    start_btn = next(b for b in at.button if b.label == "Araştırmayı Başlat")
    start_btn.click().run()
    assert not at.exception
    assert at.session_state["active_view"] == "Rapor"
    assert len(repo.list_runs()) == 1

    # Markdown ve PDF indirme düğmeleri mevcut olmalı
    dl_buttons = at.get("download_button")
    assert len(dl_buttons) == 2

    # 3) Rapordaki [S2] düğmesine basınca Kanıt ve Kaynak Ayrıntısı'na geçip S2 + C2 zincirini gösterir
    s2_btn = next(b for b in at.button if "[S2]" in b.label)
    s2_btn.click().run()
    assert not at.exception
    assert at.session_state["active_view"] == "Kanıt ve Kaynak Ayrıntısı"
    assert at.session_state["selected_source_id"] == "S2"

    # 4) Geçmiş ekranından aynı raporu yeniden yüklemek yeni koşum başlatmaz (K08)
    at.radio(key="active_view").set_value("Geçmiş ve Dışa Aktarım").run()
    assert not at.exception
    open_hist_btn = next(b for b in at.button if b.label == "Raporu Aç")
    open_hist_btn.click().run()
    assert not at.exception
    assert at.session_state["active_view"] == "Rapor"
    assert len(repo.list_runs()) == 1
