# P04 Multi-Agent Research Assistant — Mimari İnceleme ve İyileştirme Planı

## 1. Başlangıç Doğrulama Sonuçları (Baseline)

Planlama öncesi mevcut proje durumu doğrudan çalıştırılarak ölçülmüştür:
- `ruff check src tests app.py`: **0 hata** (`All checks passed!`)
- `pytest -v`: **29 / 29 test geçti** (`4.05s`)
  - Kapsanan mevcut test dosyaları: `test_db_integrity.py` (5), `test_eval_k11.py` (2), `test_exports.py` (2), `test_live_adapters.py` (4), `test_ui_apptest.py` (1), `test_validation_and_rules.py` (5), `test_workflow_offline.py` (10).

---

## 2. Mevcut LangGraph Mimarisi ve Uçtan Uca Akış

Uygulama girişi `app.py` (Streamlit 5 ekran) üzerinden `execute_research_run` (`src/p04_research/workflow/graph.py`) fonksiyonunu çağırır:

1. **Planlayıcı (`planner_node`, `src/p04_research/workflow/nodes.py`)**: `LLMProvider.plan(...)` çağrısı ile `PlannerOutput` üretir, `SubQuestion` listesi ve `pending_queries` alanlarını `ResearchGraphState` (`src/p04_research/workflow/state.py`) içine yazar.
2. **Web Araştırmacısı (`researcher_node`, `nodes.py`)**: `pending_queries` içindeki sorguları `SearchProvider.search(...)` (canlıda `TavilySearchProvider`, demoda `MockDemoProvider`) ile yürütür. URL'leri `normalize_url` ile tekilleştirir (`S1, S2...`), metni `truncate_source_text` ile 3.500 karaktere kesip `EvidenceChunk` (`E1, E2...`) üretir ve `compute_content_fingerprint` ile kopya metinleri işaretler.
3. **Analist (`analyst_node`, `nodes.py`)**: `LLMProvider.extract_claims(...)` çağrısı ile alt sorulara bağlı `ClaimRecord` (`C1, C2...`) ve ilk alıntıları çıkarır.
4. **Kanıt Denetçisi (`verifier_node`, `nodes.py`)**: `LLMProvider.verify_claims(...)` çıktısını aldıktan sonra `enforce_verification_guardrails` (`src/p04_research/verification_rules.py`) ile kaynak kimliği, birebir alıntı, kesme sınırı ve kopya kaynak kurallarını uygular.
5. **Koşullu Yönlendirici (`route_after_verifier` -> `plan_followup_node`, `graph.py` / `nodes.py`)**: Standart modda, 1. tur sonunda `VerdictStatus.INSUFFICIENT` durumunda kalan ve yeni bir `follow_up_query` içeren iddia varsa (bütçe izin verdiği sürece) 2. tur aramasına döner.
6. **Raportör (`reporter_node`, `nodes.py`)**: Kararları `ReportFindingItem`, `ConflictComparisonItem`, `LimitationItem` ve `BibliographyItem` yapılarına ayırır; `LLMProvider.synthesize_report(...)` çağrısıyla `FinalReport` üretip `render_report_markdown` (`src/p04_research/export/markdown_export.py`) ile Markdown metnini derler.
7. **Kalıcılık ve Dışa Aktarım**: `ResearchRepository.save_run_bundle` (`src/p04_research/db/repository.py`) 11 SQLite tablosunu (`src/p04_research/db/schema.py`, `PRAGMA foreign_keys = ON`) atomik tek işlemde kaydeder; `render_report_pdf_bytes` (`src/p04_research/export/pdf_export.py`) aynı `FinalReport` kaydından gömülü Türkçe TTF fontlu PDF üretir.
8. **Değerlendirme (`src/p04_research/eval/`)**: `runner.py` 10 sabit kaynaklı soru (`benchmark_dataset.py`) üzerinde `single_agent.py` ve çok ajanlı akışı `metrics.py` kurallarıyla karşılaştırır.

---

## 3. Tespit Edilen Sorunlar ve Kök Neden Eşlemesi

| # | Sorun | Kök Neden | Gerçek Dosya ve Fonksiyon İsimleri |
|---|---|---|---|
| 1 | **Konuyla ilgisiz kanıtların kabul edilmesi** | Denetim katmanı yalnızca `quote_matches_source` alt-dize eşleşmesine bakmakta; soruyla anlamsal/zamansal ilgililik, soru cümlesi olma durumu, dipnot/zaman damgası rakamları, konuşmacı beyanı veya paydasız yüzde genellemesi denetlenmemekte, `IRRELEVANT` kararı bulunmamaktadır. | `src/p04_research/providers/live_extractive.py` (`_extract_verbatim_sentences`, `ExtractiveLiveLLMProvider.extract_claims`, `ExtractiveLiveLLMProvider.verify_claims`), `src/p04_research/verification_rules.py` (`enforce_verification_guardrails`), `src/p04_research/models.py` (`VerdictStatus`) |
| 2 | **Kopuk cümlelerden rapor üretilmesi** | `synthesize_report` desteklenen iddiaların ham `initial_quote` cümlelerini arka arkaya birleştirmekte (`" ".join(bullet_summaries)`), yapılandırılmış akademik sentez, tekrar engelleme ve kalite geçidi kontrolü yapmamaktadır. | `src/p04_research/providers/live_extractive.py` (`ExtractiveLiveLLMProvider.synthesize_report`, `_summarize_claim_statement`), `src/p04_research/workflow/nodes.py` (`reporter_node`) |
| 3 | **Başlıkların üç nokta ile kesilmesi** | `clean_question_title` fonksiyonu varsayılan `max_len=88` ve `reporter_node` / `plan` / `synthesize_report` çağrıları `75`–`85` karakter kesmesi uygulayarak sonuna `"..."` eklemektedir. | `src/p04_research/providers/mock_demo.py` (`clean_question_title`, `MockDemoProvider.plan`, `MockDemoProvider.synthesize_report`), `src/p04_research/providers/live_extractive.py`, `src/p04_research/workflow/nodes.py` (`reporter_node`) |
| 4 | **Kaynak kalitesinin yetersiz değerlendirilmesi** | `_infer_evidence_level` yalnızca `.gov`/`.edu` alan adı uzantısına ve 300 karakter uzunluğuna bakmaktadır. 7 kaynak türü, yazar/yöntem/erişilebilirlik/arama özeti vs tam metin ayrımı ve içerik uygunluk puanı yoktur. | `src/p04_research/providers/tavily_search.py` (`_infer_evidence_level`, `_infer_publisher`, `TavilySearchProvider.search`), `src/p04_research/models.py` (`SourceRecord`, `EvidenceLevel`) |
| 5 | **Kanıt yetersizken alt soruların tamamlanması** | `ExtractiveLiveLLMProvider.synthesize_report` ve `FixedBenchmarkProvider.synthesize_report` destekli bulgu olmayan alt soruları `PARTIALLY_ANSWERED` veya `ANSWERED` işaretleyebilmekte; `reporter_node` alt sorular için `findings` ve `remaining_gaps` takibi yapmamaktadır. | `src/p04_research/providers/live_extractive.py` (`synthesize_report`), `src/p04_research/eval/benchmark_dataset.py` (`FixedBenchmarkProvider.synthesize_report`), `src/p04_research/workflow/nodes.py` (`reporter_node`) |
| 6 | **Token kullanımının ölçülmemesi** | Sağlayıcı gerçek token verisi dönmediğinde (`MockDemoProvider`, `ExtractiveLiveLLMProvider`) kullanım ölçüm türü (`provider` vs `estimated` vs `unmeasured`) ve tahmini token/maliyet hesaplaması yapılmamakta, tek biçimde `None` (`Ölçülmedi`) kalmaktadır. | `src/p04_research/providers/mock_demo.py`, `src/p04_research/providers/live_extractive.py`, `src/p04_research/workflow/nodes.py` (`_add_usage`), `src/p04_research/models.py` (`RunMetadata`) |

---

## 4. Değişecek Modüller ve Korunacak İşlevler

### Değişecek Modüller
- `src/p04_research/config.py`: `max_quality_retries` ayarı eklenmesi.
- `src/p04_research/models.py`: `SourceType`, `VerdictStatus.IRRELEVANT`, `SubQuestion`, `SourceRecord`, `VerificationRecord`, `FinalReport`, `RunMetadata` genişletmeleri (geriye dönük uyumlu varsayılan değerlerle).
- `src/p04_research/verification_rules.py`: Çok faktörlü kaynak sınıflandırma, erişilemeyen URL eleme, konu ilgililik kontrolü, C7/C5/C4/C3/C2 kanıt reddi kuralları ve tek kaynak sınırlılık uyarısı.
- `src/p04_research/providers/base.py`, `mock_demo.py`, `tavily_search.py`, `openai_llm.py`, `live_extractive.py`: Başlık kesmeyi kaldırma, 3–6 zengin alt soru üretimi, kaynak meta verisi toplama, gerçekçi `tiktoken` tabanlı isteğe bağlı token tahmini (sağlayıcı tokenı yoksa etiketli).
- `src/p04_research/workflow/state.py`, `i18n.py`, `nodes.py`, `graph.py`: Sentezlenmiş akademik rapor üretimi, `quality_gate_node` ve sınırlı düzeltme döngüsü (`max_quality_retries`), teknik başarı vs araştırma kalitesi ayrımı.
- `src/p04_research/db/schema.py`, `repository.py`: SQLite v2 şeması (`irrelevant` kararı ve yeni meta veriler) + mevcut veritabanına güvenli göç (`ALTER TABLE`).
- `src/p04_research/export/markdown_export.py`, `pdf_export.py`: Tam başlık, 12 akademik bölüm, tıklanabilir URL ve iç `[S#]` bağlantıları, sayfa numarası/üstbilgi, taşmasız tablo ve URL kırılması.
- `src/p04_research/eval/metrics.py`, `runner.py`, `benchmark_dataset.py`: Genişletilmiş 12 ölçüt ve kanıt yetersizliğinde alt soru kuralının uyumu.
- `app.py`: Yeni alanların (kalite durumu, token ölçüm türü, 7 kaynak türü, `Konuyla ilgisiz` rozeti) temiz gösterimi.

### Korunacak İşlevler
- 5 ana LangGraph düğümü (`planner`, `researcher`, `analyst`, `verifier`, `reporter`) ve 2. tur takip araması yönlendiricisi (`route_after_verifier` -> `plan_followup`).
- Mevcut tüm `RunBundle` / `ClaimProvenanceView` sözleşmeleri, `PRAGMA foreign_keys = ON`, `idempotency_key` koruması ve atomik geri alma davranışı.
- 29 mevcut otomatik testin tamamı.

---

## 5. Test Planı ve Olası Regresyon Riskleri

- **Yeni Test Paketleri:**
  - `tests/test_planner_and_sources.py`: Tam uzun başlık, tekrarsız alt sorular, yanıtlanamayan alt sorunun açık kalması, 7 kaynak türü, erişilemeyen URL, snippet vs tam metin, eksik tarih.
  - `tests/test_verification_regressions.py`: C7, C5, C4, C3, C2 regresyon örneklerinin ayrı ayrı reddedildiğini/doğru sınıflandırıldığını doğrulayan testler.
  - `tests/test_report_quality_and_pdf.py`: Kopuk cümle/tekrar engelleme, 12 rapor bölümü, Quality Gate düzeltme döngüsü sınırı, 6 yeni senaryo, çok sayfalı/uzun URL'li/Türkçe karakterli PDF testleri.
- **Olası Regresyon Riskleri:**
  1. Mevcut `SQLite` veritabanında `verifications.verdict CHECK(...)` kısıtının `'irrelevant'` değerini kabul etmemesi (çözüm: v1 -> v2 güvenli tablo göçü).
  2. Sınıf demosu (`DEMO_PRESETS[0]`) ve `test_workflow_offline.py` testlerindeki `SQ1..SQ3` veya `C1..C3` beklentilerinin bozulması (çözüm: `library_pilot` preset kimliklerini koruyup yeni alanları eklemek).
  3. `MockDemoProvider` testinde `test_markdown_and_turkish_pdf_roundtrip` içinde `"Ölçülmedi"` metni beklentisinin bozulması (çözüm: gerçek sağlayıcı tokenı yokken tahmini maliyet/token alanını ayrı tutup `total_tokens is None` olduğunda `"Ölçülmedi"` etiketini korumak veya tahmini etiketi ayrıca göstermek).
