# P04 · Kısa Uygulama Planı

## 1. Ürün Hedefi
Öğrencinin normal bir Türkçe (veya İngilizce) soruyla başlattığı araştırmada; ilk ekranda kısa cevabı okuyabilmesi, her bulgudan `[S#]` kaynak işaretine veya iddia kimliğine (`C2`) tıklayarak özgün alıntıyı, yayıncıyı, yayın tarihini ve sistem yorumunu denetleyebilmesi, iki kaynak çeliştiğinde her ikisini yan yana görebilmesi ve aynı raporu bir hafta sonra SQLite geçmişinden açıp Markdown/PDF olarak indirebilmesidir.

## 2. Beş İş Akışı Düğümü ve Kurallı Yönlendirici

| Düğüm | Rol | Girdi | Çıktı |
|---|---|---|---|
| **1 · Planlayıcı** | Dil modeli | Soru, rapor dili, derinlik | Kısa plan, 2–5 alt soru (`SQ1..`), ilk sorgular |
| **2 · Web Araştırmacısı** | Araç düğümü | İlk sorgular veya 2. tur takip sorgusu | Normalize URL'li `S1, S2..` kaynaklar, 3.500 karaktere kesilmiş `EvidenceChunk` parçaları, içerik kopya tespiti |
| **3 · Analist** | Dil modeli | Alt sorular ve kayıtlı parçalar | Tekil iddialar (`C1, C2..`), alt soru ve birebir ilk alıntı eşlemesi (2. turda önceki kimlikler korunur) |
| **4 · Kanıt Denetçisi** | Dil modeli + programatik denetim | İddialar, ilk alıntı ve tüm parçalar | 4 karardan biri (`Destekleniyor`, `Kaynak karşı çıkıyor`, `Kaynaklar çelişiyor`, `Kanıt yetersiz`), gerekçe, eksik bilgi ve varsa 1 takip sorgusu |
| **5 · Raportör** | Dil modeli + kanonik derleyici | Son kararlar, iddialar, plan ve kaynakça | Kısa yanıt, alt soru durumları, kanıtlı bulgular, çelişkiler, sınırlılıklar, kaynakça ve UTF-8 Markdown / PDF |

**Kurallı Yönlendirici (LangGraph):** Hızlı modda 1. tur denetiminden sonra doğrudan Raportöre gider. Standart modda yalnızca önemli bir iddiada `Kanıt yetersiz` kararı, yeni bir takip sorgusu ve kalan arama/model/süre bütçesi birlikte varsa 1 kez Web Araştırmacısına döner (`plan_followup` -> `researcher`).

## 3. Seçilen Teknolojiler

- **Dil ve Ortam:** Python 3.11 (`.venv`), Windows 10/11
- **Arayüz:** Streamlit (`app.py`, 5 ekran + `st.session_state` güvenli yönlendirme)
- **Ajan Akışı:** LangGraph `StateGraph` (`src/p04_research/workflow/graph.py`)
- **Canlı Sağlayıcılar:** OpenAI Responses API yapılandırılmış çıktı (`gpt-4.1-mini` başlangıç adayı) + Tavily Search API
- **Çevrimdışı Demo:** `MockDemoProvider` (5 deterministik senaryo, ağsız ve anahtarsız)
- **Veri Denetimi & Kalıcılık:** Pydantic v2 + SQLite (`PRAGMA foreign_keys = ON`, 11 tablo, atomik işlem)
- **Dışa Aktarım:** UTF-8 Markdown + ReportLab A4 PDF (gömülü Türkçe TTF yazı tipi)
- **Doğrulama:** `pytest`, Streamlit `AppTest`, `ruff`, `pypdf` ve Windows Chrome denetimi

## 4. Veri İlişkileri (SQLite)
`research_runs` tüm alt tabloların sahibidir (`run_id` ve `idempotency_key`). `sub_questions`, `sources`, `evidence_chunks` ve `claims` bileşik birincil anahtar `(run_id, <kimlik>)` kullanır; böylece başka bir çalışmanın `S1` kaydı yanlışlıkla eşleşemez. `verifications` her tur kararını (`round_no`, `is_final`) ayrı tutar; böylece `C2` gibi bir iddianın 1. tur ilk dayanağı ile 2. tur karşı kanıtı uygulama kapatılıp açıldıktan sonra da birlikte görüntülenir.
