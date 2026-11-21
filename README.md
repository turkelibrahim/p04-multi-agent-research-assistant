# P04 · Çok Ajanlı Araştırma Asistanı

**İbrahim Türkel · Okul Projesi · Python 3.11 / Streamlit / LangGraph / SQLite / ReportLab**

Öğrenci Türkçe veya İngilizce bir araştırma sorusu sorduğunda; araştırmayı alt sorulara bölen, kaynak toplayan, iddiaları birebir alıntıyla çıkaran, kanıt denetimi yapıp gerekirse 2. tur takip aramasına dönen ve tüm kanıt zincirini SQLite üzerinde saklayarak Markdown / Türkçe PDF rapora dönüştüren yerel Windows uygulamasıdır.

## 1. Windows Kurulumu ve Başlatma

> **Not:** Bazı Windows ortamlarında uygulama denetimi (`pip.exe`, `streamlit.exe` vb.) doğrudan `.exe` çalıştırmayı engelleyebilir. Bu nedenle tüm komutlar `python -m ...` biçiminde çalıştırılır.

```powershell
cd "C:\Users\black man\.gemini\antigravity\scratch\p04-multi-agent-research-assistant"

# Sanal ortam zaten hazırsa aşağıdaki kurulum adımını atlayabilirsiniz:
# Windows uygulama denetimi (WDAC) indirilen yeni .pyd dosyalarını kısıtlayabildiği için
# --system-site-packages önerilir.
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install --upgrade-strategy only-if-needed -r requirements.txt
```

### Uygulamayı Başlatma

```powershell
.\run_app.ps1
```

Alternatif olarak çift tıklamak için `run_app.bat` veya doğrudan komut satırından:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Uygulama açıldıktan sonra varsayılan **Çevrimdışı Demo** modu internet bağlantısı ve API anahtarı olmadan çalışır.

## 2. Canlı Araştırma Ayarları (.env)

Canlı mod (`OpenAI Responses API` + `Tavily Search API`) kullanmak isterseniz `.env.example` dosyasını `.env` olarak kopyalayıp kendi anahtarlarınızı girin:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
OPENAI_MODEL=gpt-4.1-mini
```

- Canlı mod seçildiğinde ekranda dış servislere gönderilecek veriler (soru, arama sorgusu, kanıt parçaları) ve çalışma sınırları (10 model isteği, Standartta en çok 4 arama, 60 sn model / 20 sn arama, 300 sn toplam süre) açıkça gösterilir.
- API anahtarları hiçbir zaman rapora, veritabanına veya çalışma günlüğüne yazılmaz.

## 3. Otomatik Testler ve K11 Akademik Değerlendirme

```powershell
# Tüm testleri çalıştır (K01-K09 ve K11 çevrimdışı doğrulamaları)
.\.venv\Scripts\python.exe -m pytest -v

# Kod kalitesi denetimi
.\.venv\Scripts\python.exe -m ruff check src tests app.py

# K11 10 soruluk sabit kaynaklı karşılaştırma (çevrimdışı)
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m p04_research.eval.runner --mode offline --output-dir docs/eval_output
```

Canlı model karşılaştırması için (`OPENAI_API_KEY` gerekir):

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m p04_research.eval.runner --mode live --output-dir docs/eval_output_live
```

## 4. Proje Belgeleri

- [Kısa Uygulama Planı](docs/UYGULAMA_PLANI.md)
- [Ajan Akışı ve Durum Makinesi](docs/AJAN_AKISI.md)
- [3 Dakikalık Sınıf Demosu Akışı](docs/SINIF_DEMOSU.md)
- [K01–K11 Kabul Kanıtları Tablosu](docs/KABUL_TABLOSU.md)
- [Örnek Demo Raporu (Markdown)](examples/ornek_rapor_demo.md) · [Örnek Demo Raporu (PDF)](examples/ornek_rapor_demo.pdf)
