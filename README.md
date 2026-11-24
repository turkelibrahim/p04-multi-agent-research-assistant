# Çok Ajanlı Araştırma Asistanı

Türkçe veya İngilizce bir araştırma sorusunu alt sorulara bölen, web kaynaklarından kanıt toplayan, iddiaları birebir alıntılarla doğrulayan ve tüm geçmişi SQLite üzerinde saklayarak Markdown / PDF rapor üreten yerel araştırma uygulamasıdır.

## Kurulum ve Başlatma (Windows)

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install --upgrade-strategy only-if-needed -r requirements.txt
```

Uygulamayı başlatmak için `run_app.bat` dosyasına çift tıklayın veya terminalden çalıştırın:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Çevrimdışı demo modu internet ve API anahtarı olmadan çalışır.

## Canlı Web Araştırması (.env)

Gerçek web araması ve model sentezi için `.env.example` dosyasını `.env` olarak kopyalayıp anahtarlarınızı ekleyin:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
OPENAI_MODEL=gpt-4.1-mini
```

## Doğrulama ve Testler

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m ruff check src tests app.py
.\.venv\Scripts\python.exe -m pytest -v
```
