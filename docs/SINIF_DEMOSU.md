# P04 · Sınıfta Gösterilecek 3 Dakikalık Demo Senaryosu

Bu akış, ürün tanımındaki **10. bölüm** adımlarının birebir uygulama karşılığıdır ve tamamen çevrimdışı demo modunda çalışır.

## Adım Adım Gösterim

1. **İlk 30 saniye (Başlangıç ve Mod Görünürlüğü)**
   - Uygulamayı `.\run_app.ps1` ile açın.
   - `Yeni Araştırma` ekranında hazır senaryolardan **1 · Sınıf Demosu: Kampüs Kütüphanesi Pilotu** seçilidir; soru kutusunda *"Kampüs kütüphanesi pilotunda bekleme süresi ve hafta sonu ziyaretleri nasıl değişti?"* yazar.
   - Üstte **"Kurmaca eğitim verisi"** rozetini, **Standart** derinlik seçiminin seçili olduğunu gösterip **Araştırmayı Başlat** düğmesine basın.

2. **30–70 saniye (1. Tur: Desteklenen İddia ve Ön Tahmin Ayrımı)**
   - Rapor açıldığında üstte **Kısa Yanıt** özetini gösterin.
   - `Kanıt ve Kaynak Ayrıntısı` ekranına geçip `C1` ve `C2` iddialarını gösterin:
     - `C1`: `S1` danışma kaydında hafta içi beklemenin 8 dakikadan 6 dakikaya düştüğü `Destekleniyor`.
     - `C2` (1. Tur): `S1` ön notunda *"hafta sonu ziyaretleri iki katına çıkmış olabilir"* yazdığı ve kaynak yalnızca hafta içini ölçtüğü için Denetçi `Kanıt yetersiz` kararı vermiştir.

3. **70–115 saniye (2. Tur Takip Araması ve Karar Değişimi)**
   - `İlerleme ve Ajan Akışı` ekranında **2. Tur Takip Araması Tetiklendi** kutusunu açın; takip sorgusunun nedeninin `C2` için eksik turnike sayımı olduğunu gösterin.
   - `Kanıt ve Kaynak Ayrıntısı` ekranında `C2` için sol sütunda **1. Tur (`Kanıt yetersiz`)**, sağ sütunda **2. Tur (`Kaynak karşı çıkıyor`)** ve `S2` turnike sayımı (`100` -> `120`, +%20 artış) karşı alıntısını gösterin.
   - Raporda hem 100 ve 120 değerlerinin yer aldığını hem de iki kat artış iddiasının neden kesin sonuçlara girmediğini vurgulayın.

4. **115–150 saniye (Kaynak Atlama, Türkçe PDF ve Geçmiş)**
   - Rapordaki `[S2] Kaynağını Aç` düğmesine tıklayarak tek etkileşimle `S2` kartındaki özgün alıntıyı açın.
   - `PDF (.pdf) İndir` ile Türkçe karakterli PDF çıktısını indirin.
   - `Geçmiş ve Dışa Aktarım` ekranına geçip kaydı **Raporu Aç** ile yeniden yükleyin (sıfır yeni model çağrısı ile aynı `S1`/`S2` kimlikleri ve alıntılar gelir).

5. **Son 30 saniye (Hızlı vs Standart ve K11 Değerlendirmesi)**
   - Aynı soruyu **Hızlı (1 arama, tek tur)** modda çalıştırıp 2. turun tetiklenmediğini (`C2` açık kalır, `S2` hiç çağrılmaz) fark olarak gösterin.
   - `docs/eval_output/eval_summary.md` dosyasından K11 10 soruluk sabit kaynak değerlendirme tablosunu paylaşın.
