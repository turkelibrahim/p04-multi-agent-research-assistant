# Kampüs Kütüphanesi Pilotu: Bekleme Süresi ve Hafta Sonu Ziyaretleri Raporu

> **Kurmaca eğitim verisi** (Çevrimdışı Demo)

- **Soru:** Kampüs kütüphanesi pilotunda bekleme süresi ve hafta sonu ziyaretleri nasıl değişti?
- **Çalışma Kimliği:** `run_20261120_090508_c0c29b`
- **Mod / Derinlik / Teknik Durum:** Çevrimdışı Demo / Standart / **Başarılı**
- **Araştırma Kalitesi (Quality Gate):** **Yüksek (Kalite Geçidinden Geçti)**
- **Tarih:** 2026-11-20T09:05:08+00:00
- **Kullanım:** Model çağrısı: 6 · Arama çağrısı: 2 · Süre: 15 ms · Token: Ölçülmedi

## Kısa Yanıt — Yönetici Özeti

Kampüs kütüphanesi pilotunda hafta içi ortalama bekleme süresi 8 dakikadan 6 dakikaya düşmüştür [S1]. Hafta sonu ziyaretlerinin iki katına çıktığı iddiası ise doğrudan turnike sayımıyla çürütülmüştür: hafta sonu ziyaretleri pilot öncesi ortalama 100 iken pilot döneminde 120 (+%20) olarak ölçülmüştür [S2].

## Araştırma Amacı

Kampüs kütüphanesi pilot uygulamasının bekleme süresi ve hafta sonu ziyaret hacmi üzerindeki doğrulanmış etkisini sınamak.

## Yöntem ve Kapsam

Derinlik: standard · Taranan kaynak: 2 · Arama turu: 2. Dönem: Ekim 2026 pilot uygulaması · Ölçü birimleri: dakika (bekleme) ve ziyaret sayısı (turnike).

## Alt Araştırma Soruları

Hafta içi bekleme süresi ölçümünü, ölçüm kapsamını, hafta sonu ziyaret sayımını ve sınırlılıkları ayrı alt sorularda inceler.

- **SQ1** [Yanıtlandı] — Bekleme süresi ve ziyaret sayısı hangi yöntem ve zaman aralığında ölçüldü?
  - _Ölçüm kapsamı / Measurement scope_
  - Arama amacı: Ölçüm yöntemi ve kapsanan saat aralığını doğrula
- **SQ2** [Yanıtlandı] — Pilot döneminde hafta içi danışma bekleme süresi nasıl değişti?
  - _Dakika cinsinden ortalama / Mean in minutes_
  - Arama amacı: Pilot öncesi ve sırasındaki bekleme süresi dakikalarını karşılaştır
- **SQ3** [Yanıtlandı] — Hafta sonu ziyaret sayısı pilot döneminde gerçekten iki katına çıktı mı?
  - _Turnike sayımı / Turnstile count_
  - Arama amacı: Ön tahmin ile doğrudan hafta sonu turnike sayımını karşılaştır
- **SQ4** [Yanıtsız] — Danışma kaydı ve turnike ölçümlerinin kapsam sınırlılıkları nelerdir?
  - _Kapsam ve genellenebilirlik_
  - Arama amacı: Ölçüm zaman aralığı ve ön tahmin eksiklerini tespit et

## Temel Bulgular

### C1 (SQ2) — Pilot döneminde hafta içi danışma masasında ölçülen ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü (n=420 işlem). [S1]
- **Gerekçe:** S1 danışma kaydı aynı dönem (Ekim 2026) ve aynı ölçüm için 8 dakikadan 6 dakikaya düşüşü doğrudan doğruluyor (n=420).
- **Sayısal Bağlam:** 8 dk -> 6 dk (n=420 işlem, hafta içi 09.00-17.00, Ekim 2026)
  - [S1] (tr): "Pilot uygulama boyunca hafta içi danışma masasında ölçülen ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü (n=420 işlem, Ekim 2026)."
- *Not:* Tek bağımsız kaynağa dayanmaktadır; ikinci bağımsız doğrulama henüz eklenmemiştir.

### C3 (SQ3) — Turnike sayımına göre hafta sonu ziyaret sayısı pilot öncesi ortalama 100 iken pilot döneminde 120 oldu (+%20 artış). [S2]
- **Gerekçe:** S2 turnike sayım raporu hafta sonu ziyaretlerinin 100'den 120'ye çıktığını (+%20) kesin sayımla doğruluyor.
- **Sayısal Bağlam:** 100 -> 120 ziyaret (+%20 artış, 4 hafta sonu)
  - [S2] (tr): "Hafta sonu turnike sayımına göre pilot öncesi ortalama 100 olan hafta sonu ziyaret sayısı, pilot döneminde 120 olarak ölçüldü (+%20 artış, 4 hafta sonu ortalaması)."
- *Not:* Tek bağımsız kaynağa dayanmaktadır; ikinci bağımsız doğrulama henüz eklenmemiştir.

## Sayısal Veriler ve Karşılaştırmalar

- **C1** [S1]: Pilot döneminde hafta içi danışma masasında ölçülen ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü (n=420 işlem). — Bağlam: 8 dk -> 6 dk (n=420 işlem, hafta içi 09.00-17.00, Ekim 2026)
- **C3** [S2]: Turnike sayımına göre hafta sonu ziyaret sayısı pilot öncesi ortalama 100 iken pilot döneminde 120 oldu (+%20 artış). — Bağlam: 100 -> 120 ziyaret (+%20 artış, 4 hafta sonu)

## Karşıt Kanıtlar

- C2 ([S2]): S2'deki doğrudan turnike sayımı hafta sonu ziyaretlerinin 100'den 120'ye çıktığını (+%20) gösteriyor; iki katına (+%100) çıkış iddiasıyla çelişiyor.

## Sınırlılıklar ve Elenen İddialar

### C2 (SQ3) · **Kaynak karşı çıkıyor** — Hafta sonu ziyaretleri pilot döneminde iki katına çıktı.
- **1. Tur İlk Karar:** Kanıt yetersiz — S1'deki ifade yalnızca kesin sayım eklenmemiş bir ön tahmindir ('çıkmış olabilir'). Ayrıca bu kaynak yalnızca hafta içini ölçtüğü için hafta sonu hakkında sonuç çıkaramadık.
- **Son Gerekçe:** S2'deki doğrudan turnike sayımı hafta sonu ziyaretlerinin 100'den 120'ye çıktığını (+%20) gösteriyor; iki katına (+%100) çıkış iddiasıyla çelişiyor.
  - [S2]: "Hafta sonu turnike sayımına göre pilot öncesi ortalama 100 olan hafta sonu ziyaret sayısı, pilot döneminde 120 olarak ölçüldü (+%20 artış, 4 hafta sonu ortalaması)."

## Genel Değerlendirme

Pilot verileri hafta içi bekleme süresinde iyileşme ve hafta sonu ziyaretlerinde ılımlı artış göstermektedir; ilk ön tahminin abartılı olduğu ikinci tur turnike kaydıyla düzeltilmiştir.

## Açık Sorular

- Danışma kaydı ve turnike ölçümlerinin kapsam sınırlılıkları nelerdir?

## Kaynakça

- **[S1] Kampüs Kütüphanesi Pilot Danışma Kaydı** · [Kurmaca] — Üniversite Kütüphane Müdürlüğü (Kurmaca) | Resmî kurum yayını (Doğrudan ölçüm / birincil kayıt, primary_record) | Yayın Tarihi: 2026-10-15 | Erişim Tarihi: 2026-11-20T09:05:08+00:00 | https://kurmaca.p04.local/kutuphane-danisma-kaydi?utm_source=demo
- **[S2] Kütüphane Turnike Sayım Raporu — Ekim 2026** · [Kurmaca] — Kampüs Bilgi İşlem ve Kütüphane Birimi (Kurmaca) | Kurumsal birincil veri (Doğrudan ölçüm / birincil kayıt, primary_record) | Yayın Tarihi: 2026-10-28 | Erişim Tarihi: 2026-11-20T09:05:08+00:00 | https://kurmaca.p04.local/kutuphane-turnike-sayimi-2026
