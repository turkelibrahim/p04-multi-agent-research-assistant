# P04 · Ajan Akışı ve LangGraph Mimarisi

```mermaid
flowchart TD
    START([Başlat]) --> PLAN["1 · Planlayıcı (Planner)\n2-5 alt soru + ilk sorgular"]
    PLAN -->|Hata / Bütçe| REP["5 · Raportör (Reporter)\nDurum + Rapor + Kaynakça"]
    PLAN -->|Geçerli sorgu| RES["2 · Web Araştırmacısı (Researcher)\nTavily / Demo + URL/Kopya tekilleştirme"]
    RES -->|Teknik hata ve 0 parça| REP
    RES -->|0 sonuç (boş arama)| REP
    RES -->|Parçalar mevcut| ANA["3 · Analist (Analyst)\nTekil iddialar + birebir alıntılar"]
    ANA -->|Hata / Bütçe| REP
    ANA --> VER["4 · Kanıt Denetçisi (Verifier)\n4 karar + programatik alıntı/kopya kilidi"]
    VER -->|Hızlı mod veya 2. tur bitti veya bütçe dolu| REP
    VER -->|Standart mod + Kanıt yetersiz + yeni sorgu + kalan bütçe| FU["Yönlendirici (plan_followup)\n2. tur nedenini günlüğe yazar"]
    FU --> RES
    REP --> ENDNODE([Bitiş + SQLite Atomik Kayıt])
```

## Programatik Güvenlik ve Bütünlük Kuralları

1. **Birebir Alıntı Kilidi (`enforce_verification_guardrails`):** Denetçinin `Destekleniyor` veya `Kaynak karşı çıkıyor` kararındaki her alıntı, `evidence_chunks.excerpt_text` içinde NFC/boşluk normalizasyonu ile birebir aranır. Bulunamazsa karar yazılımla `Kanıt yetersiz` düzeyine düşürülür.
2. **Kesme Sınırı Kilidi:** 3.500 karakter sınırında kesilmiş bir parçanın son 15 karakterine denk düşen yarım alıntı `Destekleniyor` olarak yayımlanamaz.
3. **Kopya Kaynak Kilidi:** Aynı içerik parmak izine (`SHA-256`) sahip iki farklı URL `duplicate_of_source_id` ile bağlanır ve `Kaynaklar çelişiyor` kararı için iki bağımsız kanıt sayılmaz.
4. **Hata ve Bütçe Sınırları:** Hızlı mod 1 arama; Standart mod ilk tur en çok 3 + koşullu 1 ek arama (toplam en çok 4); toplam en çok 10 model isteği; 60 sn model, 20 sn arama, 300 sn toplam süre sınırı. Boş HTTP 200 arama sonucu teknik arıza olarak etiketlenmez.
