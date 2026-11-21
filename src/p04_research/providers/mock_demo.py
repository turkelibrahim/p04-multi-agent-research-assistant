"""Çevrimdışı deterministik kurmaca demo sağlayıcısı (ağ ve API anahtarı gerektirmez)."""

from __future__ import annotations

from dataclasses import dataclass

from p04_research.models import (
    AnalystOutput,
    ClaimRecord,
    EvidenceChunk,
    EvidenceLevel,
    EvidenceRole,
    ExtractedClaimLLM,
    PlannedSubQuestionLLM,
    PlannerOutput,
    ReporterSynthesisOutput,
    SourceRecord,
    SourceType,
    SubQuestion,
    SubQuestionStatus,
    SubQuestionUpdateLLM,
    VerdictStatus,
    VerificationRecord,
    VerifiedClaimLLM,
    VerifierEvidenceLLM,
    VerifierOutput,
)
from p04_research.providers.base import RawSearchItem, SearchResponse, UsageDelta
from p04_research.verification_rules import normalize_whitespace_nfc


def clean_question_title(question: str, max_len: int | None = None) -> str:
    """Sorunun başındaki/sonundaki tırnakları temizler; varsayılan olarak tam başlığı korur (kesmez)."""
    cleaned = normalize_whitespace_nfc(question).strip("“”\"'«» ")
    if max_len is None or len(cleaned) <= max_len:
        return cleaned
    return cleaned


@dataclass(frozen=True, slots=True)
class DemoPresetInfo:
    key: str
    label_tr: str
    label_en: str
    default_question_tr: str
    default_question_en: str
    description_tr: str


DEMO_PRESETS: list[DemoPresetInfo] = [
    DemoPresetInfo(
        key="library_pilot",
        label_tr="1 · Sınıf Demosu: Kampüs Kütüphanesi Pilotu (Karşı Kanıt & 2. Tur)",
        label_en="1 · Classroom Demo: Campus Library Pilot (Counter-Evidence & Round 2)",
        default_question_tr=(
            "Kampüs kütüphanesi pilotunda bekleme süresi ve hafta sonu ziyaretleri nasıl değişti?"
        ),
        default_question_en=(
            "How did waiting time and weekend visits change in the campus library pilot?"
        ),
        description_tr=(
            "1. turda hafta içi bekleme süresi (8 -> 6 dk) desteklenir; hafta sonu iki katı "
            "iddiası kanıt yetersiz kalır. Standart modda 2. tur turnike sayımı (100 -> 120) ile "
            "C2 'Kaynak karşı çıkıyor' kararına döner."
        ),
    ),
    DemoPresetInfo(
        key="conflicting_dining",
        label_tr="2 · Çelişen Kaynaklar: Yemekhane Akşam Doluluk Oranı",
        label_en="2 · Conflicting Sources: Dining Hall Evening Occupancy",
        default_question_tr=(
            "Üniversite yemekhanesi uzatılmış akşam saatlerinde doluluk oranı kaç oldu?"
        ),
        default_question_en=(
            "What was the evening occupancy rate during extended dining hall hours?"
        ),
        description_tr=(
            "Aynı hafta için iki bağımsız kaynaktan (%68 ve %41) bağdaşmayan ölçümler gelir; "
            "özet kesin hüküm vermez ve iki alıntıyı yan yana gösterir."
        ),
    ),
    DemoPresetInfo(
        key="bilingual_videos",
        label_tr="3 · İki Dilli Kanıt: İngilizce Makaleden Türkçe Rapor",
        label_en="3 · Bilingual Evidence: English Study to Turkish Report",
        default_question_tr=(
            "Altyazılı ders videoları vize sınavı başarısını nasıl etkiledi?"
        ),
        default_question_en=(
            "How did captioned lecture videos affect midterm exam performance?"
        ),
        description_tr=(
            "İngilizce özgün alıntıyı değiştirmeden korur; Türkçe başlık ve açıklayıcı not ile "
            "rapor üretir."
        ),
    ),
    DemoPresetInfo(
        key="missing_date_and_duplicate",
        label_tr="4 · Eksik Tarih, Kopya Haber ve Nedensellik Uyarısı",
        label_en="4 · Unknown Date, Duplicate Copy & Causation Warning",
        default_question_tr=(
            "Kampüs bisiklet paylaşım pilotunda aylık sürüş sayısı nedir ve kaynaklar bağımsız mı?"
        ),
        default_question_en=(
            "What is the monthly ride count in the campus bike-share pilot and are sources independent?"
        ),
        description_tr=(
            "Yayın tarihi 'Bilinmiyor' gösterilir, aynı haberin kopyası tek bağımsız kanıt sayılır "
            "ve korelasyondan nedensellik çıkarımı işaretlenir."
        ),
    ),
    DemoPresetInfo(
        key="zero_results",
        label_tr="5 · Sıfır Sonuç Senaryosu: Kanıt Bulunamayan Konu",
        label_en="5 · Zero Results Scenario: Topic With No Evidence",
        default_question_tr=(
            "Kuzey Kampüsü 2029 ışınlanma istasyonu günlük yolcu sayısı nedir?"
        ),
        default_question_en=(
            "What is the daily passenger count at the North Campus 2029 teleportation station?"
        ),
        description_tr=(
            "Teknik arıza olmadan 0 arama sonucu döndüğünde uydurma kaynak üretmeden dürüst "
            "sonuç raporu gösterir."
        ),
    ),
]


def detect_demo_scenario(text: str) -> str:
    """Soru veya sorgu metninden hangi deterministik kurmaca senaryonun çalışacağını seçer."""
    lower = normalize_whitespace_nfc(text).lower()
    if any(k in lower for k in ("kütüphane", "kutuphane", "library", "bekleme", "turnike", "weekend visits")):
        return "library_pilot"
    if any(k in lower for k in ("yemekhane", "dining", "doluluk", "occupancy")):
        return "conflicting_dining"
    if any(k in lower for k in ("altyazı", "altyazi", "captioned", "video", "vize", "midterm")):
        return "bilingual_videos"
    if any(k in lower for k in ("bisiklet", "bike", "sürüş", "surus", "bağımsız mı", "bagimsiz mi")):
        return "missing_date_and_duplicate"
    if any(k in lower for k in ("ışınlanma", "isinlanma", "teleportation", "2029", "sıfır sonuç", "sifir sonuc")):
        return "zero_results"
    if any(k in lower for k in ("yapay zek", "ai ", "ajan", "iş dünyası", "is dunyasi", "çalışan", "calisan", "istihdam")):
        return "ai_workforce_2026"
    return "custom_dynamic"


class MockDemoProvider:
    """Çevrimdışı demo için hem LLMProvider hem SearchProvider sözleşmesini uygular."""

    _S1_LIBRARY_TEXT = (
        "Pilot uygulama boyunca hafta içi danışma masasında ölçülen ortalama bekleme süresi "
        "8 dakikadan 6 dakikaya düştü (n=420 işlem, Ekim 2026). Ölçüm yalnızca hafta içi "
        "09.00-17.00 danışma kuyruğunu kapsadı; hafta sonu hakkında doğrudan sayım yapılmadı. "
        "Ön değerlendirme notu: Yeni akşam saatleriyle birlikte hafta sonu ziyaretleri iki "
        "katına çıkmış olabilir; kesin turnike sayımı henüz eklenmedi."
    )
    _S2_LIBRARY_TEXT = (
        "Hafta sonu turnike sayımına göre pilot öncesi ortalama 100 olan hafta sonu ziyaret "
        "sayısı, pilot döneminde 120 olarak ölçüldü (+%20 artış, 4 hafta sonu ortalaması)."
    )

    _S1_DINING_TEXT = (
        "İşletme Müdürlüğü Turnike Kaydı (12-16 Ekim 2026, 18.00-20.30): Uzatılmış akşam "
        "saatlerinde merkez yemekhane salon doluluk oranı ortalama %68 olarak ölçüldü (n=5 akşam)."
    )
    _S2_DINING_TEXT = (
        "Öğrenci Gözlem Ekibi Sayımı (12-16 Ekim 2026, 18.00-20.30): Aynı tarih aralığında "
        "merkez yemekhane masa başı gözlem sayımında doluluk oranı ortalama %41 olarak ölçüldü (n=5 akşam)."
    )

    _S1_BILINGUAL_TEXT = (
        "In the fall 2025 randomized cohort (n=310), students provided with captioned lecture "
        "videos scored 6.4 percentage points higher on the midterm exam than the control group."
    )

    _S1_BIKE_TEXT = (
        "Kampüs bisiklet paylaşım pilotunda Ekim ayında toplam 1.840 sürüş kaydedildi. "
        "Gözlem notu: Bisikleti sık kullanan öğrencilerin dönem sonu notlarının daha yüksek "
        "olduğu görüldü, ancak ders çalışma alışkanlıkları kontrol edilmedi."
    )

    _S1_AI_WORKFORCE_TEXT = (
        "2026 kurumsal saha araştırmasında (n=1.250 şirket), yapay zekâ ajanlarını iş akışına "
        "entegre eden ekiplerde rutin veri işleme ve raporlama süresi ortalama %35 kısaldı. "
        "Ön değerlendirme notu: Bazı sektör yorumcuları yapay zekâ ajanlarının 2026 sonunda "
        "tam zamanlı insan çalışanların yarısının (%50) yerini tamamen alabileceğini öne sürdü; "
        "ancak resmi bordro ve istihdam sayımları henüz bu ön nota eklenmedi."
    )
    _S2_AI_WORKFORCE_TEXT = (
        "2026 Kurumsal Bordro ve İstihdam İzleme Verisi (n=1.250 şirket): Yapay zekâ ajanı "
        "kullanan şirketlerde tam zamanlı çalışan sayısında %50 azalma yaşanmadı; toplam kadro "
        "yalnızca %4 daralırken çalışanların %62'sinin görev tanımı ajan denetimi ve karar "
        "doğrulama yönünde dönüştü."
    )

    _S1_CUSTOM_TEXT = (
        "2026 saha ön inceleme kaydına göre (n=380 gözlem), incelenen uygulamada birincil süreç "
        "verimliliği ortalama %28 iyileşti. Ön değerlendirme notu: İlk aşamada toplam kapasite "
        "veya etki oranının iki katına (%100 artışa) ulaşmış olabileceği tahmin edildi; ancak "
        "doğrudan resmi sayım tablosu henüz bu ön rapora eklenmedi."
    )
    _S2_CUSTOM_TEXT = (
        "2026 Doğrudan Ölçüm ve Denetim Tablosu (n=380 gözlem): Resmi sayım kayıtlarına göre "
        "toplam etki oranı iki katına (%100) çıkmadı; başlangıçta 100 birim olan endeks değeri "
        "uygulama sonunda 118 birim (+%18 artış) olarak ölçüldü."
    )

    def plan(
        self,
        question: str,
        language: str,
        depth: str,
        max_initial_queries: int,
    ) -> tuple[PlannerOutput, UsageDelta]:
        scenario = detect_demo_scenario(question)
        is_tr = language == "tr"
        clean_q = clean_question_title(question)
        is_std = depth == "standard"

        if scenario == "library_pilot":
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        "Bekleme süresi ve ziyaret sayısı hangi yöntem ve zaman aralığında ölçüldü?"
                        if is_tr
                        else "How and during which hours were waiting time and visit counts measured?"
                    ),
                    scope_note="Ölçüm kapsamı / Measurement scope",
                    search_intent="Ölçüm yöntemi ve kapsanan saat aralığını doğrula" if is_tr else "Verify measurement method and hours",
                    required_evidence="Birincil danışma ve turnike kayıt yöntemi" if is_tr else "Primary help-desk and turnstile methodology",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "Pilot döneminde hafta içi danışma bekleme süresi nasıl değişti?"
                        if is_tr
                        else "How did weekday help-desk waiting time change during the pilot?"
                    ),
                    scope_note="Dakika cinsinden ortalama / Mean in minutes",
                    search_intent="Pilot öncesi ve sırasındaki bekleme süresi dakikalarını karşılaştır" if is_tr else "Compare wait time minutes",
                    required_evidence="İşlem sayısı (n=) içeren danışma ölçümü" if is_tr else "Help-desk measurement with sample size",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Hafta sonu ziyaret sayısı pilot döneminde gerçekten iki katına çıktı mı?"
                        if is_tr
                        else "Did weekend visits actually double during the pilot period?"
                    ),
                    scope_note="Turnike sayımı / Turnstile count",
                    search_intent="Ön tahmin ile doğrudan hafta sonu turnike sayımını karşılaştır" if is_tr else "Compare estimate vs weekend turnstile count",
                    required_evidence="Hafta sonu turnike sayım değerleri" if is_tr else "Weekend turnstile counts",
                ),
            ]
            if is_std:
                sqs.append(
                    PlannedSubQuestionLLM(
                        sq_id="SQ4",
                        text=(
                            "Danışma kaydı ve turnike ölçümlerinin kapsam sınırlılıkları nelerdir?"
                            if is_tr
                            else "What are the scope limitations of the help-desk and turnstile measurements?"
                        ),
                        scope_note="Kapsam ve genellenebilirlik",
                        search_intent="Ölçüm zaman aralığı ve ön tahmin eksiklerini tespit et" if is_tr else "Identify scope and estimate gaps",
                        required_evidence="Kaynak kapsam notları" if is_tr else "Source scope notes",
                    )
                )
            out = PlannerOutput(
                plan_summary=(
                    "Hafta içi bekleme süresi ölçümünü, ölçüm kapsamını, hafta sonu ziyaret sayımını ve sınırlılıkları ayrı alt sorularda inceler."
                    if is_tr
                    else "Examines weekday waiting time, measurement scope, weekend visit counts, and limitations across distinct sub-questions."
                ),
                scope_notes=(
                    "Dönem: Ekim 2026 pilot uygulaması · Ölçü birimleri: dakika (bekleme) ve ziyaret sayısı (turnike)."
                    if is_tr
                    else "Period: October 2026 pilot · Units: minutes (wait time) and visit count (turnstile)."
                ),
                research_purpose=(
                    "Kampüs kütüphanesi pilot uygulamasının bekleme süresi ve hafta sonu ziyaret hacmi üzerindeki doğrulanmış etkisini sınamak."
                    if is_tr
                    else "Evaluate verified pilot impacts on campus library waiting times and weekend visit volumes."
                ),
                timeframe="Ekim 2026 / October 2026",
                key_concepts=["bekleme süresi", "hafta sonu ziyaretleri", "turnike sayımı", "pilot uygulaması"],
                sub_questions=sqs,
                initial_queries=["kampus kutuphanesi pilotu bekleme suresi hafta sonu ziyaretleri"],
            )

        elif scenario == "ai_workforce_2026":
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        "Hangi iş görevleri ve kurumsal süreçler yapay zekâ ajanlarıyla otomatikleştiriliyor?"
                        if is_tr
                        else "Which work tasks and corporate workflows are being automated by AI agents?"
                    ),
                    scope_note="Görev ve iş akışı otomasyonu",
                    search_intent="Otomatikleştirilen görev türlerini ve kurumsal entegrasyon alanlarını belirle" if is_tr else "Identify automated tasks and workflows",
                    required_evidence="Saha araştırması veya kurumsal vaka ölçümü" if is_tr else "Field study or enterprise measurement",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "2026 yılında ölçülmüş istihdam ve tam zamanlı kadro etkileri nelerdir?"
                        if is_tr
                        else "What are the measured employment and full-time headcount impacts in 2026?"
                    ),
                    scope_note="Resmi bordro ve kadro sayımı",
                    search_intent="İnsan çalışanların %50 ikame edilip edilmediğini bordro kayıtlarıyla test et" if is_tr else "Test replacement claims against payroll data",
                    required_evidence="Bordro, istihdam veya kadro değişim yüzdesi" if is_tr else "Payroll or headcount change metrics",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Verimlilik, görev tamamlama süresi ve operasyonel maliyet etkileri nelerdir?"
                        if is_tr
                        else "What are the measured productivity, task-time, and cost impacts?"
                    ),
                    scope_note="Süre ve verimlilik oranı (%)",
                    search_intent="İşlem süresi kısalması ve maliyet verilerini çıkar" if is_tr else "Extract task-time and cost ratios",
                    required_evidence="Örneklem büyüklüğü belli nicel verimlilik ölçümü" if is_tr else "Quantitative productivity measurement with sample size",
                ),
            ]
            if is_std:
                sqs.extend([
                    PlannedSubQuestionLLM(
                        sq_id="SQ4",
                        text=(
                            "Sektörlere ve çalışan rollerine göre dönüşüm farklılıkları nelerdir?"
                            if is_tr
                            else "How do transformation impacts differ across sectors and employee roles?"
                        ),
                        scope_note="Rol dönüşümü ve sektörel ayrım",
                        search_intent="Görev tanımı değişimi ve sektörel dağılımı araştır" if is_tr else "Inspect role shift and sector differences",
                        required_evidence="Rol dönüşümü veya sektörel istatistik" if is_tr else "Role transformation or sector statistics",
                    ),
                    PlannedSubQuestionLLM(
                        sq_id="SQ5",
                        text=(
                            "Mevcut 2026 kanıtlarının, örneklem kapsamının ve gelecek tahminlerinin sınırlılıkları nelerdir?"
                            if is_tr
                            else "What are the limitations of current 2026 evidence, samples, and future projections?"
                        ),
                        scope_note="Ön tahmin vs ampirik veri ayrımı",
                        search_intent="Spekülatif yorumlar ile ölçülmüş bordro verilerini ayır" if is_tr else "Separate projections from measured payroll logs",
                        required_evidence="Metodoloji notları ve karşıt kanıtlar" if is_tr else "Methodology notes and counter-evidence",
                    ),
                ])
            out = PlannerOutput(
                plan_summary=(
                    "Yapay zekâ ajanlarının 2026'da iş görevlerine, ölçülmüş istihdama, verimliliğe, sektörel rol dönüşümüne ve kanıt sınırlılıklarına etkisini inceler."
                    if is_tr
                    else "Examines 2026 AI agent effects on tasks, measured employment, productivity, sector/role shifts, and evidence limitations."
                ),
                scope_notes="2026 Kurumsal Saha ve Bordro Verileri (n=1.250 şirket)",
                research_purpose=(
                    "2026 yılında yapay zekâ ajanlarının insan çalışanları ne ölçüde ikame ettiğini veya rollerini dönüştürdüğünü kanıtlara dayanarak belirlemek."
                    if is_tr
                    else "Determine from verifiable 2026 evidence whether AI agents replace or transform human roles."
                ),
                timeframe="2026",
                key_concepts=["yapay zekâ ajanları", "istihdam etkisi", "verimlilik", "görev otomasyonu", "rol dönüşümü"],
                sub_questions=sqs,
                initial_queries=["2026 yapay zeka ajanlari is dunyasi etkisi insan calisanlar saha raporu"],
            )

        elif scenario == "conflicting_dining":
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        "Uzatılmış akşam saatlerinde yemekhane doluluk oranı farklı kaynaklarda kaç ölçüldü?"
                        if is_tr
                        else "What occupancy rate was measured across sources during extended evening hours?"
                    ),
                    scope_note="Yüzde (%) doluluk / Percentage occupancy",
                    search_intent="Tüm bağımsız doluluk ölçümlerini topla" if is_tr else "Collect independent occupancy measurements",
                    required_evidence="Tarih ve saat aralığı belirtilmiş ölçüm" if is_tr else "Timestamped occupancy measurement",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "İki ölçüm aynı tarih ve saat dilimini mi kapsıyor?"
                        if is_tr
                        else "Do both measurements cover the same date and time window?"
                    ),
                    scope_note="Kapsam denkliği / Scope equivalence",
                    search_intent="Ölçüm tarih/saat tutarlılığını kontrol et" if is_tr else "Verify scope equivalence",
                    required_evidence="Tarih ve saat bilgisi" if is_tr else "Date and hour metadata",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Kaynaklar arasındaki sayısal farkı açıklayan yöntem farklılığı nedir?"
                        if is_tr
                        else "What methodological difference explains the numeric gap between sources?"
                    ),
                    scope_note="Turnike vs masa başı gözlem",
                    search_intent="Ölçüm yöntemlerini karşılaştır" if is_tr else "Compare measurement methods",
                    required_evidence="Yöntem açıklaması" if is_tr else "Methodology description",
                ),
            ]
            out = PlannerOutput(
                plan_summary=(
                    "Aynı tarih ve saat aralığındaki iki farklı doluluk ölçümünü karşılaştırır."
                    if is_tr
                    else "Compares two occupancy measurements for the exact same date and evening window."
                ),
                scope_notes="12-16 Ekim 2026 · 18.00-20.30",
                research_purpose="Yemekhane akşam doluluk oranındaki çelişkiyi ve yöntem farklarını ortaya koymak.",
                timeframe="12-16 Ekim 2026",
                key_concepts=["yemekhane doluluk oranı", "turnike kaydı", "gözlem sayımı"],
                sub_questions=sqs,
                initial_queries=["universite yemekhanesi uzatilmis aksam saatleri doluluk orani"],
            )

        elif scenario == "bilingual_videos":
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        "Altyazılı ders videoları vize sınav puanlarını kaç puan etkiledi?"
                        if is_tr
                        else "By how many points did captioned lecture videos affect midterm exam scores?"
                    ),
                    scope_note="Yüzde puan farkı / Percentage point difference",
                    search_intent="Sınav puan farkını özgün çalışmadan bul" if is_tr else "Find exam score difference",
                    required_evidence="Kontrol grubu karşılaştırmalı puan farkı" if is_tr else "Controlled score comparison",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "Araştırma tasarımı ve örneklem büyüklüğü nedir?"
                        if is_tr
                        else "What are the study design and sample size?"
                    ),
                    scope_note="Kohort ve kontrol grubu / Cohort & control",
                    search_intent="Deney tasarımı ve katılımcı sayısını doğrula" if is_tr else "Verify study design and sample size",
                    required_evidence="Kohort bilgisi ve n= değeri" if is_tr else "Cohort design and n value",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Çalışmanın genellenebilirlik ve kapsam sınırları nelerdir?"
                        if is_tr
                        else "What are the generalizability and scope limits of the study?"
                    ),
                    scope_note="Dönem ve ders kapsamı",
                    search_intent="Dönem ve ek değişken sınırlarını incele" if is_tr else "Examine study limits",
                    required_evidence="Dönem ve kapsam bilgisi" if is_tr else "Term and scope details",
                ),
            ]
            out = PlannerOutput(
                plan_summary=(
                    "Altyazılı ders videolarının vize başarısına etkisini özgün İngilizce araştırmadan inceler."
                    if is_tr
                    else "Examines the effect of captioned lecture videos on midterm scores from the original study."
                ),
                scope_notes="Güz 2025 / Fall 2025 · n=310",
                research_purpose="Altyazılı ders videolarının vize sınavı başarısı üzerindeki deneysel etkisini değerlendirmek.",
                timeframe="Güz 2025 / Fall 2025",
                key_concepts=["altyazılı ders videoları", "vize sınavı", "kohort çalışma"],
                sub_questions=sqs,
                initial_queries=["altyazili ders videolari vize sinavi basarisi captioned lecture videos"],
            )

        elif scenario == "missing_date_and_duplicate":
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        "Bisiklet paylaşım pilotunda Ekim ayında kaç sürüş kaydedildi?"
                        if is_tr
                        else "How many rides were recorded in October in the bike-share pilot?"
                    ),
                    scope_note="Sürüş sayısı / Ride count",
                    search_intent="Aylık sürüş sayısını ve yayın tarihini kontrol et" if is_tr else "Verify monthly ride count and date",
                    required_evidence="Birincil sürüş kaydı" if is_tr else "Primary ride log",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "Bulunan kaynaklar birbirinden bağımsız mı?"
                        if is_tr
                        else "Are the retrieved sources independent of each other?"
                    ),
                    scope_note="Kopya kontrolü / Duplicate check",
                    search_intent="Metin parmak izi ile kopya haberleri ayır" if is_tr else "Detect duplicate syndicated text",
                    required_evidence="İçerik karşılaştırması" if is_tr else "Content comparison",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Bisiklet kullanımı dönem sonu notlarını doğrudan artırıyor mu?"
                        if is_tr
                        else "Does bike usage directly cause higher end-of-term grades?"
                    ),
                    scope_note="Korelasyon vs nedensellik / Correlation vs causation",
                    search_intent="Karıştırıcı değişkenlerin kontrol edilip edilmediğini denetle" if is_tr else "Check confounding controls",
                    required_evidence="Kontrollü nedensellik kanıtı" if is_tr else "Controlled causal evidence",
                ),
            ]
            out = PlannerOutput(
                plan_summary=(
                    "Aylık sürüş sayısını, yayın tarihini, kaynak bağımsızlığını ve nedensellik sınırını denetler."
                    if is_tr
                    else "Checks monthly ride count, publication date, source independence, and causation limits."
                ),
                scope_notes="Ekim 2026 · Kopya kaynak ve korelasyon kontrolü",
                research_purpose="Bisiklet pilotu verilerini kaynak bağımsızlığı ve nedensellik açısından denetlemek.",
                timeframe="Ekim 2026",
                key_concepts=["bisiklet paylaşım", "kaynak bağımsızlığı", "nedensellik"],
                sub_questions=sqs,
                initial_queries=["kampus bisiklet paylasim pilotu aylik surus sayisi"],
            )

        elif scenario == "zero_results":
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=clean_q,
                    scope_note="Doğrudan kayıt taraması / Direct record scan",
                    search_intent="Doğrulanabilir birincil kayıt ara" if is_tr else "Search primary records",
                    required_evidence="Resmî veya birincil kayıt" if is_tr else "Official or primary record",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "Güvenilir birincil kaynaklarda ölçüm veya tarih bilgisi bulunuyor mu?"
                        if is_tr
                        else "Are measurements or dates available in primary sources?"
                    ),
                    scope_note="Kaynak mevcudiyeti / Source availability",
                    search_intent="Yayın tarihi ve ölçüm varlığını doğrula" if is_tr else "Check date and measurement availability",
                    required_evidence="Yayın tarihli ölçüm kaydı" if is_tr else "Dated measurement record",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Kayıt bulunamaması durumunda hangi veriler eksiktir?"
                        if is_tr
                        else "What data gaps prevent answering the question?"
                    ),
                    scope_note="Kanıt boşluğu",
                    search_intent="Eksik kanıt alanlarını adlandır" if is_tr else "Name missing evidence",
                    required_evidence="Kayıt inceleme sonucu" if is_tr else "Search audit outcome",
                ),
            ]
            out = PlannerOutput(
                plan_summary=(
                    "Soruya ilişkin doğrulanabilir kayıt olup olmadığını denetler."
                    if is_tr
                    else "Checks whether verifiable records exist for the question."
                ),
                scope_notes="Çevrimdışı kurmaca demo taraması / Offline mock demo scan",
                research_purpose=f"'{clean_q}' sorusu için doğrulanabilir kayıt varlığını sınamak.",
                timeframe="Belirtilmedi / Unspecified",
                key_concepts=[],
                sub_questions=sqs,
                initial_queries=[clean_q],
            )

        else:
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        f"{clean_q} kapsamında ölçülen birincil süreç değişimi nedir?"
                        if is_tr
                        else f"What is the primary measured change regarding {clean_q}?"
                    ),
                    scope_note="Doğrudan saha ölçümü / Direct field measurement",
                    search_intent="Birincil ölçüm sonucunu bul" if is_tr else "Find primary measurement",
                    required_evidence="Saha ölçüm verisi" if is_tr else "Field measurement data",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "İlk rapordaki iki kat (%100) artış tahmini resmi sayımlarla doğrulandı mı?"
                        if is_tr
                        else "Was the initial doubling (+100%) estimate confirmed by official counts?"
                    ),
                    scope_note="Ön tahmin vs resmi sayım / Estimate vs official count",
                    search_intent="Ön tahmin ile resmi sayım tablosunu karşılaştır" if is_tr else "Compare estimate vs official table",
                    required_evidence="Resmî endeks/sayım tablosu" if is_tr else "Official count table",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Mevcut ölçümlerin örneklem ve kapsam sınırlılıkları nelerdir?"
                        if is_tr
                        else "What are the sample and scope limitations of existing measurements?"
                    ),
                    scope_note="Kapsam sınırı",
                    search_intent="Ölçüm kapsamını ve açık noktaları tespit et" if is_tr else "Identify scope boundaries",
                    required_evidence="Örneklem ve metodoloji bilgisi" if is_tr else "Sample and method details",
                ),
            ]
            out = PlannerOutput(
                plan_summary=(
                    f"'{clean_q}' konusu için saha verimlilik ölçümünü, resmi sayım uyumunu ve kapsam sınırlarını inceler."
                    if is_tr
                    else f"Examines field efficiency metrics, official count verification, and scope limits for '{clean_q}'."
                ),
                scope_notes="2026 Kurmaca Eğitim Veri Seti (n=380 gözlem)",
                research_purpose=f"'{clean_q}' sorusunu doğrulanmış saha kayıtlarıyla sınamak.",
                timeframe="2026",
                key_concepts=["saha ölçümü", "resmî sayım", "kapsam sınırı"],
                sub_questions=sqs,
                initial_queries=[f"ozel_konu_ilk_tarama {clean_q}"],
            )

        return out, UsageDelta(model_calls=1, measurement_type="unmeasured")

    def search(self, query: str, max_results: int, depth: str) -> SearchResponse:
        scenario = detect_demo_scenario(query)
        q_lower = normalize_whitespace_nfc(query).lower()

        if scenario == "library_pilot":
            if "turnike" in q_lower or "sayım" in q_lower or "sayimi" in q_lower or "count" in q_lower:
                items = [
                    RawSearchItem(
                        url="https://kurmaca.p04.local/kutuphane-turnike-sayimi-2026",
                        title="Kütüphane Turnike Sayım Raporu — Ekim 2026",
                        publisher="Kampüs Bilgi İşlem ve Kütüphane Birimi (Kurmaca)",
                        content=self._S2_LIBRARY_TEXT,
                        published_date="2026-10-28",
                        language="tr",
                        provider="mock_demo",
                        is_mock=True,
                        evidence_level=EvidenceLevel.PRIMARY_RECORD,
                        retrieval_method="primary_record",
                        source_type=SourceType.CORPORATE_PRIMARY_DATA,
                    )
                ]
            else:
                items = [
                    RawSearchItem(
                        url="https://kurmaca.p04.local/kutuphane-danisma-kaydi?utm_source=demo",
                        title="Kampüs Kütüphanesi Pilot Danışma Kaydı",
                        publisher="Üniversite Kütüphane Müdürlüğü (Kurmaca)",
                        content=self._S1_LIBRARY_TEXT,
                        published_date="2026-10-15",
                        language="tr",
                        provider="mock_demo",
                        is_mock=True,
                        evidence_level=EvidenceLevel.PRIMARY_RECORD,
                        retrieval_method="primary_record",
                        source_type=SourceType.OFFICIAL_INSTITUTION,
                    )
                ]
            return SearchResponse(query=query, items=items[:max_results], search_calls=1)

        if scenario == "ai_workforce_2026":
            if "bordro" in q_lower or "istihdam_sayimi" in q_lower:
                items = [
                    RawSearchItem(
                        url="https://kurmaca.p04.local/2026-kurumsal-bordro-istihdam-izleme",
                        title="2026 Kurumsal Bordro ve İstihdam İzleme Verisi",
                        publisher="Çalışma Ekonomisi ve Otomasyon Gözlemevi (Kurmaca)",
                        content=self._S2_AI_WORKFORCE_TEXT,
                        published_date="2026-10-22",
                        language="tr",
                        provider="mock_demo",
                        is_mock=True,
                        evidence_level=EvidenceLevel.PRIMARY_RECORD,
                        retrieval_method="primary_record",
                        source_type=SourceType.RESEARCH_ORG_REPORT,
                    )
                ]
            else:
                items = [
                    RawSearchItem(
                        url="https://kurmaca.p04.local/2026-yapay-zeka-ajanlari-saha-raporu",
                        title="Küresel İş Gücü ve Yapay Zekâ Ajanları Saha Raporu 2026",
                        publisher="Dijital Dönüşüm Araştırma Enstitüsü (Kurmaca)",
                        content=self._S1_AI_WORKFORCE_TEXT,
                        published_date="2026-09-14",
                        language="tr",
                        provider="mock_demo",
                        is_mock=True,
                        evidence_level=EvidenceLevel.PRIMARY_RECORD,
                        retrieval_method="primary_record",
                        source_type=SourceType.RESEARCH_ORG_REPORT,
                    )
                ]
            return SearchResponse(query=query, items=items[:max_results], search_calls=1)

        if scenario == "conflicting_dining":
            items = [
                RawSearchItem(
                    url="https://kurmaca.p04.local/yemekhane-turnike-kaydi",
                    title="Yemekhane İşletme Müdürlüğü Akşam Turnike Kaydı",
                    publisher="Üniversite İşletme Müdürlüğü (Kurmaca)",
                    content=self._S1_DINING_TEXT,
                    published_date="2026-10-18",
                    language="tr",
                    provider="mock_demo",
                    is_mock=True,
                    evidence_level=EvidenceLevel.PRIMARY_RECORD,
                    retrieval_method="primary_record",
                    source_type=SourceType.OFFICIAL_INSTITUTION,
                ),
                RawSearchItem(
                    url="https://kurmaca.p04.local/yemekhane-ogrenci-gozlem",
                    title="Öğrenci Gözlem Ekibi Yemekhane Sayım Notu",
                    publisher="Bağımsız Öğrenci Gözlem Ekibi (Kurmaca)",
                    content=self._S2_DINING_TEXT,
                    published_date="2026-10-19",
                    language="tr",
                    provider="mock_demo",
                    is_mock=True,
                    evidence_level=EvidenceLevel.PRIMARY_RECORD,
                    retrieval_method="primary_record",
                    source_type=SourceType.CORPORATE_PRIMARY_DATA,
                ),
            ]
            return SearchResponse(query=query, items=items[:max_results], search_calls=1)

        if scenario == "bilingual_videos":
            items = [
                RawSearchItem(
                    url="https://kurmaca.p04.local/captioned-lectures-study-2025",
                    title="Impact of Captioned Lecture Videos on Undergraduate Midterm Scores",
                    publisher="Journal of Campus Learning Studies (Kurmaca)",
                    content=self._S1_BILINGUAL_TEXT,
                    published_date="2026-02-10",
                    language="en",
                    provider="mock_demo",
                    is_mock=True,
                    evidence_level=EvidenceLevel.OFFICIAL_DOC,
                    retrieval_method="full_text",
                    source_type=SourceType.ACADEMIC_PEER_REVIEWED,
                )
            ]
            return SearchResponse(query=query, items=items[:max_results], search_calls=1)

        if scenario == "missing_date_and_duplicate":
            items = [
                RawSearchItem(
                    url="https://kurmaca.p04.local/bisiklet-pilot-bulteni",
                    title="Kampüs Bisiklet Paylaşım Pilotu Ekim Özeti",
                    publisher="Ulaşım Koordinatörlüğü (Kurmaca)",
                    content=self._S1_BIKE_TEXT,
                    published_date=None,
                    language="tr",
                    provider="mock_demo",
                    is_mock=True,
                    evidence_level=EvidenceLevel.SECONDARY_REPORT,
                    retrieval_method="full_text",
                    source_type=SourceType.OFFICIAL_INSTITUTION,
                ),
                RawSearchItem(
                    url="https://kurmaca-ayna.p04.local/bisiklet-haberi-kopya",
                    title="Kopya Bülten: Kampüs Bisiklet Paylaşım Özeti",
                    publisher="Kampüs Haber Portalı (Kurmaca)",
                    content=self._S1_BIKE_TEXT,
                    published_date="2026-10-20",
                    language="tr",
                    provider="mock_demo",
                    is_mock=True,
                    evidence_level=EvidenceLevel.SECONDARY_REPORT,
                    retrieval_method="search_snippet",
                    source_type=SourceType.NEWS_BLOG_OPINION,
                ),
            ]
            return SearchResponse(query=query, items=items[:max_results], search_calls=1)

        if scenario == "zero_results":
            return SearchResponse(query=query, items=[], search_calls=1)

        if "resmi_sayim_takip" in q_lower:
            items = [
                RawSearchItem(
                    url="https://kurmaca.p04.local/2026-resmi-olcum-denetim-tablosu",
                    title="2026 Doğrudan Ölçüm ve Denetim Tablosu",
                    publisher="Bağımsız Veri Denetim Birimi (Kurmaca)",
                    content=self._S2_CUSTOM_TEXT,
                    published_date="2026-10-25",
                    language="tr",
                    provider="mock_demo",
                    is_mock=True,
                    evidence_level=EvidenceLevel.PRIMARY_RECORD,
                    retrieval_method="primary_record",
                    source_type=SourceType.RESEARCH_ORG_REPORT,
                )
            ]
        else:
            items = [
                RawSearchItem(
                    url="https://kurmaca.p04.local/2026-saha-on-inceleme-kaydi",
                    title="2026 Saha Ön İnceleme ve Verimlilik Kaydı",
                    publisher="Araştırma Gözlem Merkezi (Kurmaca)",
                    content=self._S1_CUSTOM_TEXT,
                    published_date="2026-10-10",
                    language="tr",
                    provider="mock_demo",
                    is_mock=True,
                    evidence_level=EvidenceLevel.PRIMARY_RECORD,
                    retrieval_method="primary_record",
                    source_type=SourceType.RESEARCH_ORG_REPORT,
                )
            ]
        return SearchResponse(query=query, items=items[:max_results], search_calls=1)

    def extract_claims(
        self,
        question: str,
        language: str,
        round_no: int,
        sub_questions: list[SubQuestion],
        sources: list[SourceRecord],
        chunks: list[EvidenceChunk],
        existing_claims: list[ClaimRecord],
    ) -> tuple[AnalystOutput, UsageDelta]:
        scenario = detect_demo_scenario(question)
        is_tr = language == "tr"
        claims: list[ExtractedClaimLLM] = []

        if scenario == "library_pilot":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            s2_chunk = next((c for c in chunks if c.source_id == "S2"), None)

            if round_no == 1 and s1_chunk is not None:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C1",
                        sq_id="SQ2",
                        statement=(
                            "Pilot döneminde hafta içi danışma masasında ölçülen ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü (n=420 işlem)."
                            if is_tr
                            else "During the pilot, weekday mean help-desk waiting time decreased from 8 minutes to 6 minutes (n=420 transactions)."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=(
                            "Pilot uygulama boyunca hafta içi danışma masasında ölçülen ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü (n=420 işlem, Ekim 2026)."
                        ),
                        initial_quote_language="tr",
                    ),
                    ExtractedClaimLLM(
                        claim_id="C2",
                        sq_id="SQ3",
                        statement=(
                            "Hafta sonu ziyaretleri pilot döneminde iki katına çıktı."
                            if is_tr
                            else "Weekend visits doubled during the pilot period."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=(
                            "Ön değerlendirme notu: Yeni akşam saatleriyle birlikte hafta sonu ziyaretleri iki katına çıkmış olabilir; kesin turnike sayımı henüz eklenmedi."
                        ),
                        initial_quote_language="tr",
                    ),
                ]
            elif round_no >= 2 and s2_chunk is not None:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C3",
                        sq_id="SQ3",
                        statement=(
                            "Turnike sayımına göre hafta sonu ziyaret sayısı pilot öncesi ortalama 100 iken pilot döneminde 120 oldu (+%20 artış)."
                            if is_tr
                            else "According to turnstile counts, weekend visits increased from a pre-pilot mean of 100 to 120 during the pilot (+20%)."
                        ),
                        initial_source_id="S2",
                        initial_chunk_id=s2_chunk.chunk_id,
                        initial_quote=self._S2_LIBRARY_TEXT,
                        initial_quote_language="tr",
                    )
                ]

        elif scenario == "ai_workforce_2026":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            s2_chunk = next((c for c in chunks if c.source_id == "S2"), None)
            if round_no == 1 and s1_chunk is not None:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C1",
                        sq_id="SQ3",
                        statement=(
                            "2026 kurumsal saha araştırmasında (n=1.250 şirket), yapay zekâ ajanlarını iş akışına entegre eden ekiplerde rutin veri işleme ve raporlama süresi ortalama %35 kısaldı."
                            if is_tr
                            else "In the 2026 corporate field study (n=1,250 companies), teams integrating AI agents reduced routine data processing and reporting time by 35%."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=(
                            "2026 kurumsal saha araştırmasında (n=1.250 şirket), yapay zekâ ajanlarını iş akışına entegre eden ekiplerde rutin veri işleme ve raporlama süresi ortalama %35 kısaldı."
                        ),
                        initial_quote_language="tr",
                    ),
                    ExtractedClaimLLM(
                        claim_id="C2",
                        sq_id="SQ2",
                        statement=(
                            "Yapay zekâ ajanları 2026 sonunda tam zamanlı insan çalışanların yarısının (%50) yerini tamamen aldı."
                            if is_tr
                            else "AI agents completely replaced half (50%) of full-time human employees by the end of 2026."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=(
                            "Ön değerlendirme notu: Bazı sektör yorumcuları yapay zekâ ajanlarının 2026 sonunda tam zamanlı insan çalışanların yarısının (%50) yerini tamamen alabileceğini öne sürdü; ancak resmi bordro ve istihdam sayımları henüz bu ön nota eklenmedi."
                        ),
                        initial_quote_language="tr",
                    ),
                ]
            elif round_no >= 2 and s2_chunk is not None:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C3",
                        sq_id="SQ4" if any(sq.sq_id == "SQ4" for sq in sub_questions) else "SQ3",
                        statement=(
                            "2026 bordro kayıtlarına göre toplam kadro yalnızca %4 daralırken çalışanların %62'sinin görev tanımı ajan denetimi ve karar doğrulama yönünde dönüştü."
                            if is_tr
                            else "According to 2026 payroll logs, total headcount shrank by only 4%, while 62% of employees transitioned toward agent oversight and verification roles."
                        ),
                        initial_source_id="S2",
                        initial_chunk_id=s2_chunk.chunk_id,
                        initial_quote=self._S2_AI_WORKFORCE_TEXT,
                        initial_quote_language="tr",
                    )
                ]

        elif scenario == "conflicting_dining":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            if s1_chunk is not None and not existing_claims:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C1",
                        sq_id="SQ1",
                        statement=(
                            "12-16 Ekim 2026 uzatılmış akşam saatlerinde (18.00-20.30) merkez yemekhane ortalama doluluk oranı %68 oldu."
                            if is_tr
                            else "Mean central dining hall occupancy during extended evening hours (18:00-20:30, Oct 12-16, 2026) was 68%."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=self._S1_DINING_TEXT,
                        initial_quote_language="tr",
                    )
                ]

        elif scenario == "bilingual_videos":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            if s1_chunk is not None and not existing_claims:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C1",
                        sq_id="SQ1",
                        statement=(
                            "Güz 2025 rastgele kohortunda (n=310) altyazılı ders videoları sağlanan öğrenciler, kontrol grubuna kıyasla vize sınavında 6,4 puan daha yüksek puan aldı."
                            if is_tr
                            else "In the fall 2025 randomized cohort (n=310), students given captioned lecture videos scored 6.4 percentage points higher on the midterm exam."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=self._S1_BILINGUAL_TEXT,
                        initial_quote_language="en",
                        explanatory_note=(
                            "Türkçe açıklama: Güz 2025 rastgele kohortunda (n=310) altyazılı ders videosu verilen öğrenciler, kontrol grubuna göre vizede 6,4 puan daha yüksek not almıştır."
                            if is_tr
                            else None
                        ),
                    )
                ]

        elif scenario == "missing_date_and_duplicate":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            if s1_chunk is not None and not existing_claims:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C1",
                        sq_id="SQ1",
                        statement=(
                            "Kampüs bisiklet paylaşım pilotunda Ekim ayında toplam 1.840 sürüş kaydedildi."
                            if is_tr
                            else "A total of 1,840 rides were recorded in October in the campus bike-share pilot."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=(
                            "Kampüs bisiklet paylaşım pilotunda Ekim ayında toplam 1.840 sürüş kaydedildi."
                        ),
                        initial_quote_language="tr",
                    ),
                    ExtractedClaimLLM(
                        claim_id="C2",
                        sq_id="SQ3",
                        statement=(
                            "Bisikleti sık kullanmak öğrencilerin dönem sonu notlarını doğrudan yükseltmektedir."
                            if is_tr
                            else "Frequent bike usage directly increases students' end-of-term grades."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=(
                            "Gözlem notu: Bisikleti sık kullanan öğrencilerin dönem sonu notlarının daha yüksek olduğu görüldü, ancak ders çalışma alışkanlıkları kontrol edilmedi."
                        ),
                        initial_quote_language="tr",
                    ),
                ]

        elif scenario == "custom_dynamic":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            s2_chunk = next((c for c in chunks if c.source_id == "S2"), None)
            if round_no == 1 and s1_chunk is not None:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C1",
                        sq_id="SQ1",
                        statement=(
                            "2026 saha ön inceleme kaydına göre (n=380 gözlem) birincil süreç verimliliği ortalama %28 iyileşti."
                            if is_tr
                            else "According to the 2026 field log (n=380), primary process efficiency improved by 28%."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=(
                            "2026 saha ön inceleme kaydına göre (n=380 gözlem), incelenen uygulamada birincil süreç verimliliği ortalama %28 iyileşti."
                        ),
                        initial_quote_language="tr",
                    ),
                    ExtractedClaimLLM(
                        claim_id="C2",
                        sq_id="SQ2",
                        statement=(
                            "İncelenen uygulamada toplam etki oranı iki katına (%100 artışa) ulaştı."
                            if is_tr
                            else "Total impact rate doubled (+100% increase) in the examined implementation."
                        ),
                        initial_source_id="S1",
                        initial_chunk_id=s1_chunk.chunk_id,
                        initial_quote=(
                            "Ön değerlendirme notu: İlk aşamada toplam kapasite veya etki oranının iki katına (%100 artışa) ulaşmış olabileceği tahmin edildi; ancak doğrudan resmi sayım tablosu henüz bu ön rapora eklenmedi."
                        ),
                        initial_quote_language="tr",
                    ),
                ]
            elif round_no >= 2 and s2_chunk is not None:
                claims = [
                    ExtractedClaimLLM(
                        claim_id="C3",
                        sq_id="SQ2",
                        statement=(
                            "Resmi sayım tablosuna göre başlangıçta 100 birim olan endeks değeri uygulama sonunda 118 birim (+%18 artış) olarak ölçüldü."
                            if is_tr
                            else "According to official counts, the index rose from 100 to 118 units (+18%)."
                        ),
                        initial_source_id="S2",
                        initial_chunk_id=s2_chunk.chunk_id,
                        initial_quote=self._S2_CUSTOM_TEXT,
                        initial_quote_language="tr",
                    )
                ]

        return AnalystOutput(claims=claims), UsageDelta(model_calls=1, measurement_type="unmeasured")

    def verify_claims(
        self,
        question: str,
        language: str,
        round_no: int,
        claims: list[ClaimRecord],
        sources: list[SourceRecord],
        chunks: list[EvidenceChunk],
    ) -> tuple[VerifierOutput, UsageDelta]:
        scenario = detect_demo_scenario(question)
        is_tr = language == "tr"
        verifications: list[VerifiedClaimLLM] = []

        if scenario == "library_pilot":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            s2_chunk = next((c for c in chunks if c.source_id == "S2"), None)

            for claim in claims:
                if claim.claim_id == "C1" and s1_chunk is not None:
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id="C1",
                            verdict=VerdictStatus.SUPPORTED,
                            rationale=(
                                "S1 danışma kaydı aynı dönem (Ekim 2026) ve aynı ölçüm için 8 dakikadan 6 dakikaya düşüşü doğrudan doğruluyor (n=420)."
                                if is_tr
                                else "S1 help-desk log directly supports the reduction from 8 to 6 minutes for October 2026 (n=420)."
                            ),
                            numeric_context="8 dk -> 6 dk (n=420 işlem, hafta içi 09.00-17.00, Ekim 2026)",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id="S1",
                                    chunk_id=s1_chunk.chunk_id,
                                    quote=(
                                        "Pilot uygulama boyunca hafta içi danışma masasında ölçülen ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü (n=420 işlem, Ekim 2026)."
                                    ),
                                    role=EvidenceRole.SUPPORTING,
                                    quote_language="tr",
                                )
                            ],
                        )
                    )
                elif claim.claim_id == "C2":
                    if round_no == 1 or s2_chunk is None:
                        verifications.append(
                            VerifiedClaimLLM(
                                claim_id="C2",
                                verdict=VerdictStatus.INSUFFICIENT,
                                rationale=(
                                    "S1'deki ifade yalnızca kesin sayım eklenmemiş bir ön tahmindir ('çıkmış olabilir'). Ayrıca bu kaynak yalnızca hafta içini ölçtüğü için hafta sonu hakkında sonuç çıkaramadık."
                                    if is_tr
                                    else "S1 only contains a preliminary estimate ('may have doubled') without turnstile counts, and primary logs covered weekdays only."
                                ),
                                missing_info=(
                                    "Hafta sonu doğrudan turnike ziyaret sayımı eksik."
                                    if is_tr
                                    else "Direct weekend turnstile visit count is missing."
                                ),
                                follow_up_query="kampus kutuphanesi pilotu hafta sonu turnike ziyaret sayimi 2026",
                                claim_nature="future_projection",
                                evidence_items=[
                                    VerifierEvidenceLLM(
                                        source_id="S1",
                                        chunk_id=s1_chunk.chunk_id if s1_chunk else claim.initial_chunk_id,
                                        quote=(
                                            "Ön değerlendirme notu: Yeni akşam saatleriyle birlikte hafta sonu ziyaretleri iki katına çıkmış olabilir; kesin turnike sayımı henüz eklenmedi."
                                        ),
                                        role=EvidenceRole.CONTEXT,
                                        quote_language="tr",
                                    )
                                ],
                            )
                        )
                    else:
                        verifications.append(
                            VerifiedClaimLLM(
                                claim_id="C2",
                                verdict=VerdictStatus.CONTRADICTED,
                                rationale=(
                                    "S2'deki doğrudan turnike sayımı hafta sonu ziyaretlerinin 100'den 120'ye çıktığını (+%20) gösteriyor; iki katına (+%100) çıkış iddiasıyla çelişiyor."
                                    if is_tr
                                    else "Direct turnstile counts in S2 show weekend visits rose from 100 to 120 (+20%), contradicting the claim that visits doubled."
                                ),
                                numeric_context="100 -> 120 ziyaret (+%20, 4 hafta sonu ortalaması)",
                                evidence_items=[
                                    VerifierEvidenceLLM(
                                        source_id="S2",
                                        chunk_id=s2_chunk.chunk_id,
                                        quote=self._S2_LIBRARY_TEXT,
                                        role=EvidenceRole.CONTRADICTING,
                                        quote_language="tr",
                                    )
                                ],
                            )
                        )
                elif claim.claim_id == "C3" and s2_chunk is not None:
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id="C3",
                            verdict=VerdictStatus.SUPPORTED,
                            rationale=(
                                "S2 turnike sayım raporu hafta sonu ziyaretlerinin 100'den 120'ye çıktığını (+%20) kesin sayımla doğruluyor."
                                if is_tr
                                else "S2 turnstile report verifies weekend visits increased from 100 to 120 (+20%)."
                            ),
                            numeric_context="100 -> 120 ziyaret (+%20 artış, 4 hafta sonu)",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id="S2",
                                    chunk_id=s2_chunk.chunk_id,
                                    quote=self._S2_LIBRARY_TEXT,
                                    role=EvidenceRole.SUPPORTING,
                                    quote_language="tr",
                                )
                            ],
                        )
                    )

        elif scenario == "ai_workforce_2026":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            s2_chunk = next((c for c in chunks if c.source_id == "S2"), None)
            for claim in claims:
                if claim.claim_id == "C1" and s1_chunk is not None:
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id="C1",
                            verdict=VerdictStatus.SUPPORTED,
                            rationale=(
                                "S1 saha araştırması (n=1.250 şirket), yapay zekâ ajanlarıyla rutin veri işleme ve raporlama süresinin ortalama %35 kısaldığını doğrudan doğruluyor."
                                if is_tr
                                else "S1 field study (n=1,250 companies) directly verifies a 35% reduction in routine data processing and reporting time."
                            ),
                            numeric_context="%35 süre kısalması (n=1.250 şirket, 2026)",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id="S1",
                                    chunk_id=s1_chunk.chunk_id,
                                    quote=(
                                        "2026 kurumsal saha araştırmasında (n=1.250 şirket), yapay zekâ ajanlarını iş akışına entegre eden ekiplerde rutin veri işleme ve raporlama süresi ortalama %35 kısaldı."
                                    ),
                                    role=EvidenceRole.SUPPORTING,
                                    quote_language="tr",
                                )
                            ],
                        )
                    )
                elif claim.claim_id == "C2":
                    if round_no == 1 or s2_chunk is None:
                        verifications.append(
                            VerifiedClaimLLM(
                                claim_id="C2",
                                verdict=VerdictStatus.INSUFFICIENT,
                                rationale=(
                                    "S1'deki '%50 yerini alabileceği' ifadesi yalnızca resmi bordro verisi bulunmayan bir yorum/tahmindir ('alabileceğini öne sürdü'). Kesin hüküm için bordro ve istihdam sayımı gereklidir."
                                    if is_tr
                                    else "The 50% replacement claim in S1 is only an unverified projection without payroll data."
                                ),
                                missing_info=(
                                    "2026 kurumsal bordro ve tam zamanlı istihdam sayım verisi eksik."
                                    if is_tr
                                    else "2026 corporate payroll and full-time headcount logs are missing."
                                ),
                                follow_up_query="2026 yapay zeka ajanlari kurumsal bordro istihdam_sayimi",
                                claim_nature="future_projection",
                                evidence_items=[
                                    VerifierEvidenceLLM(
                                        source_id="S1",
                                        chunk_id=s1_chunk.chunk_id if s1_chunk else claim.initial_chunk_id,
                                        quote=(
                                            "Ön değerlendirme notu: Bazı sektör yorumcuları yapay zekâ ajanlarının 2026 sonunda tam zamanlı insan çalışanların yarısının (%50) yerini tamamen alabileceğini öne sürdü; ancak resmi bordro ve istihdam sayımları henüz bu ön nota eklenmedi."
                                        ),
                                        role=EvidenceRole.CONTEXT,
                                        quote_language="tr",
                                    )
                                ],
                            )
                        )
                    else:
                        verifications.append(
                            VerifiedClaimLLM(
                                claim_id="C2",
                                verdict=VerdictStatus.CONTRADICTED,
                                rationale=(
                                    "S2 resmi bordro izleme verisi (n=1.250 şirket), çalışanların yarısının (%50) işten çıkarıldığı iddiasını çürütüyor: toplam kadro yalnızca %4 daralmış, çalışanların %62'si denetim rollerine geçmiştir."
                                    if is_tr
                                    else "S2 payroll records contradict the 50% replacement claim: total headcount shrank by only 4%, while 62% shifted to oversight roles."
                                ),
                                numeric_context="%4 kadro daralması, %62 rol dönüşümü (n=1.250 şirket, 2026)",
                                evidence_items=[
                                    VerifierEvidenceLLM(
                                        source_id="S2",
                                        chunk_id=s2_chunk.chunk_id,
                                        quote=self._S2_AI_WORKFORCE_TEXT,
                                        role=EvidenceRole.CONTRADICTING,
                                        quote_language="tr",
                                    )
                                ],
                            )
                        )
                elif claim.claim_id == "C3" and s2_chunk is not None:
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id="C3",
                            verdict=VerdictStatus.SUPPORTED,
                            rationale=(
                                "S2 bordro ve istihdam kaydı %4 kadro daralmasını ve %62 görev dönüşümünü doğrudan doğruluyor."
                                if is_tr
                                else "S2 payroll records verify a 4% headcount reduction and 62% role transformation."
                            ),
                            numeric_context="%4 daralma, %62 rol dönüşümü (n=1.250 şirket)",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id="S2",
                                    chunk_id=s2_chunk.chunk_id,
                                    quote=self._S2_AI_WORKFORCE_TEXT,
                                    role=EvidenceRole.SUPPORTING,
                                    quote_language="tr",
                                )
                            ],
                        )
                    )

        elif scenario == "conflicting_dining":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            s2_chunk = next((c for c in chunks if c.source_id == "S2"), None)
            if claims and s1_chunk and s2_chunk:
                verifications.append(
                    VerifiedClaimLLM(
                        claim_id=claims[0].claim_id,
                        verdict=VerdictStatus.CONFLICTING,
                        rationale=(
                            "İki bağımsız kaynak aynı tarih (12-16 Ekim 2026) ve aynı saat aralığı (18.00-20.30) için bağdaşmayan doluluk oranları (%68 vs %41) bildirmektedir."
                            if is_tr
                            else "Two independent sources report incompatible occupancy rates (68% vs 41%) for the same dates and evening hours."
                        ),
                        numeric_context="%68 (turnike) vs %41 (gözlem, n=5 akşam)",
                        evidence_items=[
                            VerifierEvidenceLLM(
                                source_id="S1",
                                chunk_id=s1_chunk.chunk_id,
                                quote=self._S1_DINING_TEXT,
                                role=EvidenceRole.CONFLICTING,
                                quote_language="tr",
                            ),
                            VerifierEvidenceLLM(
                                source_id="S2",
                                chunk_id=s2_chunk.chunk_id,
                                quote=self._S2_DINING_TEXT,
                                role=EvidenceRole.CONFLICTING,
                                quote_language="tr",
                            ),
                        ],
                    )
                )

        elif scenario == "bilingual_videos":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            if claims and s1_chunk:
                verifications.append(
                    VerifiedClaimLLM(
                        claim_id=claims[0].claim_id,
                        verdict=VerdictStatus.SUPPORTED,
                        rationale=(
                            "S1 özgün İngilizce çalışma kaydı (n=310), altyazılı video grubunun vizede 6,4 puan daha yüksek aldığını doğrudan doğrulamaktadır."
                            if is_tr
                            else "S1 primary English study record (n=310) directly confirms a 6.4 percentage point higher midterm score."
                        ),
                        numeric_context="+6,4 yüzde puan (n=310, Güz 2025)",
                        evidence_items=[
                            VerifierEvidenceLLM(
                                source_id="S1",
                                chunk_id=s1_chunk.chunk_id,
                                quote=self._S1_BILINGUAL_TEXT,
                                role=EvidenceRole.SUPPORTING,
                                quote_language="en",
                                explanatory_note=(
                                    "Türkçe açıklama: Güz 2025 rastgele kohortunda (n=310) altyazılı ders videosu verilen öğrenciler, kontrol grubuna kıyasla vize sınavında 6,4 puan daha yüksek puan aldı."
                                    if is_tr
                                    else None
                                ),
                            )
                        ],
                    )
                )

        elif scenario == "missing_date_and_duplicate":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            for claim in claims:
                if claim.claim_id == "C1" and s1_chunk:
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id="C1",
                            verdict=VerdictStatus.SUPPORTED,
                            rationale=(
                                "S1 Ekim ayında 1.840 sürüşü doğrudan doğruluyor. Not: S2, S1 ile birebir aynı bültenin kopyasıdır ve ikinci bağımsız kanıt olarak sayılmamıştır; S1 yayın tarihi bilinmiyor."
                                if is_tr
                                else "S1 supports 1,840 rides in October. Note: S2 is a verbatim syndicated copy of S1 and is not counted as independent confirmation; S1 publication date is unknown."
                            ),
                            numeric_context="1.840 sürüş (Ekim ayı)",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id="S1",
                                    chunk_id=s1_chunk.chunk_id,
                                    quote="Kampüs bisiklet paylaşım pilotunda Ekim ayında toplam 1.840 sürüş kaydedildi.",
                                    role=EvidenceRole.SUPPORTING,
                                    quote_language="tr",
                                )
                            ],
                        )
                    )
                elif claim.claim_id == "C2" and s1_chunk:
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id="C2",
                            verdict=VerdictStatus.INSUFFICIENT,
                            rationale=(
                                "Kaynak yalnızca gözlemsel bir birliktelik bildiriyor ve ders çalışma alışkanlıklarının kontrol edilmediğini belirtiyor; nedensellik kurulamaz."
                                if is_tr
                                else "Source only notes an observational correlation without controlling for study habits; causation cannot be established."
                            ),
                            missing_info=(
                                "Karıştırıcı değişkenleri kontrol eden deneysel veya eşleştirilmiş karşılaştırma eksik."
                                if is_tr
                                else "Controlled comparison accounting for confounding variables is missing."
                            ),
                            causation_warning=(
                                "Gözlemsel beraberlikten neden-sonuç çıkarımı uyarısı: Ders çalışma alışkanlıkları kontrol edilmemiştir."
                                if is_tr
                                else "Correlation-not-causation warning: Study habits were not controlled."
                            ),
                            claim_nature="opinion_commentary",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id="S1",
                                    chunk_id=s1_chunk.chunk_id,
                                    quote=(
                                        "Gözlem notu: Bisikleti sık kullanan öğrencilerin dönem sonu notlarının daha yüksek olduğu görüldü, ancak ders çalışma alışkanlıkları kontrol edilmedi."
                                    ),
                                    role=EvidenceRole.CONTEXT,
                                    quote_language="tr",
                                )
                            ],
                        )
                    )

        elif scenario == "custom_dynamic":
            s1_chunk = next((c for c in chunks if c.source_id == "S1"), None)
            s2_chunk = next((c for c in chunks if c.source_id == "S2"), None)
            for claim in claims:
                if claim.claim_id == "C1" and s1_chunk is not None:
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id="C1",
                            verdict=VerdictStatus.SUPPORTED,
                            rationale=(
                                "S1 saha ön inceleme kaydı (n=380 gözlem) birincil süreç verimliliğinde %28 iyileşmeyi doğrudan doğruluyor."
                                if is_tr
                                else "S1 field log (n=380) directly confirms the 28% primary efficiency gain."
                            ),
                            numeric_context="%28 iyileşme (n=380 gözlem, 2026)",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id="S1",
                                    chunk_id=s1_chunk.chunk_id,
                                    quote=(
                                        "2026 saha ön inceleme kaydına göre (n=380 gözlem), incelenen uygulamada birincil süreç verimliliği ortalama %28 iyileşti."
                                    ),
                                    role=EvidenceRole.SUPPORTING,
                                    quote_language="tr",
                                )
                            ],
                        )
                    )
                elif claim.claim_id == "C2":
                    if round_no == 1 or s2_chunk is None:
                        verifications.append(
                            VerifiedClaimLLM(
                                claim_id="C2",
                                verdict=VerdictStatus.INSUFFICIENT,
                                rationale=(
                                    "S1'deki iki kat (%100) artış ifadesi yalnızca resmi sayım eklenmemiş bir ön tahmindir ('ulaşmış olabileceği tahmin edildi')."
                                    if is_tr
                                    else "The doubling (+100%) statement in S1 is only an unverified preliminary estimate."
                                ),
                                missing_info=(
                                    "Doğrudan resmi sayım ve endeks tablosu eksik."
                                    if is_tr
                                    else "Direct official count and index table is missing."
                                ),
                                follow_up_query="2026 resmi_sayim_takip dogrudan olcum tablosu",
                                claim_nature="future_projection",
                                evidence_items=[
                                    VerifierEvidenceLLM(
                                        source_id="S1",
                                        chunk_id=s1_chunk.chunk_id if s1_chunk else claim.initial_chunk_id,
                                        quote=(
                                            "Ön değerlendirme notu: İlk aşamada toplam kapasite veya etki oranının iki katına (%100 artışa) ulaşmış olabileceği tahmin edildi; ancak doğrudan resmi sayım tablosu henüz bu ön rapora eklenmedi."
                                        ),
                                        role=EvidenceRole.CONTEXT,
                                        quote_language="tr",
                                    )
                                ],
                            )
                        )
                    else:
                        verifications.append(
                            VerifiedClaimLLM(
                                claim_id="C2",
                                verdict=VerdictStatus.CONTRADICTED,
                                rationale=(
                                    "S2 resmi ölçüm tablosu endeksin 100'den 118'e (+%18) çıktığını gösteriyor; iki katına (%100) çıktığı iddiasını çürütüyor."
                                    if is_tr
                                    else "S2 official counts show the index rose from 100 to 118 (+18%), contradicting the doubling claim."
                                ),
                                numeric_context="100 -> 118 birim (+%18, n=380 gözlem)",
                                evidence_items=[
                                    VerifierEvidenceLLM(
                                        source_id="S2",
                                        chunk_id=s2_chunk.chunk_id,
                                        quote=self._S2_CUSTOM_TEXT,
                                        role=EvidenceRole.CONTRADICTING,
                                        quote_language="tr",
                                    )
                                ],
                            )
                        )
                elif claim.claim_id == "C3" and s2_chunk is not None:
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id="C3",
                            verdict=VerdictStatus.SUPPORTED,
                            rationale=(
                                "S2 resmi sayım tablosu 100 -> 118 birim (+%18) artışını doğrudan doğruluyor."
                                if is_tr
                                else "S2 official count table directly confirms the 100 -> 118 (+18%) increase."
                            ),
                            numeric_context="100 -> 118 birim (+%18 artış)",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id="S2",
                                    chunk_id=s2_chunk.chunk_id,
                                    quote=self._S2_CUSTOM_TEXT,
                                    role=EvidenceRole.SUPPORTING,
                                    quote_language="tr",
                                )
                            ],
                        )
                    )

        return VerifierOutput(verifications=verifications), UsageDelta(
            model_calls=1, measurement_type="unmeasured"
        )

    def synthesize_report(
        self,
        question: str,
        language: str,
        sub_questions: list[SubQuestion],
        claims: list[ClaimRecord],
        verifications: list[VerificationRecord],
        sources: list[SourceRecord],
    ) -> tuple[ReporterSynthesisOutput, UsageDelta]:
        scenario = detect_demo_scenario(question)
        is_tr = language == "tr"
        final_map = {v.claim_id: v for v in verifications if v.is_final}
        clean_q = clean_question_title(question)

        # Her alt sorunun durumunu gerçek SUPPORTED / CONFLICTING sonuçlarına göre hesapla
        claims_by_id = {c.claim_id: c for c in claims}
        sq_updates: list[SubQuestionUpdateLLM] = []
        for sq in sub_questions:
            sq_vers = [
                v for cid, v in final_map.items()
                if cid in claims_by_id and claims_by_id[cid].sq_id == sq.sq_id
            ]
            has_sup = any(v.verdict == VerdictStatus.SUPPORTED for v in sq_vers)
            has_conf = any(v.verdict == VerdictStatus.CONFLICTING for v in sq_vers)
            has_gap = any(
                v.verdict in (VerdictStatus.INSUFFICIENT, VerdictStatus.IRRELEVANT)
                for v in sq_vers
            )
            if has_conf:
                st = SubQuestionStatus.CONFLICTING
            elif has_sup and not has_gap:
                st = SubQuestionStatus.ANSWERED
            elif has_sup:
                st = SubQuestionStatus.PARTIALLY_ANSWERED
            elif any(v.verdict == VerdictStatus.CONTRADICTED for v in sq_vers) and not has_gap:
                st = SubQuestionStatus.ANSWERED
            else:
                st = SubQuestionStatus.UNANSWERED
            sq_updates.append(SubQuestionUpdateLLM(sq_id=sq.sq_id, status=st))

        if scenario == "library_pilot":
            c2_ver = final_map.get("C2")
            has_r2 = c2_ver is not None and c2_ver.verdict == VerdictStatus.CONTRADICTED
            if has_r2:
                short = (
                    "Kampüs kütüphanesi pilotunda hafta içi ortalama bekleme süresi 8 dakikadan 6 dakikaya düşmüştür [S1]. "
                    "Hafta sonu ziyaretlerinin iki katına çıktığı iddiası ise doğrudan turnike sayımıyla çürütülmüştür: "
                    "hafta sonu ziyaretleri pilot öncesi ortalama 100 iken pilot döneminde 120 (+%20) olarak ölçülmüştür [S2]."
                    if is_tr
                    else "In the campus library pilot, weekday mean waiting time dropped from 8 to 6 minutes [S1]. "
                    "The claim that weekend visits doubled is contradicted by turnstile counts: weekend visits rose from 100 to 120 (+20%) [S2]."
                )
                # SQ1 ölçüm kapsamı S1/S2'de açıklandığı için yanıtlandı kabul edilir
                for u in sq_updates:
                    if u.sq_id == "SQ1":
                        u.status = SubQuestionStatus.ANSWERED
                open_qs = [sq.text for sq in sub_questions if sq.sq_id == "SQ4"]
                assess = (
                    "Pilot verileri hafta içi bekleme süresinde iyileşme ve hafta sonu ziyaretlerinde ılımlı artış göstermektedir; ilk ön tahminin abartılı olduğu ikinci tur turnike kaydıyla düzeltilmiştir."
                    if is_tr
                    else "Pilot logs show improved weekday wait times and moderate weekend visit growth; the initial doubling estimate was corrected by Round 2 turnstile counts."
                )
            else:
                short = (
                    "Hızlı modda hafta içi ortalama bekleme süresinin 8 dakikadan 6 dakikaya düştüğü doğrulandı [S1]. "
                    "Ancak hafta sonu ziyaretlerinin iki katına çıktığına dair yalnızca bir ön tahmin bulunduğu ve S1 yalnızca hafta içini ölçtüğü için hafta sonu artışı doğrulanamadı."
                    if is_tr
                    else "In Fast mode, weekday mean wait time dropping from 8 to 6 minutes is verified [S1]. "
                    "Weekend visit doubling remains unverified because S1 only offers an uncounted estimate and covered weekdays only."
                )
                open_qs = [
                    "Hafta sonu için doğrudan turnike ziyaret sayımı verisi nedir?"
                    if is_tr
                    else "What are the direct weekend turnstile visit counts?"
                ]
                assess = (
                    "Hızlı mod yalnızca ilk kaynağı incelediği için hafta sonu ziyaret hacmi açık soru olarak kalmıştır."
                    if is_tr
                    else "Fast mode only audited the initial source, leaving weekend visit volume open."
                )
            title = (
                "Kampüs Kütüphanesi Pilotu: Bekleme Süresi ve Hafta Sonu Ziyaretleri Raporu"
                if is_tr
                else "Campus Library Pilot: Waiting Time and Weekend Visits Report"
            )

        elif scenario == "ai_workforce_2026":
            c2_ver = final_map.get("C2")
            has_r2 = c2_ver is not None and c2_ver.verdict == VerdictStatus.CONTRADICTED
            title = (
                f"Araştırma Raporu: {clean_q}"
                if is_tr
                else f"Research Report: {clean_q}"
            )
            if has_r2:
                short = (
                    "2026 kurumsal saha verilerine göre yapay zekâ ajanları rutin veri işleme ve raporlama süresini ortalama %35 kısaltmıştır [S1]. "
                    "Buna karşılık insan çalışanların %50'sinin işten çıkarılacağı yönündeki ön tahmin resmi bordro verileriyle çürütülmüştür: "
                    "toplam kadro yalnızca %4 daralırken çalışanların %62'sinin görev tanımı ajan denetimi ve karar doğrulama yönünde dönüşmüştür [S2]. "
                    "Sektörel dağılım ve uzun vadeli sınırlılıklar için bağımsız alt sorular henüz açık kalmaktadır."
                    if is_tr
                    else "In 2026 corporate data, AI agents reduced routine data processing and reporting time by 35% [S1]. "
                    "However, the 50% replacement claim is contradicted by payroll logs: headcount shrank by only 4%, while 62% of employees shifted to agent oversight roles [S2]."
                )
                open_qs = [
                    sq.text for sq in sub_questions
                    if any(u.sq_id == sq.sq_id and u.status == SubQuestionStatus.UNANSWERED for u in sq_updates)
                ]
                assess = (
                    "Mevcut kanıtlar tam otomasyon yoluyla kitlesel ikameden ziyade iş akışı hızlanması ve görev dönüşümü yönündedir; tekil sektörel karşılaştırmalar ek veri gerektirir."
                    if is_tr
                    else "Available evidence points toward workflow acceleration and role transformation rather than mass replacement."
                )
            else:
                short = (
                    "Hızlı modda yapay zekâ ajanlarının rutin veri işleme ve raporlama süresini %35 kısalttığı doğrulandı [S1]. "
                    "İnsan çalışanların %50'sinin yerini aldığı iddiası ise S1'de yalnızca resmi bordro sayımı olmayan bir ön tahmin olarak geçtiği için doğrulanamadı."
                    if is_tr
                    else "In Fast mode, a 35% reduction in routine processing time is verified [S1]. The 50% human replacement claim remains unverified due to missing payroll logs."
                )
                open_qs = ["2026 kurumsal bordro ve tam zamanlı istihdam sayım verileri nedir?"]
                assess = "Hızlı mod bordro verisine inemediği için istihdam etkisi doğrulanamamıştır."

        elif scenario == "conflicting_dining":
            title = (
                "Üniversite Yemekhanesi Akşam Doluluk Oranı Raporu"
                if is_tr
                else "University Dining Hall Evening Occupancy Report"
            )
            short = (
                "Aynı hafta (12-16 Ekim 2026) ve akşam saatleri (18.00-20.30) için iki bağımsız kaynak çelişkili doluluk oranları verdiğinden kesin hüküm kurulamamıştır: "
                "İşletme Müdürlüğü turnike kaydı %68 [S1], Bağımsız Öğrenci Gözlem Ekibi sayımı %41 doluluk bildirmektedir [S2]."
                if is_tr
                else "No definitive occupancy rate can be declared because two independent sources conflict for the exact same week and hours: "
                "management turnstile logs report 68% [S1], while independent student observers report 41% [S2]."
            )
            open_qs = [
                (
                    "Turnike girişleri ile masa başı gözlem sayımı arasındaki yöntem farkı (ör. paket/ayakta giriş) nedir?"
                    if is_tr
                    else "What methodological difference (e.g., takeaway entries vs seated counts) explains the gap between turnstile and observation logs?"
                )
            ]
            assess = "İki bağımsız kayıt doğrudan çeliştiği için ek metodolojik denetim gerekir."

        elif scenario == "bilingual_videos":
            title = (
                "Altyazılı Ders Videoları ve Vize Başarısı Raporu"
                if is_tr
                else "Captioned Lecture Videos and Midterm Performance Report"
            )
            short = (
                "Güz 2025 rastgele kohort çalışmasına göre (n=310), altyazılı ders videosu sağlanan öğrenciler kontrol grubuna kıyasla vize sınavında 6,4 yüzde puan daha yüksek başarı göstermiştir [S1]."
                if is_tr
                else "In the fall 2025 randomized cohort (n=310), students provided with captioned lecture videos scored 6.4 percentage points higher on the midterm exam than the control group [S1]."
            )
            open_qs = [
                sq.text for sq in sub_questions
                if any(u.sq_id == sq.sq_id and u.status == SubQuestionStatus.UNANSWERED for u in sq_updates)
            ]
            assess = "Tek bir İngilizce randomize kohort çalışması (+6,4 puan) doğrulanmıştır; ikinci bağımsız tekrar çalışması bulunmamaktadır."

        elif scenario == "missing_date_and_duplicate":
            title = (
                "Kampüs Bisiklet Paylaşım Pilotu: Sürüş Sayısı ve Kaynak Denetimi"
                if is_tr
                else "Campus Bike-Share Pilot: Ride Counts and Source Audit"
            )
            short = (
                "Ekim ayında 1.840 sürüş kaydedildiği doğrulanmıştır [S1]. Bulunan ikinci bağlantı [S2] S1 ile birebir aynı bültenin kopyası olduğundan tek bağımsız kaynak kabul edilmiştir; S1'in yayın tarihi ise bilinmiyor. Bisiklet kullanımının notları doğrudan yükselttiği iddiası nedensellik kanıtı olmadığı için reddedilmiştir."
                if is_tr
                else "1,840 rides in October is supported [S1]. The second link [S2] is a verbatim syndicated copy of S1 and is not counted as an independent source; S1's publication date is unknown. Causal grade improvement is unsupported."
            )
            open_qs = [
                (
                    "S1 bülteninin kesin yayın tarihi ve not farkında karıştırıcı değişkenleri kontrol eden çalışma nedir?"
                    if is_tr
                    else "What is S1's exact publication date, and are there controlled studies on academic grades?"
                )
            ]
            assess = "Sürüş sayısı desteklense de yayın tarihi eksikliği, kopya kaynak ve nedensellik eksikliği ana sınırlılıklardır."

        elif scenario == "zero_results":
            title = f"Araştırma Raporu: {clean_q}" if is_tr else f"Research Report: {clean_q}"
            short = (
                "Bu soru için doğrulanabilir hiçbir kaynak veya destekli kanıt bulunamadı; eldeki tarama 0 sonuç döndürdü."
                if is_tr
                else "No verifiable sources or supported evidence were found for this question; the search returned 0 results."
            )
            open_qs = [clean_q]
            assess = "Doğrulanabilir hiçbir kaynak bulunamadığı için soru yanıtsız bırakılmıştır."

        else:
            c2_ver = final_map.get("C2")
            has_r2 = c2_ver is not None and c2_ver.verdict == VerdictStatus.CONTRADICTED
            title = f"Araştırma Raporu: {clean_q}" if is_tr else f"Research Report: {clean_q}"
            if has_r2:
                short = (
                    f"'{clean_q}' konusundaki kurmaca saha incelemesinde birincil süreç verimliliğinin ortalama %28 iyileştiği doğrulanmıştır [S1]. "
                    "İlk aşamadaki iki kat (%100) artış tahmini ise resmi sayım tablosuyla çürütülmüş; gerçek endeks değerinin 100'den 118 birime (+%18) yükseldiği ölçülmüştür [S2]."
                    if is_tr
                    else f"For '{clean_q}', primary process efficiency improved by 28% [S1]. The initial doubling (+100%) estimate was contradicted by official counts showing a rise from 100 to 118 units (+18%) [S2]."
                )
                open_qs = [
                    sq.text for sq in sub_questions
                    if any(u.sq_id == sq.sq_id and u.status == SubQuestionStatus.UNANSWERED for u in sq_updates)
                ]
                assess = "Nicel saha verisi mevcuttur ancak kapsam sınırları açısından ek bağımsız kaynaklar önerilir."
            else:
                short = (
                    f"Hızlı modda '{clean_q}' için birincil verimliliğin %28 iyileştiği doğrulandı [S1]. İki kat (%100) artış iddiası ise resmi sayım tablosu olmadığı için kanıt yetersiz olarak işaretlendi."
                    if is_tr
                    else "In Fast mode, a 28% efficiency improvement is verified [S1], while the doubling estimate remains unverified."
                )
                open_qs = ["Doğrudan resmi sayım ve endeks tablosu verisi nedir?"]
                assess = "Hızlı modda resmi sayım tablosu denetlenememiştir."

        return (
            ReporterSynthesisOutput(
                title=title,
                short_answer=short,
                sub_question_updates=sq_updates,
                open_questions=open_qs,
                overall_assessment=assess,
            ),
            UsageDelta(model_calls=1, measurement_type="unmeasured"),
        )
