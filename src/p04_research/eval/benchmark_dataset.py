"""K11 sabit kaynaklı 10 soruluk akademik değerlendirme veri seti."""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(frozen=True, slots=True)
class BenchmarkClaimSpec:
    claim_id: str
    sq_id: str
    statement: str
    source_idx: int
    quote: str
    quote_language: str
    verdict: VerdictStatus
    rationale: str
    role: EvidenceRole
    second_source_idx: int | None = None
    second_quote: str | None = None
    causation_warning: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    topic_label: str
    question: str
    language: str
    sources: list[RawSearchItem]
    claims: list[BenchmarkClaimSpec] = field(default_factory=list)


BENCHMARK_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="Q01",
        topic_label="Kütüphane Bekleme & Turnike",
        question="Kampüs kütüphanesi pilotunda bekleme süresi ve hafta sonu ziyaretleri nasıl değişti?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q01-s1",
                title="Danışma Kaydı (Ekim 2026)",
                publisher="Kütüphane Müdürlüğü",
                content=(
                    "Hafta içi ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü (n=420). "
                    "Ön değerlendirme notu: hafta sonu ziyaretleri iki katına çıkmış olabilir."
                ),
                published_date="2026-10-15",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.PRIMARY_RECORD,
            ),
            RawSearchItem(
                url="https://eval.p04.local/q01-s2",
                title="Hafta Sonu Turnike Sayım Raporu",
                publisher="Bilgi İşlem Birimi",
                content="Hafta sonu turnike sayımı pilot öncesi 100 iken pilot döneminde 120 oldu (+%20).",
                published_date="2026-10-28",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.PRIMARY_RECORD,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="Hafta içi ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü.",
                source_idx=0,
                quote="Hafta içi ortalama bekleme süresi 8 dakikadan 6 dakikaya düştü (n=420).",
                quote_language="tr",
                verdict=VerdictStatus.SUPPORTED,
                rationale="Doğrudan danışma kaydı (n=420) ile desteklenir.",
                role=EvidenceRole.SUPPORTING,
            ),
            BenchmarkClaimSpec(
                claim_id="C2",
                sq_id="SQ2",
                statement="Hafta sonu ziyaretleri iki katına çıktı.",
                source_idx=1,
                quote="Hafta sonu turnike sayımı pilot öncesi 100 iken pilot döneminde 120 oldu (+%20).",
                quote_language="tr",
                verdict=VerdictStatus.CONTRADICTED,
                rationale="Turnike sayımı 100 -> 120 (+%20) artış göstermektedir; iki kat artışı çürütür.",
                role=EvidenceRole.CONTRADICTING,
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q02",
        topic_label="Yemekhane Çelişkisi",
        question="Üniversite yemekhanesi uzatılmış akşam saatlerinde doluluk oranı kaç oldu?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q02-s1",
                title="İşletme Müdürlüğü Turnike Kaydı",
                publisher="İşletme Müdürlüğü",
                content="12-16 Ekim akşam saatlerinde yemekhane doluluk oranı %68 olarak ölçüldü.",
                published_date="2026-10-18",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.PRIMARY_RECORD,
            ),
            RawSearchItem(
                url="https://eval.p04.local/q02-s2",
                title="Öğrenci Gözlem Ekibi Raporu",
                publisher="Öğrenci Gözlem Ekibi",
                content="12-16 Ekim akşam saatlerinde masa başı sayım doluluğu %41 olarak kaydedildi.",
                published_date="2026-10-19",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.PRIMARY_RECORD,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="12-16 Ekim akşam saatlerinde yemekhane doluluk oranı iki bağımsız kaynakta farklıdır.",
                source_idx=0,
                quote="12-16 Ekim akşam saatlerinde yemekhane doluluk oranı %68 olarak ölçüldü.",
                quote_language="tr",
                verdict=VerdictStatus.CONFLICTING,
                rationale="İki bağımsız kaynak (%68 vs %41) aynı kapsam için çelişmektedir.",
                role=EvidenceRole.CONFLICTING,
                second_source_idx=1,
                second_quote="12-16 Ekim akşam saatlerinde masa başı sayım doluluğu %41 olarak kaydedildi.",
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q03",
        topic_label="İki Dilli Altyazılı Video",
        question="Altyazılı ders videoları vize sınavı başarısını nasıl etkiledi?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q03-s1",
                title="Captioned Lecture Videos Cohort Study",
                publisher="Journal of Campus Learning",
                content="In the fall 2025 cohort (n=310), captioned video users scored 6.4 points higher on the midterm.",
                published_date="2026-02-10",
                language="en",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.OFFICIAL_DOC,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="Güz 2025 kohortunda (n=310) altyazılı video grubu vizede 6,4 puan daha yüksek aldı.",
                source_idx=0,
                quote="In the fall 2025 cohort (n=310), captioned video users scored 6.4 points higher on the midterm.",
                quote_language="en",
                verdict=VerdictStatus.SUPPORTED,
                rationale="Özgün İngilizce araştırma bulgusuyla birebir desteklenir.",
                role=EvidenceRole.SUPPORTING,
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q04",
        topic_label="Kopya Haber & Eksik Tarih",
        question="Bisiklet paylaşım pilotunda Ekim sürüş sayısı nedir ve kaynaklar bağımsız mı?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q04-s1",
                title="Bisiklet Pilotu Özeti",
                publisher="Ulaşım Koordinatörlüğü",
                content="Kampüs bisiklet paylaşım pilotunda Ekim ayında 1.840 sürüş kaydedildi.",
                published_date=None,
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.SECONDARY_REPORT,
            ),
            RawSearchItem(
                url="https://eval-mirror.p04.local/q04-s2",
                title="Ayna Haber: Bisiklet Pilotu",
                publisher="Kampüs Haber",
                content="Kampüs bisiklet paylaşım pilotunda Ekim ayında 1.840 sürüş kaydedildi.",
                published_date="2026-10-20",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.SECONDARY_REPORT,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="Kampüs bisiklet paylaşım pilotunda Ekim ayında 1.840 sürüş kaydedildi.",
                source_idx=0,
                quote="Kampüs bisiklet paylaşım pilotunda Ekim ayında 1.840 sürüş kaydedildi.",
                quote_language="tr",
                verdict=VerdictStatus.SUPPORTED,
                rationale="S1 tarafından desteklenir; S2 birebir kopyadır ve bağımsız kanıt sayılmaz.",
                role=EvidenceRole.SUPPORTING,
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q05",
        topic_label="Nedensellik Tuzağı",
        question="Sabah kahvaltısı kampanyası ders başarısını doğrudan artırdı mı?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q05-s1",
                title="Öğrenci Sağlık Anketi",
                publisher="Sağlık Birimi",
                content="Düzenli kahvaltı yapan öğrencilerin not ortalamasının daha yüksek olduğu gözlendi; uyku ve gelir kontrol edilmedi.",
                published_date="2026-05-12",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.SECONDARY_REPORT,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="Sabah kahvaltısı ders başarısını doğrudan artırmaktadır.",
                source_idx=0,
                quote="Düzenli kahvaltı yapan öğrencilerin not ortalamasının daha yüksek olduğu gözlendi; uyku ve gelir kontrol edilmedi.",
                quote_language="tr",
                verdict=VerdictStatus.INSUFFICIENT,
                rationale="Gözlemsel beraberlikte karıştırıcı değişkenler kontrol edilmediğinden nedensellik kurulamaz.",
                role=EvidenceRole.CONTEXT,
                causation_warning="Korelasyon nedensellik anlamına gelmez.",
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q06",
        topic_label="Güneş Enerjisi Çatı Tasarrufu",
        question="Mühendislik binası güneş paneli kurulumu elektrik tüketimini yüzde kaç azalttı?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q06-s1",
                title="Enerji İzleme Raporu 2026",
                publisher="Yapı İşleri Dairesi",
                content="Mühendislik binası çatı panelleri 2026 ilk çeyreğinde şebekeden çekilen elektriği %24,5 azalttı.",
                published_date="2026-04-05",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.PRIMARY_RECORD,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="Çatı güneş panelleri 2026 ilk çeyrekte şebeke elektrik tüketimini %24,5 azalttı.",
                source_idx=0,
                quote="Mühendislik binası çatı panelleri 2026 ilk çeyreğinde şebekeden çekilen elektriği %24,5 azalttı.",
                quote_language="tr",
                verdict=VerdictStatus.SUPPORTED,
                rationale="Yapı İşleri enerji izleme verisiyle birebir doğrulanmıştır.",
                role=EvidenceRole.SUPPORTING,
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q07",
        topic_label="Payda / Örneklem Kontrolü",
        question="Yeni çevrimiçi ders modülü öğrenci tamamlama oranını kaç yaptı?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q07-s1",
                title="Uzaktan Eğitim Modül Raporu",
                publisher="UZEM",
                content="Kayıtlı 200 öğrenciden 150'si modülü tamamladı (%75); ancak yalnızca sınavı çözen 160 kişi hesaba katıldığında oran %93,75'tir.",
                published_date="2026-06-01",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.PRIMARY_RECORD,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="Kayıtlı tüm öğrenciler (n=200) payda alındığında modül tamamlama oranı %75'tir.",
                source_idx=0,
                quote="Kayıtlı 200 öğrenciden 150'si modülü tamamladı (%75); ancak yalnızca sınavı çözen 160 kişi hesaba katıldığında oran %93,75'tir.",
                quote_language="tr",
                verdict=VerdictStatus.SUPPORTED,
                rationale="Payda (n=200) açıkça tanımlandığı için %75 oranı doğrulanmıştır.",
                role=EvidenceRole.SUPPORTING,
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q08",
        topic_label="Kapsam Dışı Ölçüm",
        question="Gece otobüs hattının kış dönemindeki ortalama yolcu sayısı nedir?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q08-s1",
                title="Ulaşım Yaz Dönemi Özeti",
                publisher="Belediye Ulaşım A.Ş.",
                content="Gece hattında Temmuz 2026 yaz döneminde sefer başına ortalama 18 yolcu taşındı; kış verisi ölçülmedi.",
                published_date="2026-08-03",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.OFFICIAL_DOC,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="Gece otobüs hattında kış döneminde sefer başına ortalama 18 yolcu taşınmıştır.",
                source_idx=0,
                quote="Gece hattında Temmuz 2026 yaz döneminde sefer başına ortalama 18 yolcu taşındı; kış verisi ölçülmedi.",
                quote_language="tr",
                verdict=VerdictStatus.INSUFFICIENT,
                rationale="Kayıt yalnızca Temmuz 2026 yaz dönemini kapsamaktadır; kış verisi eksiktir.",
                role=EvidenceRole.CONTEXT,
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q09",
        topic_label="Atık Ayrıştırma Pilotu",
        question="Yurtlar bölgesinde geri dönüşüm pilotu atık oranını nasıl etkiledi?",
        language="tr",
        sources=[
            RawSearchItem(
                url="https://eval.p04.local/q09-s1",
                title="Sıfır Atık Yurt Pilotu",
                publisher="Çevre Koordinatörlüğü",
                content="Ayrıştırılan ambalaj atığı oranı 3 aylık pilotta %19'dan %34'e yükseldi (tartım kaydı).",
                published_date="2026-09-22",
                language="tr",
                provider="fixed_benchmark",
                is_mock=True,
                evidence_level=EvidenceLevel.PRIMARY_RECORD,
            ),
        ],
        claims=[
            BenchmarkClaimSpec(
                claim_id="C1",
                sq_id="SQ1",
                statement="Yurtlar bölgesi pilotunda ayrıştırılan ambalaj atığı oranı %19'dan %34'e yükseldi.",
                source_idx=0,
                quote="Ayrıştırılan ambalaj atığı oranı 3 aylık pilotta %19'dan %34'e yükseldi (tartım kaydı).",
                quote_language="tr",
                verdict=VerdictStatus.SUPPORTED,
                rationale="Doğrudan tartım kaydıyla desteklenmektedir.",
                role=EvidenceRole.SUPPORTING,
            ),
        ],
    ),
    BenchmarkCase(
        case_id="Q10",
        topic_label="Sıfır Sonuç",
        question="Kuzey Kampüsü 2029 ışınlanma istasyonu günlük yolcu sayısı nedir?",
        language="tr",
        sources=[],
        claims=[],
    ),
]


class FixedBenchmarkProvider:
    """K11 çevrimdışı karşılaştırması için sabit kaynaklı deterministik çok ajanlı sağlayıcı."""

    def __init__(self, case: BenchmarkCase) -> None:
        self.case = case

    def plan(
        self,
        question: str,
        language: str,
        depth: str,
        max_initial_queries: int,
    ) -> tuple[PlannerOutput, UsageDelta]:
        sqs = [
            PlannedSubQuestionLLM(
                sq_id=f"SQ{i + 1}",
                text=f"{self.case.topic_label} — alt inceleme {i + 1}",
                scope_note="Sabit değerlendirme kapsamı",
            )
            for i in range(max(2, len(self.case.claims) or 2))
        ]
        return (
            PlannerOutput(
                plan_summary=f"{self.case.case_id} sabit kaynak değerlendirme planı",
                scope_notes="K11 sabit kaynak seti",
                sub_questions=sqs,
                initial_queries=[self.case.question[:80]],
            ),
            UsageDelta(model_calls=1),
        )

    def search(self, query: str, max_results: int, depth: str) -> SearchResponse:
        return SearchResponse(
            query=query,
            items=list(self.case.sources)[:max_results],
            search_calls=1,
        )

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
        if existing_claims or not self.case.claims:
            return AnalystOutput(claims=[]), UsageDelta(model_calls=1)

        out_claims: list[ExtractedClaimLLM] = []
        for spec in self.case.claims:
            sid = f"S{spec.source_idx + 1}"
            ch = next((c for c in chunks if c.source_id == sid), None)
            if ch is None:
                continue
            out_claims.append(
                ExtractedClaimLLM(
                    claim_id=spec.claim_id,
                    sq_id=spec.sq_id if any(s.sq_id == spec.sq_id for s in sub_questions) else "SQ1",
                    statement=spec.statement,
                    initial_source_id=sid,
                    initial_chunk_id=ch.chunk_id,
                    initial_quote=spec.quote,
                    initial_quote_language=spec.quote_language,
                )
            )
        return AnalystOutput(claims=out_claims), UsageDelta(model_calls=1)

    def verify_claims(
        self,
        question: str,
        language: str,
        round_no: int,
        claims: list[ClaimRecord],
        sources: list[SourceRecord],
        chunks: list[EvidenceChunk],
    ) -> tuple[VerifierOutput, UsageDelta]:
        vers: list[VerifiedClaimLLM] = []
        for spec in self.case.claims:
            sid = f"S{spec.source_idx + 1}"
            ch = next((c for c in chunks if c.source_id == sid), None)
            if ch is None:
                continue
            ev_items = [
                VerifierEvidenceLLM(
                    source_id=sid,
                    chunk_id=ch.chunk_id,
                    quote=spec.quote,
                    role=spec.role,
                    quote_language=spec.quote_language,
                )
            ]
            if spec.second_source_idx is not None and spec.second_quote is not None:
                sid2 = f"S{spec.second_source_idx + 1}"
                ch2 = next((c for c in chunks if c.source_id == sid2), None)
                if ch2 is not None:
                    ev_items.append(
                        VerifierEvidenceLLM(
                            source_id=sid2,
                            chunk_id=ch2.chunk_id,
                            quote=spec.second_quote,
                            role=spec.role,
                            quote_language=spec.quote_language,
                        )
                    )
            vers.append(
                VerifiedClaimLLM(
                    claim_id=spec.claim_id,
                    verdict=spec.verdict,
                    rationale=spec.rationale,
                    causation_warning=spec.causation_warning,
                    evidence_items=ev_items,
                )
            )
        return VerifierOutput(verifications=vers), UsageDelta(model_calls=1)

    def synthesize_report(
        self,
        question: str,
        language: str,
        sub_questions: list[SubQuestion],
        claims: list[ClaimRecord],
        verifications: list[VerificationRecord],
        sources: list[SourceRecord],
    ) -> tuple[ReporterSynthesisOutput, UsageDelta]:
        final_map = {v.claim_id: v for v in verifications if v.is_final}
        claims_by_id = {c.claim_id: c for c in claims}
        updates: list[SubQuestionUpdateLLM] = []
        for sq in sub_questions:
            sq_vers = [
                v for cid, v in final_map.items()
                if cid in claims_by_id and claims_by_id[cid].sq_id == sq.sq_id
            ]
            if any(v.verdict == VerdictStatus.CONFLICTING for v in sq_vers):
                st = SubQuestionStatus.CONFLICTING
            elif any(v.verdict == VerdictStatus.SUPPORTED for v in sq_vers):
                st = SubQuestionStatus.ANSWERED
            else:
                st = SubQuestionStatus.UNANSWERED
            updates.append(SubQuestionUpdateLLM(sq_id=sq.sq_id, status=st))
        return (
            ReporterSynthesisOutput(
                title=f"{self.case.case_id} · {self.case.topic_label}",
                short_answer="",
                sub_question_updates=updates,
                open_questions=[],
            ),
            UsageDelta(model_calls=1, measurement_type="unmeasured"),
        )
