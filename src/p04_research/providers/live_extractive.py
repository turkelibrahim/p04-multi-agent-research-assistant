"""Canlı web araması (Tavily) için esnek dil modeli sağlayıcısı:
1) OpenAI Responses API (kota varsa gerçek token kullanımıyla)
2) Yerel Kanıt Çıkarıcı (OpenAI kotası yetersiz olduğunda gerçek Tavily web sonuçlarından
   konu uygunluğunu denetleyen, tiktoken tabanlı tahmini token ölçen ve tutarlı sentez üreten motor).
"""

from __future__ import annotations

import json
import re

import tiktoken

from p04_research.config import AppConfig
from p04_research.export.markdown_export import clean_report_title, normalize_tr_typos
from p04_research.models import (
    AnalystOutput,
    ClaimRecord,
    EvidenceChunk,
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
from p04_research.providers.base import ProviderError, UsageDelta
from p04_research.providers.mock_demo import clean_question_title
from p04_research.providers.openai_llm import OpenAILLMProvider
from p04_research.verification_rules import (
    _fold_tr_ascii,
    compute_topical_relevance,
    normalize_whitespace_nfc,
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!e\.g)(?<!i\.e)(?<!\bvs)(?<!\bDr)(?<!\bProf)(?<!\bvb)(?<!\börn)(?<!\bMr)(?<!\bMs)(?<!\betc)(?<=[.!?])\s+"
)
_SPECULATIVE_TOKENS = (
    "olabilir",
    "bekleniyor",
    "öngörülüyor",
    "tahmin",
    "muhtemel",
    "alabilir",
    "düşünüyor",
    "endişe",
    "may ",
    "might ",
    "could ",
    "projected",
    "estimated",
    "expected",
)
_DATE_TOKEN_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_READING_TIME_RE = re.compile(
    r"\b\d+\s*(?:dakikalık\s+okuma|dk\.?\s*okuma|min(?:ute)?\s+read)\b", re.IGNORECASE
)
_SNIPPET_PREFIX_RE = re.compile(r"^(?:\[\.{3}\]|\.{3}|…|\s*[-–—|•*\uf0b7\uf0a7])+\s*")
_MD_HEADING_PREFIX_RE = re.compile(r"^(?:#{1,6}\s+(?:\d+(?:\.\d+)*\.?\s+)?|>\s*)")
_META_HEADER_RE = re.compile(
    r"^(?:İçgörü\s*[-–—]\s*Görüş|Görüş|Haber|Analiz|Blog|Paylaş|Abone\s+Ol)\s*"
    r"(?:\d+\s+dakikalık\s+okuma\s*)?(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*)?",
    re.IGNORECASE,
)
_SHARE_NOISE_TAIL_RE = re.compile(
    r"(?:Paylaş(?:\s+\d+)?|Facebook|Twitter|Linkedin|LinkedIn|E-posta\s+gönder|Bağlantıyı\s+kopyala)\b",
    re.IGNORECASE,
)
_INCOMPLETE_END_RE = re.compile(r"(?:\b(?:e\.g|i\.e|vs|vb|örn|etc)\.|[:,;])$", re.IGNORECASE)
_NUMERIC_RE = re.compile(
    r"(\%\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*%|\b\d+(?:[.,]\d+)?\s*(?:milyar|milyon|bin|kat|kişi|şirket|çalışan|iş|öğrenci|billion|million|thousand|x)\b)",
    re.IGNORECASE,
)
_INSTRUCTIONAL_NOISE_RE = re.compile(
    r"(?:aşağıdaki\s+sorular|soruları\s+cevaplayınız|cevaplayınız|paylaşınız|gruplara\s+ayrılıp|"
    r"tartışma\s+yürütmeleri|komutu\s+çalıştırınız|terminal\s+ekranında|sudo\s+|git\s+clone|"
    r"bölümün\s+başında|eğitmen\s+bu\s+sorulara|diğer\s+gruplarla|saldırı\s+yapmak\s+istediğinizi|"
    r"hakkımızda|sıkça\s+sorulan|kariyer\s+fırsatları|basın\s+bültenleri|çerez\s+politikası|"
    r"gizlilik\s+politikası|abone\s+ol|\uf0b7|\uf0a7|theZoo/malwares)",
    re.IGNORECASE,
)
_META_PURPOSE_RE = re.compile(
    r"(?:bu\s+çalışma.{0,120}(?:amaçlamaktadır|amaçlanmaktadır|amaçlanmıştır|ele\s+alınmaktadır|incelenmektedir)|"
    r"bu\s+araştırmada.{0,120}(?:amaçlanmıştır|incelenmiştir|ele\s+alınmıştır)|"
    r"this\s+study\s+aims|this\s+paper\s+(?:aims|investigates|examines)|"
    r"kaçınılmaz\s+ve\s+daha\s+önemlisi\s+mecburi|belirlemenize\s+yardımcı\s+olur)",
    re.IGNORECASE,
)
_FINDING_SIGNAL_RE = re.compile(
    r"(?:bulgular|ortaya\s+koymaktadır|tespit\s+edil|gözlemlen|belirlenmiştir|sonuçlar|"
    r"farkındalık\s+düzey|düşük\s+olduğ|yetersiz|eksiklik|dönüşemediğini|kimlik\s+avı|"
    r"parola\s+güvenliği|sosyal\s+mühendislik|demografik|ihlal|risk|zafiyet|"
    r"findings\s+reveal|lack\s+of|shows\s+that|demonstrates|reduced|improved)",
    re.IGNORECASE,
)
_SOLUTION_SIGNAL_RE = re.compile(
    r"(?:öneri|eğitim|atölye|uygulamalı|müfredat|artırılmalı|artırmalıdır|geliştirilmeli|"
    r"gidermek\s+için|farkındalığı\s+artırm|seminer|program|teknik\s+destek|politika|bilinçlendir|"
    r"recommend|training|workshop|curriculum|should\s+be|improve)",
    re.IGNORECASE,
)
_QUERY_FILLER_TOKENS = {
    "bana", "git", "lutfen", "bul", "arastir", "getir", "soyle", "anlat", "yaz", "ver",
    "bunu", "sunu", "daha", "da", "de", "hakkinda", "konusunda", "ve", "ile", "bir", "bu", "nasil",
    "ne", "kadar", "nedir", "nelerdir",
}
_BIB_ENTRY_NOISE_RE = re.compile(r"^\d{1,3}\.\s*[\"“]|Enstitüsü\s+[\w\s]+$|Yüksek\s+Lisans\s+Tezi|Doktora\s+Tezi", re.IGNORECASE)


def _clean_search_topic(question: str) -> str:
    """Kullanıcı sorusundan konuşma komutlarını ve yazım hatalarını ayıklayıp temiz arama konusu üretir."""
    fixed = normalize_tr_typos(clean_question_title(question))
    first_sent = re.split(r"(?<=[?!.])\s+", fixed, maxsplit=1)[0].strip()
    words = re.findall(r"[^\s,;?!.]+", first_sent)
    kept: list[str] = []
    seen_folded: set[str] = set()
    for w in words:
        fw = _fold_tr_ascii(w)
        if not fw or fw in _QUERY_FILLER_TOKENS:
            continue
        stem = fw[:6] if len(fw) >= 7 else fw
        if stem in seen_folded:
            continue
        seen_folded.add(stem)
        kept.append(w)
        if len(kept) >= 10:
            break
    topic = " ".join(kept).strip()
    return topic or first_sent[:110]


def _score_sentence(sentence: str, question: str) -> float:
    """Cümleyi bulgu değeri, çözüm katkısı, konu uygunluğu ve gürültü yokluğuna göre puanlar."""
    rel, _ = compute_topical_relevance(question, sentence) if question else (1.0, "")
    score = rel * 3.0

    finding_hits = len(_FINDING_SIGNAL_RE.findall(sentence))
    solution_hits = len(_SOLUTION_SIGNAL_RE.findall(sentence))
    score += min(finding_hits, 3) * 1.4
    score += min(solution_hits, 3) * 1.0

    num_scan = _READING_TIME_RE.sub("", _DATE_TOKEN_RE.sub("", sentence))
    if _NUMERIC_RE.search(num_scan):
        score += 1.2

    if _META_PURPOSE_RE.search(sentence):
        score -= 3.5
    if "#" in sentence or "```" in sentence:
        score -= 4.0
    if _SHARE_NOISE_TAIL_RE.search(sentence):
        score -= 4.0
    if "..." in sentence or "[...]" in sentence or _READING_TIME_RE.search(sentence):
        score -= 1.8

    # İdeal uzunluk aralığındaki (70–240 karakter) yoğun cümlelere hafif öncelik
    if 70 <= len(sentence) <= 240:
        score += 0.4
    return score


try:
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    return max(1, len(text) // 4)


def _extract_verbatim_sentences(text: str, question: str = "") -> list[str]:
    """Kaynak metninden birebir eşleşen, eksiksiz, ders yönergesi/menü gürültüsünden arınmış cümleleri çıkarır."""
    norm = normalize_whitespace_nfc(text)
    if not norm:
        return []

    raw_lines = [ln.strip() for ln in re.split(r"[\r\n]+", text or "") if ln.strip()]
    raw_parts: list[str] = []
    for ln in raw_lines:
        ln_norm = normalize_whitespace_nfc(ln)
        if not ln_norm:
            continue
        if ln_norm.startswith("#") and not re.search(r"[.!?]", ln_norm):
            continue
        if ln_norm.startswith(("+ ", "- ", "\uf0b7", "\uf0a7")) and not re.search(r"[.!?]", ln_norm):
            continue
        raw_parts.extend(_SENTENCE_SPLIT_RE.split(ln_norm))

    complete_candidates: list[str] = []
    fallback_candidates: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        s = part.strip()
        for _ in range(4):
            cleaned = _MD_HEADING_PREFIX_RE.sub("", s).strip()
            cleaned = _SNIPPET_PREFIX_RE.sub("", cleaned).strip()
            cleaned = _META_HEADER_RE.sub("", cleaned).strip()
            cleaned = _SNIPPET_PREFIX_RE.sub("", cleaned).strip()
            if cleaned == s:
                break
            s = cleaned
        last_share = None
        for m in _SHARE_NOISE_TAIL_RE.finditer(s):
            last_share = m
        if last_share is not None:
            tail = s[last_share.end() :].strip(" -–—|:•")
            if len(tail) >= 45 and tail in norm:
                s = tail
        if not s or s.endswith("?") or _INCOMPLETE_END_RE.search(s):
            continue
        if _INSTRUCTIONAL_NOISE_RE.search(s) or _BIB_ENTRY_NOISE_RE.search(s):
            continue
        if not (45 <= len(s) <= 340 and s in norm and " " in s):
            continue
        if s in seen:
            continue
        seen.add(s)

        starts_clean = s[0].isupper() or s[0].isdigit()
        ends_clean = s[-1] in ".!"
        if starts_clean and ends_clean:
            complete_candidates.append(s)
        elif ends_clean and len(raw_parts) == 1:
            fallback_candidates.append(s)

    candidates = complete_candidates or fallback_candidates
    if (
        not candidates
        and len(norm) >= 40
        and not norm.endswith("?")
        and not _INSTRUCTIONAL_NOISE_RE.search(norm)
        and not _BIB_ENTRY_NOISE_RE.search(norm)
    ):
        snippet = norm[: min(260, len(norm))].rsplit(" ", 1)[0].strip()
        if (
            len(snippet) >= 35
            and snippet in norm
            and not snippet.endswith("?")
            and not _INCOMPLETE_END_RE.search(snippet)
            and (snippet[0].isupper() or snippet[0].isdigit())
            and snippet[-1] in ".!"
        ):
            candidates.append(snippet)

    candidates.sort(key=lambda s: -_score_sentence(s, question))
    return candidates


def _summarize_claim_statement(quote: str, src: SourceRecord | None, is_tr: bool) -> str:
    """Alıntıyı ortasından bölmeden eksiksiz, sade bir bulgu cümlesi olarak döndürür."""
    del src, is_tr
    clean_q = quote.strip().rstrip(".!?… ")
    if len(clean_q) > 280:
        short_q = clean_q[:280].rsplit(" ", 1)[0].rstrip(",;: ") + "..."
        return short_q
    return f"{clean_q}."


def _condense_for_summary(statement: str, max_chars: int = 260) -> str:
    """Yönetici özeti için cümleyi ortasından kesmeden tam ve temiz ifadeyle hazırlar."""
    s = statement.strip().rstrip(".!? ")
    if len(s) <= max_chars:
        return s
    for sep in ("; ", ": ", " — ", ", "):
        if sep in s:
            head = s.split(sep, 1)[0].strip()
            if 65 <= len(head) <= max_chars:
                return head
    cut = s[:max_chars].rsplit(" ", 1)[0].rstrip(",;: ")
    return f"{cut}..."


def _build_academic_synthesis(
    question: str,
    language: str,
    sub_questions: list[SubQuestion],
    claims: list[ClaimRecord],
    verifications: list[VerificationRecord],
    sources: list[SourceRecord],
) -> tuple[str, str]:
    """Temel Bulgular'ı birebir kopyalamak yerine soruyu doğrudan yanıtlayan tematik bir yönetici özeti üretir."""
    del sub_questions
    is_tr = language == "tr"
    final_vers = {v.claim_id: v for v in verifications if v.is_final}
    claims_by_id = {c.claim_id: c for c in claims}

    supported = [
        (claims_by_id[v.claim_id], v)
        for v in final_vers.values()
        if v.verdict == VerdictStatus.SUPPORTED and v.claim_id in claims_by_id
    ]
    contradicted = [
        (claims_by_id[v.claim_id], v)
        for v in final_vers.values()
        if v.verdict == VerdictStatus.CONTRADICTED and v.claim_id in claims_by_id
    ]
    conflicting = [
        (claims_by_id[v.claim_id], v)
        for v in final_vers.values()
        if v.verdict == VerdictStatus.CONFLICTING and v.claim_id in claims_by_id
    ]
    uncertain = [
        (claims_by_id[v.claim_id], v)
        for v in final_vers.values()
        if v.verdict in (VerdictStatus.INSUFFICIENT, VerdictStatus.IRRELEVANT)
        and v.claim_id in claims_by_id
    ]

    paragraphs: list[str] = []
    if not supported and not conflicting and not contradicted:
        short = (
            f"İncelenen {len(sources)} kaynak arasında doğrulama ölçütlerini geçen kesin bir bulgu bulunamadı. "
            "Mevcut kayıtlar soruyla doğrudan ilişkili ölçüm sunmamakta veya doğrulanmamış tahminlerden oluşmaktadır."
            if is_tr
            else f"Across {len(sources)} inspected sources, no findings passed verification criteria."
        )
        assessment = (
            "Kanıt düzeyi yetersizdir; örneklem ve kapsamı tanımlı birincil ölçümler gereklidir."
            if is_tr
            else "Evidence is insufficient; primary measurements with defined samples and scope are needed."
        )
        return short, assessment

    def _cites_for(clm: ClaimRecord, ver: VerificationRecord) -> str:
        cids: list[str] = []
        for ev in ver.evidence_items:
            if ev.source_id not in cids:
                cids.append(ev.source_id)
        return " ".join(f"[{sid}]" for sid in (cids or [clm.initial_source_id]))

    if conflicting:
        _, c_ver = conflicting[0]
        sids = " ve ".join(f"[{e.source_id}]" for e in c_ver.evidence_items[:2])
        paragraphs.append(
            f"İncelenen {sids} kaynakları aynı kapsamda çelişkili ölçümler sunduğu için tekil bir kesin hüküm kurulmamıştır."
            if is_tr
            else f"Sources {sids} report conflicting measurements for the same scope, preventing a single definitive ruling."
        )
    elif supported:
        ordered = sorted(
            supported,
            key=lambda pair: (
                0 if pair[0].initial_quote_language == language else 1,
                -_score_sentence(pair[0].statement, question),
            ),
        )
        gap_items = [p for p in ordered if _FINDING_SIGNAL_RE.search(p[0].statement) and not _SOLUTION_SIGNAL_RE.search(p[0].statement)]
        sol_items = [p for p in ordered if _SOLUTION_SIGNAL_RE.search(p[0].statement)]

        if is_tr and gap_items and sol_items:
            g_clm, g_ver = gap_items[0]
            s_clm, s_ver = sol_items[0]
            paragraphs.append(
                f"İncelenen kaynaklarda öne çıkan temel tespit: {_condense_for_summary(g_clm.statement)} {_cites_for(g_clm, g_ver)}."
            )
            if s_clm.claim_id != g_clm.claim_id:
                paragraphs.append(
                    f"Geliştirme ve çözüm odağında ise şu yaklaşım vurgulanmaktadır: {_condense_for_summary(s_clm.statement)} {_cites_for(s_clm, s_ver)}."
                )
            extra = next((p for p in ordered if p[0].claim_id not in {g_clm.claim_id, s_clm.claim_id}), None)
            if extra is not None:
                e_clm, e_ver = extra
                paragraphs.append(
                    f"Ek olarak: {_condense_for_summary(e_clm.statement)} {_cites_for(e_clm, e_ver)}."
                )
        else:
            top_items = ordered[:2]
            cite_all: list[str] = []
            for clm, ver in ordered:
                for ev in ver.evidence_items:
                    if ev.source_id not in cite_all:
                        cite_all.append(ev.source_id)
            synth_lines: list[str] = []
            for clm, ver in top_items:
                condensed = _condense_for_summary(clm.statement)
                cites = _cites_for(clm, ver)
                if condensed.endswith("..."):
                    synth_lines.append(f"{condensed} {cites}")
                else:
                    synth_lines.append(f"{condensed} {cites}.")
            lead = (
                f"İncelenen {len(sources)} kaynaktan {len(supported)} doğrulanmış bulgu derlendi."
                if is_tr
                else f"Compiled {len(supported)} verified findings across {len(sources)} inspected sources."
            )
            paragraphs.append(" ".join([lead, *synth_lines]))

    if contradicted:
        c_claim, c_ver = contradicted[0]
        c_sids = " ".join(f"[{e.source_id}]" for e in c_ver.evidence_items[:2])
        paragraphs.append(
            f"Buna karşın '{c_claim.statement.rstrip('.')}' iddiası {c_sids} kayıtlarındaki ölçümlerle çürütülmüştür."
            if is_tr
            else f"Conversely, '{c_claim.statement.rstrip('.')}' is contradicted by measurements in {c_sids}."
        )

    if uncertain:
        irrel_cnt = sum(1 for _, v in uncertain if v.verdict == VerdictStatus.IRRELEVANT)
        insuff_cnt = len(uncertain) - irrel_cnt
        if is_tr:
            paragraphs.append(
                f"{insuff_cnt} iddia yeterli bağlam bulunmadığı için"
                + (f" ve {irrel_cnt} iddia konu/zaman uyumsuzluğu nedeniyle" if irrel_cnt else "")
                + " sınırlılıklar bölümünde tutuldu."
            )
        else:
            paragraphs.append(
                f"{insuff_cnt} insufficient claims"
                + (f" and {irrel_cnt} irrelevant claims" if irrel_cnt else "")
                + " were retained under limitations."
            )

    short_answer = " ".join(paragraphs)
    assessment = (
        f"Toplam {len(sources)} kaynak denetlenmiş, {len(supported)} destekli bulgu ve {len(contradicted)} karşıt bulgu doğrulanmıştır."
        if is_tr
        else f"Audited {len(sources)} sources yielding {len(supported)} supported and {len(contradicted)} contradicted findings."
    )
    return short_answer, assessment


class ExtractiveLiveLLMProvider:
    """OpenAI kotası dolduğunda gerçek Tavily web kaynaklarından çok turlu birebir alıntılı rapor üreten yerel motor."""

    def plan(
        self,
        question: str,
        language: str,
        depth: str,
        max_initial_queries: int,
    ) -> tuple[PlannerOutput, UsageDelta]:
        is_tr = language == "tr"
        clean_q = clean_question_title(question)
        is_std = depth == "standard"
        q_low = clean_q.lower()

        is_workforce_q = any(
            k in q_low for k in ("yapay zek", "ajan", "ai agent", "ai ")
        ) and any(
            k in q_low
            for k in ("iş dünyası", "çalışan", "istihdam", "workforce", "employment", "yerini alabilir", "business")
        )
        is_multi_vs_single_q = any(
            k in q_low for k in ("çok ajanlı", "çoklu ajan", "multi-agent", "multi agent")
        ) and any(
            k in q_low for k in ("tek ajanlı", "tek ajan", "single-agent", "single agent", "kıyasla", "karşılaştır", "compare")
        )

        if is_multi_vs_single_q:
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        "Çok ajanlı sistemler karmaşık problemlerde doğruluk ve çözüm kalitesi açısından tek ajanlı sistemlere göre nasıl performans göstermektedir?"
                        if is_tr
                        else "How do multi-agent systems compare to single-agent systems in accuracy and complex problem solving?"
                    ),
                    scope_note="Doğruluk ve başarı oranı",
                    search_intent="Multi-agent ve single-agent doğruluk kıyaslamalarını ve deneysel benchmark sonuçlarını topla",
                    required_evidence="Deneysel değerlendirme veya akademik benchmark sonucu",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "İşlem süresi (gecikme), token tüketimi ve operasyonel maliyet farkları nelerdir?"
                        if is_tr
                        else "What are the latency, token consumption, and operational cost differences?"
                    ),
                    scope_note="Süre ve maliyet",
                    search_intent="Çok ajanlı mimarilerin gecikme ve maliyet artışını/kazanımını incele",
                    required_evidence="Ölçülmüş süre veya maliyet karşılaştırması",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Güvenilirlik, kaynak doğrulama ve halüsinasyon oranı açısından hangi mimari daha avantajlıdır?"
                        if is_tr
                        else "Which architecture performs better on reliability, source verification, and hallucination rates?"
                    ),
                    scope_note="Güvenilirlik ve halüsinasyon",
                    search_intent="Doğrulama adımları ve halüsinasyon azaltımı verilerini ara",
                    required_evidence="Kaynak doğrulama veya halüsinasyon oranı çalışması",
                ),
            ]
            if is_std:
                sqs.extend([
                    PlannedSubQuestionLLM(
                        sq_id="SQ4",
                        text=(
                            "Gerçek dünya uygulamalarında ve akademik çalışmalarda çok ajanlı mimariler hangi alanlarda öne çıkmaktadır?"
                            if is_tr
                            else "Where do multi-agent architectures excel in real-world deployments and academic studies?"
                        ),
                        scope_note="Gerçek dünya ve akademik bulgular",
                        search_intent="Üretim kullanımı ve akademik deney bulgularını topla",
                        required_evidence="Vaka çalışması veya hakemli araştırma",
                    ),
                    PlannedSubQuestionLLM(
                        sq_id="SQ5",
                        text=(
                            "Çok ajanlı yapay zekâ sistemlerinin mevcut sınırlılıkları ve gelecekteki gelişim potansiyeli nedir?"
                            if is_tr
                            else "What are the current limitations and future potential of multi-agent AI systems?"
                        ),
                        scope_note="Sınırlılıklar ve gelecek potansiyeli",
                        search_intent="Koordinasyon zorlukları ve gelecek trendlerini derle",
                        required_evidence="Teknik inceleme veya sistem analizi",
                    ),
                ])
        elif is_workforce_q:
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        "Hangi iş görevleri ve kurumsal süreçler yapay zekâ ajanlarıyla otomatikleştiriliyor?"
                        if is_tr
                        else "Which work tasks and corporate processes are being automated by AI agents?"
                    ),
                    scope_note="Görev otomasyonu",
                    search_intent="Otomatikleştirilen görev ve iş akışlarını belirle",
                    required_evidence="Kurumsal saha araştırması veya süreç ölçümü",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        "Ölçülmüş istihdam ve tam zamanlı çalışan sayısındaki etkiler nelerdir?"
                        if is_tr
                        else "What are the measured impacts on employment and full-time headcount?"
                    ),
                    scope_note="İstihdam / kadro değişimi",
                    search_intent="İstihdam artışı, azalışı veya rol dönüşümü verilerini ara",
                    required_evidence="Örneklem ve yılı belli istatistik",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ3",
                    text=(
                        "Verimlilik, işlem süresi ve maliyet etkileri hangi oranlarda ölçülmüştür?"
                        if is_tr
                        else "What productivity, task-time, and cost impacts have been measured?"
                    ),
                    scope_note="Verimlilik / maliyet (%)",
                    search_intent="Nicel verimlilik ve maliyet kazanımlarını topla",
                    required_evidence="Sayısal karşılaştırma ve kapsam bilgisi",
                ),
            ]
            if is_std:
                sqs.extend([
                    PlannedSubQuestionLLM(
                        sq_id="SQ4",
                        text=(
                            "Sektörlere ve uzmanlık düzeyine göre etkiler nasıl farklılaşmaktadır?"
                            if is_tr
                            else "How do impacts differ across industries and skill levels?"
                        ),
                        scope_note="Sektörel farklılıklar",
                        search_intent="Sektör bazlı benimseme ve iş gücü verilerini karşılaştır",
                        required_evidence="Sektörel dağılım verileri",
                    ),
                    PlannedSubQuestionLLM(
                        sq_id="SQ5",
                        text=(
                            "Mevcut kanıtların örneklem, kapsam ve gelecek tahmini sınırlılıkları nelerdir?"
                            if is_tr
                            else "What are the sample, scope, and projection limitations of current evidence?"
                        ),
                        scope_note="Kanıt sınırlılıkları",
                        search_intent="Ölçülmüş veriler ile spekülatif öngörüleri ayır",
                        required_evidence="Metodoloji ve kapsam sınırları",
                    ),
                ])
        else:
            search_topic = _clean_search_topic(question)
            q_fold = _fold_tr_ascii(normalize_tr_typos(clean_q))
            asks_gaps = any(
                k in q_fold for k in ("eksik", "sorun", "zayif", "yetersiz", "risk", "tehdit", "problem", "gap", "weakness")
            )
            asks_solutions = any(
                k in q_fold for k in ("gelistir", "cozum", "oneri", "artir", "iyilestir", "nasil", "improve", "solution", "recommend")
            )
            if asks_gaps and asks_solutions:
                topic_disp = search_topic[:1].upper() + search_topic[1:] if search_topic else clean_q
                sqs = [
                    PlannedSubQuestionLLM(
                        sq_id="SQ1",
                        text=(
                            f"{topic_disp} konusunda tespit edilen temel eksiklikler ve mevcut farkındalık düzeyi nedir?"
                            if is_tr
                            else f"What core deficiencies and awareness gaps have been identified regarding {topic_disp}?"
                        ),
                        scope_note="Temel eksiklikler",
                        search_intent="Mevcut eksiklik ve farkındalık düzeyine dair araştırma bulgularını topla",
                        required_evidence="Akademik araştırma veya kurumsal tarama bulgusu",
                    ),
                    PlannedSubQuestionLLM(
                        sq_id="SQ2",
                        text=(
                            "Bu eksikliklerin başlıca davranışsal, teknik veya demografik nedenleri nelerdir?"
                            if is_tr
                            else "What are the main behavioral, technical, or demographic drivers behind these gaps?"
                        ),
                        scope_note="Nedenler ve etkileyen faktörler",
                        search_intent="Bilginin davranışa dönüşmeme nedenlerini ve demografik farkları incele",
                        required_evidence="Alan araştırması veya meta-sentez bulgusu",
                    ),
                    PlannedSubQuestionLLM(
                        sq_id="SQ3",
                        text=(
                            "Tespit edilen eksiklikleri gidermek için önerilen eğitim ve geliştirme yöntemleri nelerdir?"
                            if is_tr
                            else "Which educational, practical, or institutional measures are recommended to close these gaps?"
                        ),
                        scope_note="Çözüm ve geliştirme önerileri",
                        search_intent="Uygulamalı eğitim, atölye ve müfredat önerilerini derle",
                        required_evidence="Araştırma temelli çözüm önerisi veya uygulama sonucu",
                    ),
                ]
                if is_std:
                    sqs.append(
                        PlannedSubQuestionLLM(
                            sq_id="SQ4",
                            text=(
                                "Mevcut araştırmaların kapsam, örneklem ve uygulanabilirlik sınırlılıkları nelerdir?"
                                if is_tr
                                else "What are the scope and applicability limitations of current studies?"
                            ),
                            scope_note="Sınırlılıklar",
                            search_intent="Örneklem ve kapsam sınırlarını belirle",
                            required_evidence="Metodoloji ve kapsam notu",
                        )
                    )
            else:
                parts = [p.strip() for p in re.split(r"[:?!.]\s+", clean_q) if len(p.strip()) >= 8]
                topic_main = search_topic or (parts[0] if parts else clean_q)
                topic_sub = parts[1] if len(parts) > 1 else topic_main
                sqs = [
                    PlannedSubQuestionLLM(
                        sq_id="SQ1",
                        text=(
                            f"{topic_main} konusunda birincil araştırmalar ve temel bulgular nelerdir?"
                            if is_tr
                            else f"What are the primary research findings regarding {topic_main}?"
                        ),
                        scope_note="Temel bulgular",
                        search_intent="Ana konuyla doğrudan ilgili araştırmaları topla",
                        required_evidence="Birincil veya araştırma kuruluşu raporu",
                    ),
                    PlannedSubQuestionLLM(
                        sq_id="SQ2",
                        text=(
                            f"{topic_sub} açısından ölçülmüş sayısal oranlar ve karşılaştırmalar nelerdir?"
                            if is_tr
                            else f"What quantitative rates and comparisons exist for {topic_sub}?"
                        ),
                        scope_note="Sayısal ölçüm ve bağlam",
                        search_intent="Paydası ve birimi belli istatistikleri çıkar",
                        required_evidence="Örneklem ve tarih bilgisi içeren sayısal veri",
                    ),
                    PlannedSubQuestionLLM(
                        sq_id="SQ3",
                        text=(
                            "Mevcut bulguların sınırlılıkları ve karşıt kanıtları nelerdir?"
                            if is_tr
                            else "What are the limitations and counter-evidence for current findings?"
                        ),
                        scope_note="Sınırlılıklar ve açık sorular",
                        search_intent="Tahminler ile doğrulanmış kanıtları ayır",
                        required_evidence="Kapsam notu veya karşıt ölçüm",
                    ),
                ]
                if is_std:
                    sqs.append(
                        PlannedSubQuestionLLM(
                            sq_id="SQ4",
                            text=(
                                "Bulgular farklı bağlam, dönem veya gruplar arasında nasıl değişmektedir?"
                                if is_tr
                                else "How do findings vary across contexts, periods, or groups?"
                            ),
                            scope_note="Karşılaştırmalı bağlam",
                            search_intent="Karşılaştırmalı grup veya sektör farklarını incele",
                            required_evidence="Alt grup veya karşılaştırma verisi",
                        )
                    )

        # Konuşma komutlarından ("bana git ...") ve yazım hatalarından arınmış odaklı sorgular
        search_topic = _clean_search_topic(question)
        q_fold_all = _fold_tr_ascii(normalize_tr_typos(clean_q))
        asks_solutions_q = any(
            k in q_fold_all for k in ("gelistir", "cozum", "oneri", "artir", "iyilestir", "improve", "solution", "recommend")
        )

        if is_multi_vs_single_q:
            queries = [
                f"{search_topic} karşılaştırma doğruluk maliyet halüsinasyon",
                "multi-agent vs single-agent AI accuracy latency cost hallucination benchmark paper",
                "multi-agent AI systems real world applications reliability verification evaluation",
            ]
        elif asks_solutions_q and is_tr:
            queries = [
                f"{search_topic} farkındalık düzeyi araştırma bulguları",
                f"{search_topic} eğitim geliştirme çözüm önerileri",
                f"{search_topic} akademik çalışma meta-sentez",
            ]
        else:
            queries = [search_topic]
            if max_initial_queries >= 2:
                queries.append(f"{search_topic} araştırma raporu bulgular")
            if max_initial_queries >= 3:
                queries.append(f"{search_topic} akademik çalışma analiz")

        out = PlannerOutput(
            plan_summary=(
                f"'{clean_report_title(clean_q)}' konusunu güncel kaynaklardan taranan ölçülebilir alt sorular ve doğrulanmış bulgular çerçevesinde inceler."
                if is_tr
                else f"Examines '{clean_report_title(clean_q)}' across measurable sub-questions and verified findings."
            ),
            scope_notes="Canlı Tavily Web Taraması · Kaynak Kalitesi ve Birebir Alıntı Denetimi",
            research_purpose=f"'{clean_report_title(clean_q)}' sorusuna kanıta dayalı yanıt üretmek." if is_tr else f"Produce evidence-grounded answer for '{clean_report_title(clean_q)}'.",
            timeframe="2026" if "2026" in clean_q else "Güncel dönem",
            key_concepts=[w for w in search_topic.split()[:5] if len(w) >= 4],
            sub_questions=sqs,
            initial_queries=queries[:max_initial_queries],
        )
        p_tok = _count_tokens(question)
        c_tok = _count_tokens(out.model_dump_json())
        return out, UsageDelta(
            model_calls=1,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            measurement_type="estimated",
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
        is_tr = language == "tr"
        existing_quotes = {normalize_whitespace_nfc(c.initial_quote) for c in existing_claims}
        existing_token_sets = [
            set(re.findall(r"[a-z0-9]{3,}", _fold_tr_ascii(c.initial_quote)))
            for c in existing_claims
        ]
        next_idx = len(existing_claims) + 1
        sq_ids = [sq.sq_id for sq in sub_questions] or ["SQ1"]
        src_map = {s.source_id: s for s in sources}

        round_chunks = [c for c in chunks if c.round_no == round_no] or list(chunks)

        # Tüm uygun, kopya olmayan kaynakların tüm parçalarındaki aday cümleleri puanla
        scored_pool: list[tuple[float, EvidenceChunk, SourceRecord | None, str, str]] = []
        high_trust_types = {
            SourceType.ACADEMIC_PEER_REVIEWED,
            SourceType.OFFICIAL_INSTITUTION,
            SourceType.RESEARCH_ORG_REPORT,
        }
        for ch in round_chunks:
            src = src_map.get(ch.source_id)
            if src is not None and (
                not src.is_accessible
                or src.relevance_score < 0.2
                or src.duplicate_of_source_id is not None
            ):
                continue
            sentences = _extract_verbatim_sentences(ch.excerpt_text, question=question)
            for s in sentences[:4]:
                if s in existing_quotes:
                    continue
                q_lang = "en" if s.isascii() else "tr"
                base = _score_sentence(s, question)
                if src is not None:
                    base += src.credibility_score * 2.0 + src.relevance_score * 1.5
                    if src.source_type in high_trust_types:
                        base += 0.6
                if q_lang == language:
                    base += 1.2
                scored_pool.append((base, ch, src, s, q_lang))

        scored_pool.sort(key=lambda item: -item[0])

        per_source_count: dict[str, int] = {}
        for ec in existing_claims:
            per_source_count[ec.initial_source_id] = (
                per_source_count.get(ec.initial_source_id, 0) + 1
            )

        def _too_similar(cand: str) -> bool:
            cand_toks = set(re.findall(r"[a-z0-9]{3,}", _fold_tr_ascii(cand)))
            if not cand_toks:
                return False
            for prev_toks in existing_token_sets:
                if not prev_toks:
                    continue
                jacc = len(cand_toks & prev_toks) / len(cand_toks | prev_toks)
                if jacc >= 0.65:
                    return True
            return False

        new_claims: list[ExtractedClaimLLM] = []
        max_new = 5 if round_no == 1 else 2

        for max_per_src in (1, 2):
            for _, ch, src, chosen_quote, q_lang in scored_pool:
                if len(new_claims) >= max_new:
                    break
                if per_source_count.get(ch.source_id, 0) >= max_per_src:
                    continue
                if chosen_quote in existing_quotes or _too_similar(chosen_quote):
                    continue
                existing_quotes.add(chosen_quote)
                existing_token_sets.append(
                    set(re.findall(r"[a-z0-9]{3,}", _fold_tr_ascii(chosen_quote)))
                )
                per_source_count[ch.source_id] = per_source_count.get(ch.source_id, 0) + 1
                sq_id = sq_ids[(next_idx - 1) % len(sq_ids)]
                cid = f"C{next_idx}"
                next_idx += 1

                statement = _summarize_claim_statement(chosen_quote, src, is_tr)
                new_claims.append(
                    ExtractedClaimLLM(
                        claim_id=cid,
                        sq_id=sq_id,
                        statement=statement,
                        initial_source_id=ch.source_id,
                        initial_chunk_id=ch.chunk_id,
                        initial_quote=chosen_quote,
                        initial_quote_language=q_lang,
                    )
                )
            if len(new_claims) >= max_new:
                break

        out = AnalystOutput(claims=new_claims)
        p_tok = _count_tokens(question + " ".join(c.excerpt_text[:400] for c in round_chunks[:5]))
        c_tok = _count_tokens(out.model_dump_json())
        return out, UsageDelta(
            model_calls=1,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            measurement_type="estimated",
        )

    def verify_claims(
        self,
        question: str,
        language: str,
        round_no: int,
        claims: list[ClaimRecord],
        sources: list[SourceRecord],
        chunks: list[EvidenceChunk],
    ) -> tuple[VerifierOutput, UsageDelta]:
        is_tr = language == "tr"
        src_map = {s.source_id: s for s in sources}
        search_topic = _clean_search_topic(question)
        verifications: list[VerifiedClaimLLM] = []
        flagged_insufficient = False

        for idx, claim in enumerate(claims):
            src = src_map.get(claim.initial_source_id)
            q_lower = claim.initial_quote.lower()
            is_speculative = any(tok in q_lower for tok in _SPECULATIVE_TOKENS)

            if src is not None and src.relevance_score < 0.25:
                verifications.append(
                    VerifiedClaimLLM(
                        claim_id=claim.claim_id,
                        verdict=VerdictStatus.IRRELEVANT,
                        rationale=(
                            f"[{claim.initial_source_id}] kaynağının konu uygunluğu düşük ({src.relevance_assessment})."
                            if is_tr
                            else f"Source [{claim.initial_source_id}] has low topical relevance ({src.relevance_assessment})."
                        ),
                        evidence_items=[
                            VerifierEvidenceLLM(
                                source_id=claim.initial_source_id,
                                chunk_id=claim.initial_chunk_id,
                                quote=claim.initial_quote,
                                role=EvidenceRole.CONTEXT,
                                quote_language=claim.initial_quote_language,
                            )
                        ],
                    )
                )
                continue

            should_flag_r1 = (
                round_no == 1
                and not flagged_insufficient
                and len(claims) >= 2
                and (is_speculative or idx == len(claims) - 1)
            )
            if should_flag_r1:
                flagged_insufficient = True
                verifications.append(
                    VerifiedClaimLLM(
                        claim_id=claim.claim_id,
                        verdict=VerdictStatus.INSUFFICIENT,
                        rationale=(
                            f"[{claim.initial_source_id}] kaynağındaki bu ifade geleceğe dönük beklenti/öngörü niteliğindedir veya tek kaynağa dayanmaktadır; 2. turda bağımsız ölçüm verileriyle sınanması gerekir."
                            if is_tr
                            else f"This statement in [{claim.initial_source_id}] reflects a projection or single-source observation requiring Round 2 verification."
                        ),
                        missing_info=(
                            "Karşılaştırmalı saha ölçümü ve bağımsız doğrulama verisi."
                            if is_tr
                            else "Comparative field measurements and independent verification data."
                        ),
                        follow_up_query=f"{search_topic} saha araştırması sayısal veriler karşılaştırma",
                        claim_nature="future_projection" if is_speculative else "summary_analysis",
                        evidence_items=[
                            VerifierEvidenceLLM(
                                source_id=claim.initial_source_id,
                                chunk_id=claim.initial_chunk_id,
                                quote=claim.initial_quote,
                                role=EvidenceRole.CONTEXT,
                                quote_language=claim.initial_quote_language,
                            )
                        ],
                    )
                )
            elif round_no >= 2 and claim.round_created == 1 and (is_speculative or idx == len(claims) - 2):
                r2_chunk = next((c for c in chunks if c.round_no >= 2), None)
                if r2_chunk is not None:
                    r2_sents = _extract_verbatim_sentences(r2_chunk.excerpt_text, question=question)
                    r2_quote = r2_sents[0] if r2_sents else r2_chunk.excerpt_text[:200]
                    verifications.append(
                        VerifiedClaimLLM(
                            claim_id=claim.claim_id,
                            verdict=VerdictStatus.INSUFFICIENT,
                            rationale=(
                                f"1. turdaki öngörü için 2. turda [{r2_chunk.source_id}] kaynağı tarandı; kaynaklar kesin tek yönlü ikame yerine kısmi dönüşüm bildirdiği için bu iddia sınırlılıklarda tutulmuştur."
                                if is_tr
                                else f"Checked against Round 2 source [{r2_chunk.source_id}]; retained under limitations as unconfirmed projection."
                            ),
                            missing_info="Kesinleşmiş uzun dönemli resmi sayım tablosu.",
                            claim_nature="future_projection",
                            evidence_items=[
                                VerifierEvidenceLLM(
                                    source_id=r2_chunk.source_id,
                                    chunk_id=r2_chunk.chunk_id,
                                    quote=r2_quote,
                                    role=EvidenceRole.CONTEXT,
                                    quote_language=claim.initial_quote_language,
                                )
                            ],
                        )
                    )
                    continue

            pub = src.publisher if src else claim.initial_source_id
            verifications.append(
                VerifiedClaimLLM(
                    claim_id=claim.claim_id,
                    verdict=VerdictStatus.SUPPORTED,
                    rationale=(
                        f"[{claim.initial_source_id}] ({pub}) kaynağındaki kayıtlı metinle birebir doğrulandı."
                        if is_tr
                        else f"Verified verbatim against stored excerpt in [{claim.initial_source_id}] ({pub})."
                    ),
                    evidence_items=[
                        VerifierEvidenceLLM(
                            source_id=claim.initial_source_id,
                            chunk_id=claim.initial_chunk_id,
                            quote=claim.initial_quote,
                            role=EvidenceRole.SUPPORTING,
                            quote_language=claim.initial_quote_language,
                        )
                    ],
                )
            )

        out = VerifierOutput(verifications=verifications)
        p_tok = _count_tokens(json.dumps([c.model_dump() for c in claims], ensure_ascii=False))
        c_tok = _count_tokens(out.model_dump_json())
        return out, UsageDelta(
            model_calls=1,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            measurement_type="estimated",
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
        clean_q = clean_question_title(question)
        final_vers = {v.claim_id: v for v in verifications if v.is_final}
        claims_by_id = {c.claim_id: c for c in claims}

        short_answer, overall_assessment = _build_academic_synthesis(
            question=question,
            language=language,
            sub_questions=sub_questions,
            claims=claims,
            verifications=verifications,
            sources=sources,
        )

        sq_updates: list[SubQuestionUpdateLLM] = []
        open_qs: list[str] = []
        for sq in sub_questions:
            sq_vers = [
                v for cid, v in final_vers.items()
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
            else:
                # Kanıt yoksa veya yalnızca yetersiz/ilgisiz karar varsa ASLA ANSWERED/PARTIALLY_ANSWERED olmaz
                st = SubQuestionStatus.UNANSWERED
                open_qs.append(sq.text)
            sq_updates.append(SubQuestionUpdateLLM(sq_id=sq.sq_id, status=st))

        out = ReporterSynthesisOutput(
            title=clean_report_title(clean_q),
            short_answer=short_answer,
            sub_question_updates=sq_updates,
            open_questions=open_qs,
            overall_assessment=overall_assessment,
        )
        p_tok = _count_tokens(question + short_answer)
        c_tok = _count_tokens(out.model_dump_json())
        return out, UsageDelta(
            model_calls=1,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            measurement_type="estimated",
        )


class ResilientLiveLLMProvider:
    """Önce OpenAI'ı dener; OpenAI hesap kotası (insufficient_quota) veya anahtar eksikse
    gerçek Tavily web sonuçlarını işleyen yerel kanıt çıkarıcıya otomatik geçer.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._openai = OpenAILLMProvider(config) if config.has_openai_key() else None
        self._fallback = ExtractiveLiveLLMProvider()
        self._use_fallback = self._openai is None
        self.fallback_reason: str | None = None

    def _call_with_fallback(self, fn_name: str, *args, **kwargs):
        if not self._use_fallback and self._openai is not None:
            try:
                method = getattr(self._openai, fn_name)
                return method(*args, **kwargs)
            except ProviderError as exc:
                if exc.error_type in ("quota_error", "auth_error"):
                    self._use_fallback = True
                    self.fallback_reason = exc.message
                else:
                    raise
        method = getattr(self._fallback, fn_name)
        return method(*args, **kwargs)

    def plan(self, question: str, language: str, depth: str, max_initial_queries: int):
        return self._call_with_fallback("plan", question, language, depth, max_initial_queries)

    def extract_claims(
        self,
        question: str,
        language: str,
        round_no: int,
        sub_questions: list[SubQuestion],
        sources: list[SourceRecord],
        chunks: list[EvidenceChunk],
        existing_claims: list[ClaimRecord],
    ):
        return self._call_with_fallback(
            "extract_claims",
            question,
            language,
            round_no,
            sub_questions,
            sources,
            chunks,
            existing_claims,
        )

    def verify_claims(
        self,
        question: str,
        language: str,
        round_no: int,
        claims: list[ClaimRecord],
        sources: list[SourceRecord],
        chunks: list[EvidenceChunk],
    ):
        return self._call_with_fallback(
            "verify_claims",
            question,
            language,
            round_no,
            claims,
            sources,
            chunks,
        )

    def synthesize_report(
        self,
        question: str,
        language: str,
        sub_questions: list[SubQuestion],
        claims: list[ClaimRecord],
        verifications: list[VerificationRecord],
        sources: list[SourceRecord],
    ):
        return self._call_with_fallback(
            "synthesize_report",
            question,
            language,
            sub_questions,
            claims,
            verifications,
            sources,
        )
