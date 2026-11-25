"""Programatik kanıt denetimi, kaynak kalite/ilgililik değerlendirmesi, URL/kopya ve regresyon kuralları."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from p04_research.models import (
    ClaimRecord,
    EvidenceChunk,
    EvidenceLevel,
    EvidenceRole,
    SourceRecord,
    SourceType,
    VerdictEvidenceItem,
    VerdictStatus,
    VerificationRecord,
    VerifiedClaimLLM,
)

_TRACKING_PARAM_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref")
_PUNCT_STRIP_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_PCT_RE = re.compile(r"(?:%\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*%)")
_PLAIN_NUM_RE = re.compile(r"\b\d+\b")

_INACCESSIBLE_MARKERS = (
    "403 forbidden",
    "404 not found",
    "access denied",
    "cloudflare",
    "enable javascript and cookies",
    "sayfa bulunamadı",
    "erişim engellendi",
    "robot check",
    "verify you are human",
    "paywall",
    "subscription required",
)

_STOPWORDS = {
    "ve", "ile", "bir", "bu", "su", "o", "mi", "mi", "mu", "ne", "nasil",
    "neden", "hangi", "kac", "gibi", "icin", "olarak", "daha", "cok", "en", "da", "de",
    "bana", "git", "bul", "arastir", "getir", "soyle", "anlat", "yaz", "ver", "bunu",
    "sunu", "hakkinda", "konusunda", "lutfen", "kadar", "uzerine", "ilgili", "olan",
    "olur", "yapar", "yapilir", "yuzden", "gore", "sonra", "once", "kendi",
    "the", "and", "for", "with", "that", "this", "what", "how", "why", "which", "are",
    "can", "will", "from", "into", "about", "yilinda", "etkisi", "nedir", "nelerdir",
}

_TR_ASCII_TABLE = str.maketrans(
    "çğıİöşüÇĞÖŞÜ",
    "cgiiosucgosu",
)
_BIDI_STRIP_RE = re.compile(r"[\u200b\u200e\u200f\u202a-\u202e\ufeff]")
_SENTENCE_SPLIT_TITLE_RE = re.compile(r"(?<=[?!.])\s+")
_CONVERSATIONAL_LEAD_RE = re.compile(
    r"^(?:(?:lütfen|lutfen)\s+)?(?:bana\s+)?(?:git\s+)?(?:bana\s+)?(?:şunu\s+|sunu\s+|bunu\s+)?",
    re.IGNORECASE,
)
_COMMON_TR_TYPO_MAP: tuple[tuple[str, str], ...] = (
    (r"\böğrencılerın\b|\bogrencilerin\b|\boğrencılerın\b|\böğrencılerı\b", "öğrencilerin"),
    (r"\böğrencıler\b|\bogrenciler\b", "öğrenciler"),
    (r"\bguvenlık\b|\bguvenlik\b|\bgüvenlık\b", "güvenlik"),
    (r"\beksıklıklerını\b|\beksikliklerini\b|\beksıklıklerı\b|\beksıklık\b", "eksiklikleri"),
    (r"\bgelışıtıırız\b|\bgelıstırırız\b|\bgelistiririz\b|\bgelıştırırız\b|\bgelıstırebılırız\b", "geliştirilebilir"),
    (r"\bfarkindalik\b|\bfarkındalık\b", "farkındalık"),
    (r"\barastirma\b|\baraştırma\b", "araştırma"),
    (r"\bhakkinda\b|\bhakkında\b", "hakkında"),
    (r"\bcozum\b", "çözüm"),
    (r"\boneri\b", "öneri"),
    (r"\begitim\b", "eğitim"),
)


def _fold_tr_ascii(text: str) -> str:
    """Türkçe karakterleri ve noktalı/noktasız 'ı/i' yazım farklarını eşleştirme için katlar."""
    nfc = unicodedata.normalize("NFC", text or "")
    return nfc.translate(_TR_ASCII_TABLE).lower()


def normalize_tr_typos(text: str) -> str:
    """Sık görülen Türkçe hızlı yazım ve noktalı/noktasız 'ı/i' hatalarını yumuşakça düzeltir."""
    out = _WHITESPACE_RE.sub(" ", _BIDI_STRIP_RE.sub("", text or "")).strip()
    for pat, repl in _COMMON_TR_TYPO_MAP:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return _WHITESPACE_RE.sub(" ", out).strip()


def clean_report_title(title: str) -> str:
    """Rapor başlığındaki gereksiz önekleri ve konuşma emirlerini temizler; çok cümleli uzun istemlerde ilk ana soruyu alır."""
    cleaned = _WHITESPACE_RE.sub(" ", _BIDI_STRIP_RE.sub("", title or "")).strip()
    for prefix in ("Araştırma Raporu:", "Research Report:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()

    stripped_lead = _CONVERSATIONAL_LEAD_RE.sub("", cleaned).strip()
    if stripped_lead and stripped_lead != cleaned and len(stripped_lead) >= 12:
        cleaned = normalize_tr_typos(stripped_lead)
        cleaned = re.sub(r"\bbul\s+ve\s+", "ve ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bbunu\s+daha\s+da\s+", "", cleaned, flags=re.IGNORECASE)
        words = cleaned.split()
        if len(words) >= 4 and words[-1].lower() in {w.lower() for w in words[:-1]}:
            words.pop()
            cleaned = " ".join(words)
        cleaned = cleaned[:1].upper() + cleaned[1:]

    if len(cleaned) > 120:
        parts = _SENTENCE_SPLIT_TITLE_RE.split(cleaned, maxsplit=1)
        if len(parts) > 1 and len(parts[0].strip()) >= 25:
            return parts[0].strip()
    return cleaned

_INTERROGATIVE_HINTS = (
    "kaç ", "nasıl ", "neden ", "hangi ", "kimler ", "ne kadar ",
    "olacak mı", "olabilir mi", "alabilirler mi", "how many ", "will there be",
)

_MEASUREMENT_SCOPE_HINTS = (
    "n=", "örneklem", "kohort", "cohort", "sample", "anket", "survey", "katılımcı",
    "respondent", "işlem", "gözlem", "turnike", "bordro", "tartım", "kayıtlı",
    "öğrenci", "şirket", "departman", "müşteri", "dakika", "saat", "puan", "birim",
    "çatı", "elektrik", "atık", "veri işleme", "kadro", "görev tanımı",
)

_GENERAL_WORKFORCE_HINTS = (
    "genel iş gücü", "genel istihdam", "tüm çalışanlar", "iş gücünün genel",
    "overall workforce", "general workforce", "total employment",
)


def normalize_whitespace_nfc(text: str) -> str:
    """Unicode NFC normalizasyonu uygular ve boşlukları tek boşluğa indirger."""
    normalized = unicodedata.normalize("NFC", text or "")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_url(url: str) -> str:
    """Aynı sayfaya işaret eden adresleri birleştirmek için URL'yi normalleştirir."""
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    kept_params = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not any(k.lower().startswith(prefix) for prefix in _TRACKING_PARAM_PREFIXES)
    ]
    kept_params.sort(key=lambda pair: (pair[0], pair[1]))
    query = urlencode(kept_params)
    return urlunsplit((scheme, netloc, path, query, ""))


def compute_content_fingerprint(text: str) -> str:
    """Kopya/aynı metinli haberleri tespit etmek için içerik parmak izi üretir."""
    nfc = unicodedata.normalize("NFC", text or "").lower()
    stripped = _PUNCT_STRIP_RE.sub(" ", nfc)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


def truncate_source_text(text: str, max_chars: int = 3500) -> tuple[str, bool]:
    """Kaynak metnini sınır karakterde keser ve kesilme durumunu döndürür."""
    cleaned = unicodedata.normalize("NFC", (text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned, False
    return cleaned[:max_chars], True


def quote_matches_source(quote: str, source_text: str) -> bool:
    """Alıntının kaynak metinde birebir (yalnızca boşluk/NFC toleranslı) geçip geçmediğini doğrular."""
    q_norm = normalize_whitespace_nfc(quote)
    s_norm = normalize_whitespace_nfc(source_text)
    if not q_norm or not s_norm:
        return False
    return q_norm in s_norm


def is_quote_cut_at_boundary(quote: str, chunk: EvidenceChunk) -> bool:
    """Alıntının kesilmiş parçanın son sınırına denk gelip yarım kaldığını sınar (Bölüm 8)."""
    if not chunk.is_truncated:
        return False
    q_norm = normalize_whitespace_nfc(quote)
    s_norm = normalize_whitespace_nfc(chunk.excerpt_text)
    if not q_norm or not s_norm:
        return True
    idx = s_norm.find(q_norm)
    if idx == -1:
        return True
    end_pos = idx + len(q_norm)
    return (len(s_norm) - end_pos) <= 15


def get_independent_source_ids(
    source_ids: Iterable[str],
    sources_by_id: dict[str, SourceRecord],
) -> list[str]:
    """Kopya kaynakları eleyerek bağımsız kaynak kimliklerini döndürür."""
    seen_roots: set[str] = set()
    seen_fingerprints: set[str] = set()
    independent: list[str] = []

    for sid in source_ids:
        src = sources_by_id.get(sid)
        if src is None:
            continue
        root_id = src.duplicate_of_source_id or src.source_id
        fp = src.content_fingerprint
        if root_id in seen_roots:
            continue
        if fp and fp in seen_fingerprints:
            continue
        seen_roots.add(root_id)
        if fp:
            seen_fingerprints.add(fp)
        independent.append(src.source_id)
    return independent


def is_page_accessible(content: str, url: str = "") -> bool:
    """Sayfa içeriğinin gerçekten okunabilir olup olmadığını kontrol eder."""
    norm = normalize_whitespace_nfc(content).lower()
    if len(norm) < 15:
        return False
    if any(marker in norm for marker in _INACCESSIBLE_MARKERS) and len(norm) < 260:
        return False
    return True


def classify_source_type(
    url: str,
    title: str,
    publisher: str,
    content: str,
    author: str | None = None,
    evidence_level: EvidenceLevel = EvidenceLevel.SECONDARY_REPORT,
) -> SourceType:
    """Kaynağı alan adı, yayıncı, yazar, yöntem ve içerik sinyallerine göre 7 sınıftan birine ayırır."""
    host = urlsplit(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    combined = f"{title} {publisher} {author or ''} {content[:400]}".lower()

    if any(d in host for d in ("youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "x.com", "twitter.com", "instagram.com", "reddit.com")) or any(
        k in combined for k in ("youtube", "video transkript", "video transcript", "sosyal medya")
    ):
        return SourceType.VIDEO_SOCIAL_MEDIA

    if any(host.endswith(suf) for suf in (".edu", ".edu.tr", ".ac.uk")) or any(
        d in host
        for d in (
            "arxiv.org",
            "scholar.google",
            "pubmed",
            "sciencedirect",
            "springer",
            "ieee",
            "aclanthology",
            "nature.com",
            "science.org",
            "dergipark.org.tr",
            "trdizin.gov.tr",
            "acikerisim.",
            "inased.org",
        )
    ) or any(k in combined for k in ("peer-reviewed", "hakemli", "doi:", "journal of", "randomized cohort", "dergisi", "meta-sentez")):
        return SourceType.ACADEMIC_PEER_REVIEWED

    if any(host.endswith(suf) for suf in (".gov", ".gov.tr", ".europa.eu")) or any(
        d in host for d in ("who.int", "oecd.org", "imf.org", "worldbank.org", "ilo.org", "tuik.gov.tr", "unesco.org")
    ) or any(k in combined for k in ("bakanlığı", "valiliği", "belediyesi", "müdürlüğü", "resmî istatistik")):
        return SourceType.OFFICIAL_INSTITUTION

    if any(
        k in combined
        for k in ("pwc", "mckinsey", "gartner", "deloitte", "forrester", "pew research", "brookings", "enstitüsü", "gözlemevi", "araştırma merkezi", "stanford", "mit")
    ):
        return SourceType.RESEARCH_ORG_REPORT

    if evidence_level == EvidenceLevel.PRIMARY_RECORD or any(
        k in combined for k in ("turnike", "bordro", "danışma kaydı", "tartım kaydı", "kurumsal veri", "n=")
    ):
        return SourceType.CORPORATE_PRIMARY_DATA

    if any(d in host for d in ("medium.com", "substack.com", "blogspot.com", "wordpress.com")) or any(
        k in combined for k in ("blog", "köşe yazısı", "yorum", "opinion", "haber portalı", "bülten")
    ):
        return SourceType.NEWS_BLOG_OPINION

    return SourceType.TRUSTED_SECONDARY_ANALYSIS


def _content_tokens(text: str) -> set[str]:
    folded = _fold_tr_ascii(text or "")
    words = _PUNCT_STRIP_RE.sub(" ", folded).split()
    tokens: set[str] = set()
    for w in words:
        if len(w) >= 3 and w not in _STOPWORDS:
            tokens.add(w[:5] if len(w) >= 6 else w)
    return tokens


def compute_topical_relevance(question: str, text: str, title: str = "") -> tuple[float, str]:
    """Soru ile kaynak/iddia arasındaki konu ve zaman uygunluğunu puanlar."""
    q_toks = _content_tokens(question)
    if not q_toks:
        return 1.0, "Konu eşleşmesi değerlendirildi"
    doc_toks = _content_tokens(f"{title} {text}")
    overlap = len(q_toks & doc_toks)
    ratio = min(1.0, overlap / max(1, min(len(q_toks), 4)))

    q_years = set(_YEAR_RE.findall(question))
    d_years = set(_YEAR_RE.findall(f"{title} {text}"))
    if q_years and d_years and not (q_years & d_years):
        # Farklı yıl belirtilmişse uygunluk puanını düşür
        ratio = max(0.15, ratio * 0.45)
        return round(ratio, 2), f"Zaman aralığı uyuşmazlığı (soru: {sorted(q_years)}, metin: {sorted(d_years)})"

    if ratio >= 0.5:
        return round(ratio, 2), "Yüksek konu uygunluğu"
    if ratio >= 0.25:
        return round(ratio, 2), "Kısmi konu uygunluğu"
    return round(ratio, 2), "Düşük konu uygunluğu"


def evaluate_source_quality_and_relevance(
    *,
    url: str,
    title: str,
    publisher: str,
    content: str,
    question: str,
    author: str | None = None,
    published_date: str | None = None,
    evidence_level: EvidenceLevel = EvidenceLevel.SECONDARY_REPORT,
    retrieval_method: str | None = None,
    source_type: SourceType | None = None,
    is_accessible: bool | None = None,
) -> dict:
    """Kaynak türü, erişilebilirlik, alıntı yöntemi, çok faktörlü güvenilirlik ve ilgililik üretir."""
    accessible = (True if is_accessible is None else bool(is_accessible)) and is_page_accessible(content, url)
    s_type = source_type or classify_source_type(
        url=url,
        title=title,
        publisher=publisher,
        content=content,
        author=author,
        evidence_level=evidence_level,
    )

    if not accessible:
        method = "inaccessible"
        eff_level = EvidenceLevel.SNIPPET_ONLY
    elif retrieval_method:
        method = retrieval_method
        eff_level = evidence_level
    elif evidence_level == EvidenceLevel.PRIMARY_RECORD:
        method = "primary_record"
        eff_level = evidence_level
    elif len(content.strip()) >= 1000:
        method = "full_text"
        eff_level = evidence_level
    else:
        method = "search_snippet"
        eff_level = (
            EvidenceLevel.SNIPPET_ONLY
            if evidence_level == EvidenceLevel.SECONDARY_REPORT and len(content.strip()) <= 300
            else evidence_level
        )

    base_scores = {
        SourceType.ACADEMIC_PEER_REVIEWED: 0.85,
        SourceType.OFFICIAL_INSTITUTION: 0.85,
        SourceType.RESEARCH_ORG_REPORT: 0.78,
        SourceType.CORPORATE_PRIMARY_DATA: 0.76,
        SourceType.TRUSTED_SECONDARY_ANALYSIS: 0.65,
        SourceType.NEWS_BLOG_OPINION: 0.45,
        SourceType.VIDEO_SOCIAL_MEDIA: 0.30,
    }
    cred = base_scores[s_type]
    if author:
        cred += 0.06
    if published_date:
        cred += 0.06
    c_low = content.lower()
    if any(k in c_low for k in ("n=", "örneklem", "kohort", "sample", "metodoloji", "yöntem", "turnike", "bordro")):
        cred += 0.08
    if method == "search_snippet":
        cred -= 0.08
    if not accessible:
        cred = 0.0
    cred = round(max(0.0, min(1.0, cred)), 2)

    rel_score, rel_note = compute_topical_relevance(question, content, title)
    if not accessible:
        rel_score = 0.0
        rel_note = "Sayfaya erişilemedi veya içerik okunamadı"

    return {
        "source_type": s_type,
        "retrieval_method": method,
        "evidence_level": eff_level,
        "credibility_score": cred,
        "relevance_score": rel_score,
        "relevance_assessment": rel_note,
        "is_accessible": accessible,
    }


def _is_question_or_distant_projection(statement: str, quote: str, research_question: str) -> str | None:
    """Soru cümlelerini veya uzak gelecek tahminlerini (C7 regresyonu) yakalar."""
    s_clean = normalize_whitespace_nfc(statement)
    q_clean = normalize_whitespace_nfc(quote)
    if s_clean.endswith("?") or q_clean.endswith("?"):
        return "Soru cümlesi ölçülmüş kanıt olarak kabul edilemez."
    lower_both = f"{s_clean} {q_clean}".lower()
    if any(h in lower_both for h in _INTERROGATIVE_HINTS):
        return "Soru veya varsayımsal sorgu ifadesi kanıt olarak kabul edilemez."

    if research_question:
        rq_years = {int(y) for y in _YEAR_RE.findall(research_question)}
        doc_years = {int(y) for y in _YEAR_RE.findall(f"{s_clean} {q_clean}")}
        if rq_years and max(rq_years) <= 2027 and any(y >= 2030 for y in doc_years):
            return f"Uzak gelecek projeksiyonu ({sorted(y for y in doc_years if y >= 2030)}), mevcut dönem için ölçülmüş kanıt değildir."
    return None


def _detect_temporal_or_topical_irrelevance(
    statement: str,
    quote: str,
    research_question: str,
) -> str | None:
    """Tarihsel arka planı (ör. 2022) veya tamamen ilgisiz metinleri (C5 regresyonu) eler."""
    if not research_question:
        return None
    rq_years = {int(y) for y in _YEAR_RE.findall(research_question)}
    if rq_years:
        target_min = min(rq_years)
        quote_years = {int(y) for y in _YEAR_RE.findall(quote)}
        if quote_years and all(y < target_min for y in quote_years):
            return (
                f"Alıntı yalnızca önceki dönem ({sorted(quote_years)}) arka planını anlatmaktadır; "
                f"hedef dönem ({sorted(rq_years)}) etkisi için doğrudan kanıt değildir."
            )

    rel_score, _ = compute_topical_relevance(research_question, f"{statement} {quote}")
    if rel_score < 0.15:
        return "Alıntı ve iddia araştırma sorusuyla doğrudan konu bütünlüğü taşımamaktadır."
    return None


def _detect_footnote_or_timestamp_number(statement: str, quote: str, chunk_text: str) -> str | None:
    """Dipnot numaralarının veya video zaman damgalarının istatistik sanılmasını (C3) engeller."""
    stmt_nums = [n for n in _PLAIN_NUM_RE.findall(statement) if not (1900 <= int(n) <= 2099)]
    if not stmt_nums:
        return None
    combined = f"{quote} {chunk_text}"
    for num in stmt_nums:
        footnote_pat = re.compile(
            rf"(\[{num}\]|\({num}\)|(?<!\d)\d{{1,2}}:{num}\b|\b{num}:\d{{2}}\b|dipnot\s*{num}\b|footnote\s*{num}\b|kaynak\s*{num}\b|[A-Za-zçğıöşüÇĞİÖŞÜ]{num}(?!\d))"
        )
        metric_pat = re.compile(
            rf"(%\s*{num}\b|\b{num}\s*%|n={num}\b|\b{num}\s+(?:kişi|şirket|öğrenci|dakika|saat|puan|birim|işlem|ziyaret|yüzde|percent|users|companies|minutes))",
            re.IGNORECASE,
        )
        if footnote_pat.search(combined) and not metric_pat.search(combined):
            return (
                f"Metindeki '{num}' değeri dipnot/atıf numarası veya zaman damgası niteliğindedir; "
                "istatistiksel ölçüm olarak yorumlanamaz."
            )
    return None


def _detect_undefined_percentage_generalization(
    statement: str,
    quote: str,
    chunk_text: str,
) -> str | None:
    """Paydası ve ölçülen grubu belirsiz yüzde oranlarının genel etki sayılmasını (C2) önler."""
    pcts = _PCT_RE.findall(f"{statement} {quote}")
    if not pcts:
        return None
    s_low = statement.lower()
    q_low = f"{quote} {chunk_text}".lower()
    has_scope = any(hint in q_low for hint in _MEASUREMENT_SCOPE_HINTS)
    broad_claim = any(hint in s_low for hint in _GENERAL_WORKFORCE_HINTS)

    if broad_claim and not has_scope:
        return (
            "Yüzde oranlarının hangi popülasyonu, payda veya ölçüm birimini kapsadığı kaynakta belirtilmediği için "
            "genel iş gücü etkisi olarak kabul edilemez."
        )
    if len(pcts) >= 2 and not has_scope and any(
        w in s_low for w in ("iş gücü", "istihdam", "etki", "workforce", "employment")
    ):
        return (
            "Birden fazla yüzde oranı verilmesine karşın örneklem, payda ve ölçülen grup tanımlanmamıştır."
        )
    return None


def enforce_verification_guardrails(
    claim: ClaimRecord,
    raw_verification: VerifiedClaimLLM,
    round_no: int,
    sources_by_id: dict[str, SourceRecord],
    chunks_by_id: dict[str, EvidenceChunk],
    language: str = "tr",
    research_question: str = "",
) -> VerificationRecord:
    """Kanıt denetçisi kararını yazılımsal olarak doğrular ve regresyon koruma kurallarını uygular."""
    guardrail_notes: list[str] = []
    valid_items: list[VerdictEvidenceItem] = []
    combined_chunks: list[str] = []

    for item in raw_verification.evidence_items:
        src = sources_by_id.get(item.source_id)
        chunk = chunks_by_id.get(item.chunk_id)
        if src is None:
            guardrail_notes.append(
                f"Bilinmeyen kaynak kimliği ({item.source_id}) reddedildi."
                if language == "tr"
                else f"Rejected unknown source ID ({item.source_id})."
            )
            continue
        if not src.is_accessible:
            guardrail_notes.append(
                f"Erişilemeyen kaynak ({item.source_id}) kanıt olarak reddedildi."
                if language == "tr"
                else f"Rejected inaccessible source ({item.source_id})."
            )
            continue
        if chunk is None or chunk.source_id != src.source_id:
            guardrail_notes.append(
                f"Geçersiz kanıt parçası ({item.chunk_id}) veya kaynak uyuşmazlığı reddedildi."
                if language == "tr"
                else f"Rejected invalid chunk ID ({item.chunk_id}) or source mismatch."
            )
            continue
        if not quote_matches_source(item.quote, chunk.excerpt_text):
            guardrail_notes.append(
                f"{item.source_id} içindeki alıntı kayıtlı kaynak metniyle birebir eşleşmedi."
                if language == "tr"
                else f"Quote for {item.source_id} did not verbatim match stored excerpt."
            )
            continue
        if raw_verification.verdict == VerdictStatus.SUPPORTED and is_quote_cut_at_boundary(
            item.quote, chunk
        ):
            guardrail_notes.append(
                f"{item.source_id} alıntısı kesme sınırında yarım kaldığı için destekli sayılamaz."
                if language == "tr"
                else f"Quote from {item.source_id} is near the truncation boundary and cannot be treated as complete support."
            )
            continue
        valid_items.append(
            VerdictEvidenceItem(
                source_id=item.source_id,
                chunk_id=item.chunk_id,
                quote=normalize_whitespace_nfc(item.quote),
                role=item.role,
                quote_language=item.quote_language,
                explanatory_note=item.explanatory_note,
            )
        )
        combined_chunks.append(chunk.excerpt_text)

    verdict = raw_verification.verdict
    rationale = raw_verification.rationale
    missing_info = raw_verification.missing_info
    claim_nature = raw_verification.claim_nature

    supporting_items = [v for v in valid_items if v.role == EvidenceRole.SUPPORTING]
    independent_ids = get_independent_source_ids(
        [v.source_id for v in supporting_items] or [v.source_id for v in valid_items],
        sources_by_id,
    )
    independent_count = len(independent_ids)
    all_quote_text = " ".join(v.quote for v in valid_items) or claim.initial_quote
    all_chunk_text = " ".join(combined_chunks)

    if verdict == VerdictStatus.SUPPORTED:
        if not supporting_items:
            verdict = VerdictStatus.INSUFFICIENT
            extra = (
                "Programatik kanıt denetimi: Geçerli ve kesilmemiş bir birebir alıntı doğrulanamadığı için karar 'Kanıt yetersiz' olarak düşürüldü."
                if language == "tr"
                else "Programmatic guardrail: Downgraded to 'insufficient' because no valid non-truncated verbatim quote could be verified."
            )
            rationale = f"{rationale} {extra}".strip()
            missing_info = missing_info or (
                "Kayıtlı kaynakta birebir ve tam alıntı bulunamadı."
                if language == "tr"
                else "No complete verbatim quote verified in stored sources."
            )
            guardrail_notes.append(extra)
        else:
            # C7: Soru cümlesi veya uzak gelecek projeksiyonu
            q_err = _is_question_or_distant_projection(claim.statement, all_quote_text, research_question)
            if q_err:
                verdict = VerdictStatus.INSUFFICIENT
                claim_nature = "question_or_heading"
                rationale = f"{rationale} {q_err}".strip()
                missing_info = missing_info or q_err
                guardrail_notes.append(q_err)

            # C5: Tarihsel arka plan / konu uyumsuzluğu
            if verdict == VerdictStatus.SUPPORTED:
                irr_err = _detect_temporal_or_topical_irrelevance(
                    claim.statement, all_quote_text, research_question
                )
                if irr_err:
                    verdict = VerdictStatus.IRRELEVANT
                    rationale = f"{rationale} {irr_err}".strip()
                    missing_info = missing_info or irr_err
                    guardrail_notes.append(irr_err)

            # C3: Dipnot / zaman damgası sayısının istatistik sanılması
            if verdict == VerdictStatus.SUPPORTED:
                fn_err = _detect_footnote_or_timestamp_number(
                    claim.statement, all_quote_text, all_chunk_text
                )
                if fn_err:
                    verdict = VerdictStatus.INSUFFICIENT
                    rationale = f"{rationale} {fn_err}".strip()
                    missing_info = missing_info or fn_err
                    guardrail_notes.append(fn_err)

            # C4: Video / konuşmacı beyanının bağımsız doğrulama olmadan istatistik sayılması
            if verdict == VerdictStatus.SUPPORTED:
                supp_sources = [sources_by_id[v.source_id] for v in supporting_items if v.source_id in sources_by_id]
                all_video = bool(supp_sources) and all(
                    s.source_type == SourceType.VIDEO_SOCIAL_MEDIA for s in supp_sources
                )
                speaker_cue = any(
                    tok in f"{claim.statement} {all_quote_text}".lower()
                    for tok in ("konuşmacı", "videoda", "youtube", "speaker", "iddia etti", "röportaj")
                )
                if (all_video or speaker_cue) and independent_count < 2:
                    verdict = VerdictStatus.INSUFFICIENT
                    claim_nature = "opinion_commentary"
                    v_err = (
                        "Video veya konuşmacı beyanı bağımsız istatistiksel kaynaklarla doğrulanmadığı için kesin kanıt olarak işaretlenmedi."
                        if language == "tr"
                        else "Video or speaker assertion lacks independent statistical corroboration."
                    )
                    rationale = f"{rationale} {v_err}".strip()
                    missing_info = missing_info or v_err
                    guardrail_notes.append(v_err)

            # C2: Paydasız ve popülasyonsuz yüzde genellemesi
            if verdict == VerdictStatus.SUPPORTED:
                pct_err = _detect_undefined_percentage_generalization(
                    claim.statement, all_quote_text, all_chunk_text
                )
                if pct_err:
                    verdict = VerdictStatus.INSUFFICIENT
                    rationale = f"{rationale} {pct_err}".strip()
                    missing_info = missing_info or pct_err
                    guardrail_notes.append(pct_err)

    elif verdict == VerdictStatus.CONTRADICTED:
        contra = [
            v
            for v in valid_items
            if v.role in (EvidenceRole.CONTRADICTING, EvidenceRole.CONFLICTING)
        ]
        if not contra:
            verdict = VerdictStatus.INSUFFICIENT
            extra = (
                "Programatik kanıt denetimi: Karşı iddiayı doğrulayan birebir alıntı bulunamadığından karar 'Kanıt yetersiz' olarak güncellendi."
                if language == "tr"
                else "Programmatic guardrail: Downgraded to 'insufficient' because no verbatim contradicting quote matched."
            )
            rationale = f"{rationale} {extra}".strip()
            guardrail_notes.append(extra)

    elif verdict == VerdictStatus.CONFLICTING:
        if len(independent_ids) < 2:
            verdict = VerdictStatus.INSUFFICIENT
            extra = (
                "Programatik kanıt denetimi: Çelişki kararı için en az iki bağımsız kaynak gerekir; kopya veya tek kaynak nedeniyle karar 'Kanıt yetersiz' olarak güncellendi."
                if language == "tr"
                else "Programmatic guardrail: 'Conflicting' requires at least two independent sources; duplicate or single source downgraded to 'insufficient'."
            )
            rationale = f"{rationale} {extra}".strip()
            missing_info = missing_info or (
                "Bağımsız ikinci bir ölçüm veya birincil kaynak eksik."
                if language == "tr"
                else "Second independent measurement or primary source missing."
            )
            guardrail_notes.append(extra)

    single_source_warning: str | None = None
    if verdict == VerdictStatus.SUPPORTED and independent_count == 1:
        single_source_warning = (
            "Tek bağımsız kaynağa dayanmaktadır; ikinci bağımsız doğrulama henüz eklenmemiştir."
            if language == "tr"
            else "Supported by a single independent source; independent corroboration is not yet available."
        )

    return VerificationRecord(
        claim_id=claim.claim_id,
        round_no=round_no,
        verdict=verdict,
        rationale=rationale,
        missing_info=missing_info,
        follow_up_query=raw_verification.follow_up_query,
        evidence_items=valid_items,
        is_final=True,
        causation_warning=raw_verification.causation_warning,
        guardrail_notes=guardrail_notes,
        claim_nature=claim_nature,
        numeric_context=raw_verification.numeric_context,
        independent_support_count=independent_count,
        single_source_warning=single_source_warning,
    )
