"""Kayıtlı FinalReport nesnesinden sade, bağlantıları kaynak adına gizlenmiş UTF-8 Markdown raporu üretimi."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from p04_research.models import RunBundle, RunMode
from p04_research.verification_rules import (
    clean_report_title as clean_report_title,
)
from p04_research.verification_rules import (
    normalize_tr_typos as normalize_tr_typos,
)
from p04_research.workflow.i18n import (
    t,
    verdict_label,
)

_BIDI_NOISE_RE = re.compile(r"[\u200b\u200e\u200f\u202a-\u202e\ufeff]")
_INLINE_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_NORM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

_PUBLISHER_BRAND_MAP: dict[str, str] = {
    "pwc.com.tr": "PwC Türkiye",
    "news.microsoft.com": "Microsoft News",
    "microsoft.com": "Microsoft",
    "webrazzi.com": "Webrazzi",
    "kariyer.net": "Kariyer.net",
    "indigodergisi.com": "İndigo Dergisi",
    "incturkiye.com": "Inc. Türkiye",
    "tr.linkedin.com": "LinkedIn",
    "linkedin.com": "LinkedIn",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "instagram.com": "Instagram",
    "x.com": "X",
    "twitter.com": "X",
    "personel.com": "Personel.com",
    "tr.wikipedia.org": "Vikipedi",
    "wikipedia.org": "Wikipedia",
    "github.com": "GitHub",
    "openai.com": "OpenAI",
    "anthropic.com": "Anthropic",
    "arxiv.org": "arXiv",
    "huggingface.co": "Hugging Face",
    "langchain.com": "LangChain",
    "mindstudio.ai": "MindStudio",
    "flowhunt.io": "FlowHunt",
    "ibm.com": "IBM",
    "aws.amazon.com": "AWS",
    "cloud.google.com": "Google Cloud",
    "nvidia.com": "NVIDIA",
    "medium.com": "Medium",
    "substack.com": "Substack",
    "forbes.com": "Forbes",
    "gartner.com": "Gartner",
    "mckinsey.com": "McKinsey",
    "deloitte.com": "Deloitte",
    "dergipark.org.tr": "DergiPark",
    "search.trdizin.gov.tr": "TR Dizin",
    "trdizin.gov.tr": "TR Dizin",
    "tubitak.gov.tr": "TÜBİTAK",
    "cdn.deneyap.org": "Deneyap",
    "deneyap.org": "Deneyap",
    "cubeincubation.com": "Cube Incubation",
    "acikerisim.bandirma.edu.tr": "Bandırma Onyedi Eylül Üniv.",
    "bandirma.edu.tr": "Bandırma Onyedi Eylül Üniv.",
    "mjer.inased.org": "MJER",
    "inased.org": "INASED",
}

_BRAND_TOKEN_CASING: dict[str, str] = {
    "ibm": "IBM",
    "mit": "MIT",
    "aws": "AWS",
    "bbc": "BBC",
    "cnn": "CNN",
    "pwc": "PwC",
    "sap": "SAP",
    "oecd": "OECD",
    "ieee": "IEEE",
    "arxiv": "arXiv",
    "github": "GitHub",
    "openai": "OpenAI",
    "langchain": "LangChain",
    "mindstudio": "MindStudio",
    "flowhunt": "FlowHunt",
    "nvidia": "NVIDIA",
    "linkedin": "LinkedIn",
    "youtube": "YouTube",
    "dergipark": "DergiPark",
    "tubitak": "TÜBİTAK",
    "trdizin": "TR Dizin",
    "deneyap": "Deneyap",
}

_MULTI_PART_TLDS = (
    ".com.tr",
    ".edu.tr",
    ".gov.tr",
    ".org.tr",
    ".bel.tr",
    ".k12.tr",
    ".net.tr",
    ".co.uk",
    ".ac.uk",
    ".org.uk",
    ".com.au",
    ".co.jp",
    ".co.in",
    ".com.br",
)


def _strip_noise(text: str) -> str:
    cleaned = _BIDI_NOISE_RE.sub("", text or "")
    return _WS_RE.sub(" ", cleaned).strip()


def is_external_web_url(url: str, is_mock: bool = False) -> bool:
    """Gerçek bir harici web bağlantısı olup olmadığını denetler (kurmaca/.local adresleri eler)."""
    if is_mock or not url:
        return False
    raw = url.strip()
    if not raw.startswith(("http://", "https://")):
        return False
    host = urlsplit(raw).netloc.lower().split(":", 1)[0]
    if not host or host.endswith(".local") or "kurmaca.p04.local" in host:
        return False
    return True


def clean_publisher_name(publisher: str, title: str = "", url: str = "") -> str:
    """Ham alan adı veya URL içeren yayıncı bilgisini okunaklı marka adına çevirir."""
    raw_pub = _strip_noise(_INLINE_URL_RE.sub("", publisher or ""))
    clean_t = _strip_noise(title)
    host = ""
    if url:
        host = urlsplit(url.strip()).netloc.lower().split(":", 1)[0]
        if host.startswith("www."):
            host = host[4:]

    lookup = raw_pub.lower()
    if lookup.startswith("www."):
        lookup = lookup[4:]
    if lookup in _PUBLISHER_BRAND_MAP:
        return _PUBLISHER_BRAND_MAP[lookup]
    if host and host in _PUBLISHER_BRAND_MAP:
        return _PUBLISHER_BRAND_MAP[host]

    candidate = raw_pub or host
    if not candidate:
        return ""

    if " " not in candidate and "." in candidate:
        for sep in (" | ", " - ", " — ", " · "):
            if sep in clean_t:
                tail = clean_t.rsplit(sep, 1)[-1].strip()
                if 2 <= len(tail) <= 32 and not tail.lower().startswith(("http://", "https://")):
                    return tail

        dom = candidate.lower()
        if dom.startswith("www."):
            dom = dom[4:]
        matched_multi = False
        for suf in _MULTI_PART_TLDS:
            if dom.endswith(suf) and len(dom) > len(suf):
                dom = dom[: -len(suf)]
                matched_multi = True
                break
        if not matched_multi and "." in dom:
            dom = dom.rsplit(".", 1)[0]
        parts = [
            p
            for p in dom.split(".")
            if p and p not in ("www", "m", "blog", "news", "docs", "tr", "en", "cdn", "search", "acikerisim")
        ]
        main_token = parts[-1] if parts else dom.split(".")[-1]
        if main_token:
            low_tok = main_token.lower()
            if low_tok in _BRAND_TOKEN_CASING:
                return _BRAND_TOKEN_CASING[low_tok]
            return main_token[:1].upper() + main_token[1:]

    return candidate


def clean_source_title(title: str, publisher_label: str = "") -> str:
    """Başlıktaki yönlü görünmez karakterleri, ham URL'leri ve tekrarlayan yayıncı/blog eklerini temizler."""
    cleaned = _strip_noise(title)
    if not cleaned or cleaned.lower().startswith(("http://", "https://")):
        return publisher_label or "Kaynak"

    if publisher_label:
        pub_low = publisher_label.lower()
        for sep in (" | ", " - ", " — ", " · "):
            if sep in cleaned:
                head, tail = cleaned.rsplit(sep, 1)
                tail_low = tail.strip().lower()
                if tail_low == pub_low or pub_low in tail_low or tail_low in pub_low:
                    if head.strip():
                        cleaned = head.strip()
                        break

    cleaned = re.sub(r"\s*[-|—·]\s*(?:Blog|Haber|Makale|Yazı)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*\.\.\.\s*$", "", cleaned).strip()
    return cleaned


def format_source_display_name(title: str, publisher: str, url: str = "") -> tuple[str, str]:
    """Kaynak başlığını ve yayıncı adını insan okunur biçimde döndürür."""
    pub_label = clean_publisher_name(publisher, title=title, url=url)
    title_label = clean_source_title(title, publisher_label=pub_label)
    return title_label, pub_label


def format_source_markdown_link(
    title: str,
    publisher: str,
    url: str,
    is_mock: bool = False,
) -> str:
    """URL'yi açıkta göstermeden temiz kaynak adının içine gömen Markdown bağlantısı üretir."""
    clean_t, clean_pub = format_source_display_name(title, publisher, url)
    if clean_pub and clean_pub.lower() not in clean_t.lower():
        label = f"{clean_t} — {clean_pub}"
    else:
        label = clean_t
    safe_label = label.replace("[", "(").replace("]", ")")
    if is_external_web_url(url, is_mock=is_mock):
        safe_url = url.strip().replace(" ", "%20").replace(")", "%29")
        return f"[{safe_label}]({safe_url})"
    return f"**{safe_label}**"


def is_statement_duplicate_of_quote(statement: str, quote: str) -> bool:
    """Bulgu ifadesi ile alıntı metninin aynı cümlenin tekrarı olup olmadığını denetler."""
    s_norm = _WS_RE.sub(" ", _PUNCT_NORM_RE.sub(" ", (statement or "").lower())).strip()
    q_norm = _WS_RE.sub(" ", _PUNCT_NORM_RE.sub(" ", (quote or "").lower())).strip()
    if not s_norm or not q_norm:
        return False
    if len(s_norm) >= 20 and (s_norm in q_norm or q_norm in s_norm):
        return True
    s_toks = set(s_norm.split())
    q_toks = set(q_norm.split())
    if not s_toks or not q_toks:
        return False
    overlap = len(s_toks & q_toks) / len(s_toks | q_toks)
    return overlap >= 0.72


def _format_token_summary(bundle: RunBundle) -> str:
    m = bundle.metadata
    lang = m.language.value
    if m.total_tokens is None:
        return t("not_measured", lang)
    if m.token_measurement_type == "estimated":
        cost_part = (
            f" (~${m.estimated_cost_usd:.5f})"
            if m.estimated_cost_usd is not None
            else ""
        )
        return f"{m.total_tokens} ({t('estimated_tag', lang)}{cost_part})"
    cost_part = (
        f" (${m.estimated_cost_usd:.5f})" if m.estimated_cost_usd is not None else ""
    )
    return f"{m.total_tokens}{cost_part}"


def render_report_markdown(bundle: RunBundle) -> str:
    """Saklanmış RunBundle ve FinalReport verisinden sade ve okunaklı UTF-8 Markdown raporu üretir."""
    m = bundle.metadata
    lang = m.language.value
    rep = bundle.report
    if rep is None:
        return f"# {m.question}\n\n{t('technical_error_prefix', lang)}: {m.error_message or '-'}\n"

    lines: list[str] = []
    lines.append(f"# {clean_report_title(rep.title)}")
    lines.append("")

    mode_str = t("mode_demo", lang) if m.mode == RunMode.DEMO else t("mode_live", lang)
    depth_str = t("depth_fast", lang) if m.depth.value == "fast" else t("depth_standard", lang)
    tokens_str = _format_token_summary(bundle)
    date_short = (m.started_at or "")[:10]

    if m.mode == RunMode.DEMO:
        lines.append(f"> **{t('mock_badge', lang)}** ({mode_str})")
        lines.append("")

    if lang == "tr":
        lines.append(
            f"_Tarih: {date_short} · Mod: {mode_str} ({depth_str}) · "
            f"İncelenen kaynak: {len(bundle.sources)} · Token: {tokens_str}_"
        )
    else:
        lines.append(
            f"_Date: {date_short} · Mode: {mode_str} ({depth_str}) · "
            f"Sources inspected: {len(bundle.sources)} · Tokens: {tokens_str}_"
        )
    lines.append("")

    if rep.error_notice:
        lines.append(f"> {rep.error_notice}")
        lines.append("")

    # 1. Kısa Yanıt
    lines.append(f"## {'Kısa Yanıt' if lang == 'tr' else 'Short Answer'}")
    lines.append("")
    lines.append(rep.short_answer)
    lines.append("")

    # 2. Temel Bulgular
    lines.append(f"## {t('report_heading_findings', lang)}")
    lines.append("")
    if not rep.supported_findings:
        lines.append(t("no_supported_findings", lang))
        lines.append("")
    else:
        for item in rep.supported_findings:
            cites = " ".join(f"[{sid}]" for sid in item.citation_ids)
            cite_suffix = f" {cites}" if cites else ""
            primary_q = item.quotes[0] if item.quotes else None
            is_dup = (
                primary_q is not None
                and not primary_q.explanatory_note
                and primary_q.quote_language == lang
                and is_statement_duplicate_of_quote(item.statement, primary_q.quote)
            )
            if is_dup and primary_q is not None:
                lines.append(f"- {primary_q.quote}{cite_suffix}")
            else:
                lines.append(f"- **{item.statement}**{cite_suffix}")
                if primary_q is not None:
                    lines.append(f'  > "{primary_q.quote}"')
                    if primary_q.explanatory_note:
                        lines.append(f"  > _{primary_q.explanatory_note}_")
            if item.causation_warning:
                note_lbl = "Not" if lang == "tr" else "Note"
                lines.append(f"  - _{note_lbl}: {item.causation_warning}_")
            lines.append("")

    # 3. Sınırlılıklar ve Çelişen Bulgular (yalnızca varsa gösterilir)
    if rep.conflicts or rep.limitations:
        section_title = (
            "Sınırlılıklar ve Çelişen Bulgular"
            if lang == "tr"
            else "Limitations & Conflicting Evidence"
        )
        lines.append(f"## {section_title}")
        lines.append("")
        for conf in rep.conflicts:
            lines.append(
                f"- **[{conf.side_a.source_id}] vs [{conf.side_b.source_id}]** — "
                f"{conf.statement}: {conf.explanation}"
            )
            lines.append(f'  > [{conf.side_a.source_id}]: "{conf.side_a.quote}"')
            lines.append(f'  > [{conf.side_b.source_id}]: "{conf.side_b.quote}"')
            lines.append("")
        for lim in rep.limitations:
            v_lbl = verdict_label(lim.verdict, lang)
            lines.append(f"- **{v_lbl}:** {lim.statement} — {lim.rationale}")
            for ce in lim.counter_evidence[:1]:
                lines.append(f'  > [{ce.source_id}]: "{ce.quote}"')
            lines.append("")

    # 4. Kaynakça (URL açıkta yazılmaz, kaynak adının içine gizlenir)
    lines.append(f"## {t('report_heading_bibliography', lang)}")
    lines.append("")
    if not rep.bibliography:
        lines.append("Kaynak kaydedilmedi." if lang == "tr" else "No sources recorded.")
    else:
        for b in rep.bibliography:
            pub_date = b.published_date or t("unknown", lang)
            src_link = format_source_markdown_link(b.title, b.publisher, b.url, is_mock=b.is_mock)
            extra_parts: list[str] = []
            if b.is_mock:
                extra_parts.append(f"[{t('mock_tag', lang)}]")
            if b.duplicate_of_source_id:
                extra_parts.append(
                    f"[Kopya: {b.duplicate_of_source_id}]"
                    if lang == "tr"
                    else f"[Duplicate: {b.duplicate_of_source_id}]"
                )
            extra_str = (" · " + " ".join(extra_parts)) if extra_parts else ""
            date_lbl = "Tarih" if lang == "tr" else "Date"
            lines.append(f"- **[{b.source_id}]** {src_link} · {date_lbl}: {pub_date}{extra_str}")
    lines.append("")
    return "\n".join(lines)
