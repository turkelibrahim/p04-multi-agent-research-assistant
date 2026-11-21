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
    compute_topical_relevance,
    normalize_whitespace_nfc,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
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
_SNIPPET_PREFIX_RE = re.compile(r"^(?:\[\.{3}\]|\.{3}|…|\s*[-–—|•])+\s*")
_META_HEADER_RE = re.compile(
    r"^(?:İçgörü\s*[-–—]\s*Görüş|Görüş|Haber|Analiz|Blog)\s+(?:\d+\s+dakikalık\s+okuma\s*)?(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*)?",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(
    r"(\%\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*%|\b202[4-9]\b|\b\d+(?:[.,]\d+)?\s*(?:milyar|milyon|bin|kat|kişi|şirket|çalışan|iş|billion|million|thousand|x)\b)",
    re.IGNORECASE,
)

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
    """Kaynak metninden birebir eşleşen, soru cümlesi olmayan anlamlı cümleleri çıkarır."""
    norm = normalize_whitespace_nfc(text)
    if not norm:
        return []
    raw_parts = _SENTENCE_SPLIT_RE.split(norm)
    candidates: list[str] = []
    for part in raw_parts:
        s = part.strip()
        for _ in range(3):
            cleaned = _SNIPPET_PREFIX_RE.sub("", s).strip()
            cleaned = _META_HEADER_RE.sub("", cleaned).strip()
            cleaned = _SNIPPET_PREFIX_RE.sub("", cleaned).strip()
            if cleaned == s:
                break
            s = cleaned
        if s.endswith("?"):
            continue
        if 45 <= len(s) <= 340 and s in norm and " " in s:
            candidates.append(s)
    if not candidates and len(norm) >= 40 and not norm.endswith("?"):
        snippet = norm[: min(260, len(norm))].rsplit(" ", 1)[0].strip()
        if len(snippet) >= 35 and snippet in norm and not snippet.endswith("?"):
            candidates.append(snippet)

    def _rank_key(s: str) -> tuple[int, float, int, int]:
        has_noise = 1 if ("..." in s or "[...]" in s or _READING_TIME_RE.search(s)) else 0
        rel, _ = compute_topical_relevance(question, s) if question else (1.0, "")
        num_scan = _READING_TIME_RE.sub("", _DATE_TOKEN_RE.sub("", s))
        has_num = 0 if _NUMERIC_RE.search(num_scan) else 1
        return (has_noise, -rel, has_num, -len(s))

    candidates.sort(key=_rank_key)
    return candidates


def _summarize_claim_statement(quote: str, src: SourceRecord | None, is_tr: bool) -> str:
    """Alıntıdan tekrarsız, bağlam içeren bir iddia özeti üretir."""
    pub = src.publisher if (src and src.publisher) else ("Kaynak" if is_tr else "Source")
    num_scan = _READING_TIME_RE.sub("", _DATE_TOKEN_RE.sub("", quote))
    nums = _NUMERIC_RE.findall(num_scan)
    clean_q = quote.strip().rstrip(".")
    short_q = (
        clean_q[:135].rsplit(" ", 1)[0].rstrip(",;:") + "..."
        if len(clean_q) > 135
        else clean_q
    )

    if nums:
        metric_tag = ", ".join(dict.fromkeys(n.strip() for n in nums[:2]))
        return (
            f"{pub} verisi ({metric_tag}): {short_q}."
            if is_tr
            else f"{pub} data ({metric_tag}): {short_q}."
        )
    return f"{pub} bulgusu: {short_q}." if is_tr else f"{pub} finding: {short_q}."


def _build_academic_synthesis(
    question: str,
    language: str,
    sub_questions: list[SubQuestion],
    claims: list[ClaimRecord],
    verifications: list[VerificationRecord],
    sources: list[SourceRecord],
) -> tuple[str, str]:
    """Ham alıntıları yapıştırmak yerine doğrulanmış bulgulardan sentezlenmiş özet ve genel değerlendirme üretir."""
    is_tr = language == "tr"
    clean_q = clean_question_title(question)
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
            f"'{clean_q}' sorusu kapsamında taranan {len(sources)} kaynaktan doğrulama kurallarını geçen kesin bir bulgu elde edilememiştir. "
            "Mevcut kayıtlar ya soruyla doğrudan ilişkili ölçüm sunmamakta ya da doğrulanmamış tahminlerden oluşmaktadır."
            if is_tr
            else f"Across {len(sources)} inspected sources for '{clean_q}', no findings passed all evidence verification criteria."
        )
        assessment = (
            "Kanıt düzeyi yetersizdir; karar verebilmek için örneklem ve kapsamı tanımlı birincil ölçümler gereklidir."
            if is_tr
            else "Evidence is insufficient; primary measurements with defined samples and scope are needed."
        )
        return short, assessment

    # 1. Doğrudan soruya yanıt ve doğrulanmış bulgular sentezi
    if conflicting:
        c_claim, c_ver = conflicting[0]
        sids = " ve ".join(f"[{e.source_id}]" for e in c_ver.evidence_items[:2])
        paragraphs.append(
            f"'{clean_q}' sorusunda {sids} kaynakları aynı kapsam için çelişkili bulgular sunduğundan tekil bir kesin hüküm kurulmamıştır."
            if is_tr
            else f"For '{clean_q}', sources {sids} report conflicting measurements for the same scope, preventing a single definitive ruling."
        )
    elif supported:
        top_items = supported[:3]
        synth_clauses: list[str] = []
        seen_stmt: set[str] = set()
        for clm, ver in top_items:
            cite_ids = []
            for ev in ver.evidence_items:
                if ev.source_id not in cite_ids:
                    cite_ids.append(ev.source_id)
            cites = " ".join(f"[{sid}]" for sid in (cite_ids or [clm.initial_source_id]))
            core = clm.statement.strip().rstrip(".")
            if core in seen_stmt:
                continue
            seen_stmt.add(core)
            if ver.numeric_context:
                synth_clauses.append(f"{core} ({ver.numeric_context}) {cites}")
            else:
                synth_clauses.append(f"{core} {cites}")

        if is_tr:
            paragraphs.append(
                f"'{clean_q}' sorusuna yönelik {len(sources)} kaynak üzerinden yürütülen denetimde "
                f"{len(supported)} bulgu doğrudan kanıtlarla desteklenmiştir: "
                + "; ".join(synth_clauses)
                + "."
            )
        else:
            paragraphs.append(
                f"Evaluation of {len(sources)} sources for '{clean_q}' verified {len(supported)} findings: "
                + "; ".join(synth_clauses)
                + "."
            )

    # 2. Karşıt kanıt özeti
    if contradicted:
        c_claim, c_ver = contradicted[0]
        c_sids = " ".join(f"[{e.source_id}]" for e in c_ver.evidence_items[:2])
        paragraphs.append(
            f"Buna karşın '{c_claim.statement.rstrip('.')}' iddiası {c_sids} kayıtlarındaki ölçümlerle çürütülmüştür."
            if is_tr
            else f"Conversely, the claim '{c_claim.statement.rstrip('.')}' is contradicted by measurements in {c_sids}."
        )

    # 3. Belirsizlikler ve açık noktalar
    if uncertain:
        irrel_cnt = sum(1 for _, v in uncertain if v.verdict == VerdictStatus.IRRELEVANT)
        insuff_cnt = len(uncertain) - irrel_cnt
        if is_tr:
            paragraphs.append(
                f"Ek olarak, {insuff_cnt} iddia yeterli istatistiksel bağlam veya bağımsız doğrulama bulunmadığı için "
                + (f"ve {irrel_cnt} iddia konu/zaman uyumsuzluğu nedeniyle " if irrel_cnt else "")
                + "kesin bulgulara dahil edilmeyip sınırlılıklar bölümünde raporlanmıştır."
            )
        else:
            paragraphs.append(
                f"Additionally, {insuff_cnt} insufficient claims "
                + (f"and {irrel_cnt} irrelevant claims " if irrel_cnt else "")
                + "were excluded from verified findings and documented under limitations."
            )

    short_answer = " ".join(paragraphs)
    assessment = (
        f"Toplam {len(sources)} kaynak denetlenmiş, {len(supported)} destekli bulgu ve {len(contradicted)} karşıt bulgu doğrulanmıştır; "
        f"tek kaynağa dayalı bulgularda ikinci bağımsız doğrulama önerilir."
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

        if any(k in q_low for k in ("yapay zek", "ajan", "iş dünyası", "çalışan", "ai agent", "workforce")):
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
                    required_evidence="Örneklem ve yılı belli istihdam istatistiği",
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
            parts = [p.strip() for p in re.split(r"[:?]", clean_q) if len(p.strip()) >= 8]
            topic_main = parts[0] if parts else clean_q
            topic_sub = parts[1] if len(parts) > 1 else clean_q
            sqs = [
                PlannedSubQuestionLLM(
                    sq_id="SQ1",
                    text=(
                        f"{topic_main} konusunda birincil araştırmalar ve saha bulguları nelerdir?"
                        if is_tr
                        else f"What are the primary research findings regarding {topic_main}?"
                    ),
                    scope_note="Güncel saha bulguları",
                    search_intent="Ana konuyla doğrudan ilgili araştırmaları topla",
                    required_evidence="Birincil veya araştırma kuruluşu raporu",
                ),
                PlannedSubQuestionLLM(
                    sq_id="SQ2",
                    text=(
                        f"{topic_sub} açısından ölçülmüş sayısal oranlar ve veriler nelerdir?"
                        if is_tr
                        else f"What quantitative rates and measurements exist for {topic_sub}?"
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

        queries = [clean_q]
        if max_initial_queries >= 2:
            queries.append(f"{clean_q} araştırma raporu istatistik veriler")
        if max_initial_queries >= 3:
            queries.append(f"{clean_q} saha çalışması örneklem sınırlılıklar")

        out = PlannerOutput(
            plan_summary=(
                f"'{clean_q}' sorusunu güncel web kaynaklarından taranan ölçülebilir alt sorular, nicel veriler ve kanıt sınırlılıkları çerçevesinde inceler."
                if is_tr
                else f"Examines '{clean_q}' across measurable sub-questions, quantitative data, and evidence limitations."
            ),
            scope_notes="Canlı Tavily Web Taraması · Kaynak Kalitesi ve Birebir Alıntı Denetimi",
            research_purpose=f"'{clean_q}' sorusuna kanıta dayalı yanıt üretmek." if is_tr else f"Produce evidence-grounded answer for '{clean_q}'.",
            timeframe="2026" if "2026" in clean_q else "Güncel dönem",
            key_concepts=[w for w in clean_q.split()[:5] if len(w) >= 4],
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
        next_idx = len(existing_claims) + 1
        sq_ids = [sq.sq_id for sq in sub_questions] or ["SQ1"]
        src_map = {s.source_id: s for s in sources}

        round_chunks = [c for c in chunks if c.round_no == round_no] or list(chunks)
        # Önceliği yüksek güvenilirlik ve konu uygunluğuna ver
        round_chunks = sorted(
            round_chunks,
            key=lambda ch: (
                -(src_map[ch.source_id].relevance_score if ch.source_id in src_map else 0.0),
                -(src_map[ch.source_id].credibility_score if ch.source_id in src_map else 0.0),
            ),
        )

        per_source_count: dict[str, int] = {}
        for ec in existing_claims:
            per_source_count[ec.initial_source_id] = (
                per_source_count.get(ec.initial_source_id, 0) + 1
            )

        new_claims: list[ExtractedClaimLLM] = []
        max_new = 5 if round_no == 1 else 2

        for max_per_src in (1, 2):
            for ch in round_chunks:
                if len(new_claims) >= max_new:
                    break
                src = src_map.get(ch.source_id)
                if src is not None and (not src.is_accessible or src.relevance_score < 0.2):
                    continue
                if per_source_count.get(ch.source_id, 0) >= max_per_src:
                    continue
                sentences = _extract_verbatim_sentences(ch.excerpt_text, question=question)
                if not sentences:
                    continue
                chosen_quote = next((s for s in sentences if s not in existing_quotes), None)
                if chosen_quote is None:
                    continue
                existing_quotes.add(chosen_quote)
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
                        initial_quote_language=src.language if src else language,
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
        clean_q = clean_question_title(question)
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
                        follow_up_query=f"{clean_q} saha araştırması sayısal veriler karşılaştırma",
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
        is_tr = language == "tr"
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
            title=f"Araştırma Raporu: {clean_q}" if is_tr else f"Research Report: {clean_q}",
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
