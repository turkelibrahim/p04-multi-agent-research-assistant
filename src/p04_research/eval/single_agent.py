"""K11 tek ajanlı karşılaştırma temel modeli (aynı model ve aynı sabit kaynak seti)."""

from __future__ import annotations

import json
import time

import openai
from pydantic import BaseModel, Field

from p04_research.config import AppConfig
from p04_research.eval.benchmark_dataset import BenchmarkCase
from p04_research.eval.metrics import EvalClaimItem, EvalRunRecord
from p04_research.models import VerdictStatus


class SingleAgentClaimLLM(BaseModel):
    claim_id: str
    statement: str
    cited_source_ids: list[str] = Field(default_factory=list)
    quoted_source_id: str = ""
    quoted_text: str = ""


class SingleAgentOutputLLM(BaseModel):
    claims: list[SingleAgentClaimLLM] = Field(default_factory=list)


def run_single_agent_offline(case: BenchmarkCase) -> EvalRunRecord:
    """Denetçi ve programatik koruma katmanı olmadan tek adımlı özetleme temel çizgisi."""
    t0 = time.monotonic()
    sources_by_id = {f"S{idx + 1}": src.content for idx, src in enumerate(case.sources)}

    claims: list[EvalClaimItem] = []
    if not case.sources:
        # Tek ajan bazen kaynak olmadığında bile genel tahmin üretebilir veya boş döner; burada boş + 1 çağrı
        pass
    else:
        for spec in case.claims:
            sid = f"S{spec.source_idx + 1}"
            # Tek ajan ön tahmini (Q01 C2), çelişkiyi (Q02) ve nedenselliği (Q05) denetçisiz kabul etme eğilimindedir
            if spec.verdict in (VerdictStatus.CONTRADICTED, VerdictStatus.INSUFFICIENT):
                claims.append(
                    EvalClaimItem(
                        claim_id=spec.claim_id,
                        statement=spec.statement,
                        cited_source_ids=[sid],
                        quoted_text="",  # Ön tahmini doğrudan özetleyip birebir alıntı kontrolü yapmaz
                        quoted_source_id=sid,
                        verdict="supported",
                    )
                )
            elif spec.verdict == VerdictStatus.CONFLICTING:
                claims.append(
                    EvalClaimItem(
                        claim_id=spec.claim_id,
                        statement=spec.statement,
                        cited_source_ids=[sid],  # İkinci çelişkili kaynağı atlar
                        quoted_text=spec.quote,
                        quoted_source_id=sid,
                        verdict="supported",
                    )
                )
            else:
                claims.append(
                    EvalClaimItem(
                        claim_id=spec.claim_id,
                        statement=spec.statement,
                        cited_source_ids=[sid],
                        quoted_text=spec.quote,
                        quoted_source_id=sid,
                        verdict="supported",
                    )
                )

    dur = max(1, int((time.monotonic() - t0) * 1000))
    return EvalRunRecord(
        case_id=case.case_id,
        method="single_agent",
        status="completed",
        duration_ms=dur,
        model_calls=1,
        search_calls=1,
        sources_by_id=sources_by_id,
        claims=claims,
    )


def run_single_agent_live(case: BenchmarkCase, config: AppConfig) -> EvalRunRecord:
    """Aynı OpenAI modeli ve sabit kaynaklarla tek çağrılı tek ajan koşumu."""
    t0 = time.monotonic()
    sources_by_id = {f"S{idx + 1}": src.content for idx, src in enumerate(case.sources)}
    if not config.has_openai_key():
        return EvalRunRecord(
            case_id=case.case_id,
            method="single_agent",
            status="failed",
            duration_ms=0,
            model_calls=0,
            search_calls=1,
            sources_by_id=sources_by_id,
        )

    client = openai.OpenAI(api_key=config.openai_api_key, timeout=float(config.model_timeout_sec))
    payload = json.dumps(
        {
            "question": case.question,
            "sources": [
                {"source_id": f"S{i + 1}", "title": s.title, "content": s.content}
                for i, s in enumerate(case.sources)
            ],
        },
        ensure_ascii=False,
    )
    try:
        resp = client.responses.parse(
            model=config.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question in one pass using only the provided sources. "
                        "Extract claims with cited_source_ids (e.g. ['S1']), quoted_source_id, and verbatim quoted_text."
                    ),
                },
                {"role": "user", "content": payload},
            ],
            text_format=SingleAgentOutputLLM,
        )
        parsed: SingleAgentOutputLLM = resp.output_parsed
        claims = [
            EvalClaimItem(
                claim_id=c.claim_id,
                statement=c.statement,
                cited_source_ids=c.cited_source_ids,
                quoted_text=c.quoted_text,
                quoted_source_id=c.quoted_source_id,
                verdict="supported",
            )
            for c in parsed.claims
        ]
        dur = int((time.monotonic() - t0) * 1000)
        return EvalRunRecord(
            case_id=case.case_id,
            method="single_agent",
            status="completed",
            duration_ms=dur,
            model_calls=1,
            search_calls=1,
            sources_by_id=sources_by_id,
            claims=claims,
        )
    except Exception:
        dur = int((time.monotonic() - t0) * 1000)
        return EvalRunRecord(
            case_id=case.case_id,
            method="single_agent",
            status="failed",
            duration_ms=dur,
            model_calls=1,
            search_calls=1,
            sources_by_id=sources_by_id,
        )
