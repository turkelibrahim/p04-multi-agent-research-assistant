"""K11 10 soruluk karşılaştırma koşucusu: python -m p04_research.eval.runner --mode offline|live."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from p04_research.config import AppConfig, load_config
from p04_research.eval.benchmark_dataset import (
    BENCHMARK_CASES,
    BenchmarkCase,
    FixedBenchmarkProvider,
)
from p04_research.eval.metrics import (
    EvalClaimItem,
    EvalRunRecord,
    MethodMetrics,
    evaluate_method_runs,
)
from p04_research.eval.single_agent import run_single_agent_live, run_single_agent_offline
from p04_research.models import RunBundle, SubQuestionStatus
from p04_research.providers.base import SearchProvider, SearchResponse
from p04_research.providers.openai_llm import OpenAILLMProvider
from p04_research.workflow.graph import execute_research_run


class _FixedSourceOnlySearch(SearchProvider):
    def __init__(self, case: BenchmarkCase) -> None:
        self.case = case

    def search(self, query: str, max_results: int, depth: str) -> SearchResponse:
        return SearchResponse(
            query=query,
            items=list(self.case.sources)[:max_results],
            search_calls=1,
        )


def _bundle_to_eval_record(
    case: BenchmarkCase,
    bundle: RunBundle,
    human_scores: dict[str, bool],
) -> EvalRunRecord:
    sources_by_id = {
        s.source_id: " ".join(ch.excerpt_text for ch in bundle.chunks if ch.source_id == s.source_id)
        for s in bundle.sources
    }
    final_vers = {v.claim_id: v for v in bundle.verifications if v.is_final}
    eval_claims: list[EvalClaimItem] = []

    for claim in bundle.claims:
        ver = final_vers.get(claim.claim_id)
        if ver is None:
            continue
        cited = [ev.source_id for ev in ver.evidence_items] or [claim.initial_source_id]
        quoted_text = ver.evidence_items[0].quote if ver.evidence_items else claim.initial_quote
        quoted_sid = ver.evidence_items[0].source_id if ver.evidence_items else claim.initial_source_id
        h_key = f"multi_agent:{case.case_id}:{claim.claim_id}"
        eval_claims.append(
            EvalClaimItem(
                claim_id=claim.claim_id,
                statement=claim.statement,
                cited_source_ids=cited,
                quoted_text=quoted_text,
                quoted_source_id=quoted_sid,
                verdict=ver.verdict.value,
                human_supported=human_scores.get(h_key),
            )
        )

    return EvalRunRecord(
        case_id=case.case_id,
        method="multi_agent",
        status=bundle.metadata.status.value,
        duration_ms=bundle.metadata.duration_ms,
        model_calls=bundle.metadata.model_calls,
        search_calls=bundle.metadata.search_calls,
        sources_by_id=sources_by_id,
        claims=eval_claims,
        collected_sources=len(bundle.sources),
        accessible_sources=sum(1 for s in bundle.sources if s.is_accessible),
        answered_sub_questions=sum(
            1
            for sq in bundle.sub_questions
            if sq.status in (SubQuestionStatus.ANSWERED, SubQuestionStatus.PARTIALLY_ANSWERED)
        ),
        total_sub_questions=len(bundle.sub_questions),
        total_tokens=bundle.metadata.total_tokens,
        estimated_cost_usd=bundle.metadata.estimated_cost_usd,
    )


def _render_summary_markdown(
    mode: str,
    model_name: str,
    single_m: MethodMetrics,
    multi_m: MethodMetrics,
) -> str:
    lines = [
        "# K11 Akademik Değerlendirme Karşılaştırma Raporu",
        "",
        f"- **Çalışma Modu:** `{mode}`",
        f"- **Model:** `{model_name}`",
        "- **Veri Seti:** 10 sabit kaynaklı soru (`Q01`–`Q10`)",
        "- **Not:** Başarısız koşumların taslakları atıf/alıntı oranlarına dahil edilmez; paydası sıfır olan ölçütler `Ölçülemedi` olarak yazılır.",
        "",
        "| Ölçüt | Tek Ajan | Çok Ajanlı (P04) |",
        "|---|---|---|",
        f"| Tamamlanan Soru | {single_m.completed_cases}/{single_m.total_cases} | {multi_m.completed_cases}/{multi_m.total_cases} |",
        f"| Üretilen İddia Sayısı (Destekli) | {single_m.total_claims} ({single_m.supported_claims}) | {multi_m.total_claims} ({multi_m.supported_claims}) |",
        f"| Geçerli Atıf Oranı | {single_m.citation_validity_rate} | {multi_m.citation_validity_rate} |",
        f"| Birebir Alıntı Eşleşme Oranı | {single_m.quote_match_rate} | {multi_m.quote_match_rate} |",
        f"| İnsan Destek Oranı | {single_m.human_support_rate} | {multi_m.human_support_rate} |",
        f"| Ortalama Süre (ms) | {single_m.mean_duration_ms} | {multi_m.mean_duration_ms} |",
        f"| Toplam Model / Arama Çağrısı | {single_m.total_model_calls} / {single_m.total_search_calls} | {multi_m.total_model_calls} / {multi_m.total_search_calls} |",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    mode: str = "offline",
    config: AppConfig | None = None,
    output_dir: Path | None = None,
    human_scores_path: Path | None = None,
) -> tuple[MethodMetrics, MethodMetrics, Path | None]:
    """10 soruluk sabit kaynaklı K11 karşılaştırmasını tek ajan ve çok ajanlı için yürütür."""
    cfg = config or load_config()
    human_scores: dict[str, bool] = {}
    if human_scores_path and human_scores_path.exists():
        raw_scores = json.loads(human_scores_path.read_text(encoding="utf-8"))
        if isinstance(raw_scores, dict):
            human_scores = {str(k): bool(v) for k, v in raw_scores.items() if v is not None}

    single_runs: list[EvalRunRecord] = []
    multi_runs: list[EvalRunRecord] = []

    for case in BENCHMARK_CASES:
        if mode == "live":
            s_rec = run_single_agent_live(case, cfg)
            llm = OpenAILLMProvider(cfg)
            search = _FixedSourceOnlySearch(case)
            bundle, _, _ = execute_research_run(
                question=case.question,
                language=case.language,
                depth="fast",
                mode="live",
                config=cfg,
                llm_provider=llm,
                search_provider=search,
            )
            m_rec = _bundle_to_eval_record(case, bundle, human_scores)
        else:
            s_rec = run_single_agent_offline(case)
            fixed_prov = FixedBenchmarkProvider(case)
            bundle, _, _ = execute_research_run(
                question=case.question,
                language=case.language,
                depth="fast",
                mode="demo",
                config=cfg,
                llm_provider=fixed_prov,
                search_provider=fixed_prov,
            )
            m_rec = _bundle_to_eval_record(case, bundle, human_scores)

        for c in s_rec.claims:
            c.human_supported = human_scores.get(f"single_agent:{case.case_id}:{c.claim_id}")

        single_runs.append(s_rec)
        multi_runs.append(m_rec)

    single_m = evaluate_method_runs("single_agent", single_runs)
    multi_m = evaluate_method_runs("multi_agent", multi_runs)

    summary_path: Path | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_label = cfg.openai_model if mode == "live" else "offline-deterministic-fixed-sources"
        summary_md = _render_summary_markdown(mode, model_label, single_m, multi_m)
        summary_path = output_dir / "eval_summary.md"
        summary_path.write_text(summary_md, encoding="utf-8")

        results_payload = {
            "mode": mode,
            "model": model_label,
            "single_agent": asdict(single_m),
            "multi_agent": asdict(multi_m),
        }
        (output_dir / "eval_results.json").write_text(
            json.dumps(results_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        scoring_template: dict[str, bool | None] = {}
        for rec in [*single_runs, *multi_runs]:
            for cl in rec.claims:
                scoring_template[f"{rec.method}:{rec.case_id}:{cl.claim_id}"] = cl.human_supported
        (output_dir / "human_scoring_template.json").write_text(
            json.dumps(scoring_template, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return single_m, multi_m, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Araştırma Asistanı Değerlendirme Koşucusu")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval_output"))
    parser.add_argument("--human-scores", type=Path, default=None)
    args = parser.parse_args()

    single_m, multi_m, summary_path = run_benchmark(
        mode=args.mode,
        output_dir=args.output_dir,
        human_scores_path=args.human_scores,
    )
    print(f"Tek Ajan -> Tamamlanan: {single_m.completed_cases}/10, İddia: {single_m.total_claims}, Alıntı: {single_m.quote_match_rate}")
    print(f"Çok Ajan -> Tamamlanan: {multi_m.completed_cases}/10, İddia: {multi_m.total_claims}, Alıntı: {multi_m.quote_match_rate}")
    if summary_path:
        print(f"Rapor yazıldı: {summary_path}")


if __name__ == "__main__":
    main()
