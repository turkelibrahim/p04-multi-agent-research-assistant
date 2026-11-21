"""K11 10 soruluk akademik değerlendirme karşılaştırması ve metrik kuralları testleri."""

from __future__ import annotations

import json

from p04_research.eval.metrics import EvalClaimItem, EvalRunRecord, evaluate_method_runs
from p04_research.eval.runner import run_benchmark


def test_k11_offline_benchmark_ten_cases(tmp_path):
    out_dir = tmp_path / "eval_out"
    single_m, multi_m, summary_path = run_benchmark(mode="offline", output_dir=out_dir)

    assert single_m.total_cases == 10
    assert multi_m.total_cases == 10
    assert single_m.completed_cases == 10
    assert multi_m.completed_cases == 10
    assert multi_m.quote_match_count == multi_m.quote_checked_count
    assert multi_m.citation_valid_count == multi_m.citation_total_count
    assert single_m.human_support_rate == "Ölçülemedi"
    assert multi_m.human_support_rate == "Ölçülemedi"
    assert summary_path is not None and summary_path.exists()
    assert (out_dir / "eval_results.json").exists()
    assert (out_dir / "human_scoring_template.json").exists()


def test_k11_excludes_failed_runs_and_handles_human_scores(tmp_path):
    runs = [
        EvalRunRecord(
            case_id="Q01",
            method="multi_agent",
            status="failed",
            duration_ms=100,
            model_calls=1,
            search_calls=1,
            sources_by_id={},
            claims=[
                EvalClaimItem(
                    claim_id="C1",
                    statement="Başarısız taslak iddia",
                    cited_source_ids=["S999"],
                    quoted_text="uydurma",
                    quoted_source_id="S999",
                    verdict="supported",
                )
            ],
        ),
        EvalRunRecord(
            case_id="Q02",
            method="multi_agent",
            status="completed",
            duration_ms=200,
            model_calls=4,
            search_calls=1,
            sources_by_id={"S1": "Gerçek ölçüm değeri 42 birimdir."},
            claims=[
                EvalClaimItem(
                    claim_id="C1",
                    statement="Ölçüm değeri 42 birimdir.",
                    cited_source_ids=["S1"],
                    quoted_text="Gerçek ölçüm değeri 42 birimdir.",
                    quoted_source_id="S1",
                    verdict="supported",
                    human_supported=True,
                )
            ],
        ),
    ]
    metrics = evaluate_method_runs("multi_agent", runs)
    assert metrics.completed_cases == 1
    assert metrics.total_claims == 1
    assert metrics.citation_valid_count == 1 and metrics.citation_total_count == 1
    assert "100.0%" in metrics.human_support_rate

    # İsteğe bağlı insan puanlama dosyasından okuma
    scores_file = tmp_path / "scores.json"
    scores_file.write_text(json.dumps({"multi_agent:Q01:C1": True, "multi_agent:Q03:C1": False}), encoding="utf-8")
    _, multi_scored, _ = run_benchmark(mode="offline", output_dir=tmp_path / "eval_scored", human_scores_path=scores_file)
    assert multi_scored.human_scored_count == 2
    assert "50.0% (1/2)" == multi_scored.human_support_rate
