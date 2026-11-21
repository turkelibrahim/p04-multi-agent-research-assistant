"""K11 akademik değerlendirme modülü (tek ajan vs çok ajanlı karşılaştırma)."""

from p04_research.eval.benchmark_dataset import BENCHMARK_CASES
from p04_research.eval.metrics import MethodMetrics, evaluate_method_runs
from p04_research.eval.runner import run_benchmark

__all__ = ["BENCHMARK_CASES", "MethodMetrics", "evaluate_method_runs", "run_benchmark"]
