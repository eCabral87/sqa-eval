import math
from dataclasses import dataclass
from statistics import mean, stdev

from sqa_eval.metrics import COMMON_METRICS, MetricDef


@dataclass
class AggregateResult:
    file_name: str
    system: str
    model_used: str
    common_score: float
    extended_score: float
    raw_scores: dict[str, float]


INF_BOUND_METRICS = {"sdr", "mcd", "lsd"}


class ScoreAggregator:
    def __init__(
        self,
        metrics: dict[str, MetricDef],
        weights: dict[str, float] | None = None,
    ):
        self._metrics = metrics
        self._weights: dict[str, float] = {}
        for name, m in metrics.items():
            weight = m.weight
            if weights and name in weights:
                weight = weights[name]
            self._weights[name] = weight

    def normalize(self, raw: dict[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, score in raw.items():
            if name not in self._metrics:
                continue
            m = self._metrics[name]
            result[name] = self._normalize_one(name, score, m)
        return result

    def _normalize_one(self, name: str, score: float, m: MetricDef) -> float:
        if math.isnan(score) or math.isinf(score):
            return 0.0

        if name == "sdr":
            clamped = max(-30.0, min(30.0, score))
            return (clamped + 30.0) / 60.0
        if name in ("mcd", "lsd"):
            clamped = max(0.0, min(20.0, score))
            return clamped / 20.0

        if m.min_val == float("-inf") or m.max_val == float("inf"):
            return max(0.0, min(1.0, score))
        return max(0.0, min(1.0, (score - m.min_val) / (m.max_val - m.min_val)))

    def compute(self, raw: dict[str, float]) -> float:
        total = 0.0
        total_weight = 0.0
        normalized = self.normalize(raw)
        for name, norm_val in normalized.items():
            m = self._metrics.get(name)
            if m is None:
                continue
            w = self._weights.get(name, m.weight)
            if m.direction == -1:
                norm_val = 1.0 - norm_val
            total += w * norm_val
            total_weight += w
        if total_weight == 0:
            return 0.0
        return total / total_weight

    def compute_common(self, raw: dict[str, float]) -> float:
        filtered: dict[str, float] = {}
        for name, score in raw.items():
            if name in COMMON_METRICS and name in self._metrics:
                filtered[name] = score
        return self.compute(filtered)

    def evaluate(
        self,
        file_name: str,
        system: str,
        model: str,
        raw: dict[str, float],
    ) -> AggregateResult:
        common = self.compute_common(raw)
        extended = self.compute(raw)
        return AggregateResult(
            file_name=file_name,
            system=system,
            model_used=model,
            common_score=common,
            extended_score=extended,
            raw_scores=dict(raw),
        )


def compare_systems(results: list[AggregateResult]) -> dict[str, dict]:
    groups: dict[str, list[float]] = {}
    extended_groups: dict[str, list[float]] = {}
    for r in results:
        groups.setdefault(r.system, []).append(r.common_score)
        extended_groups.setdefault(r.system, []).append(r.extended_score)

    out: dict[str, dict] = {}
    for system, scores in groups.items():
        out[system] = {
            "count": len(scores),
            "common_mean": mean(scores),
            "common_std": stdev(scores) if len(scores) > 1 else 0.0,
            "common_min": min(scores),
            "common_max": max(scores),
            "extended_mean": mean(extended_groups[system]),
            "extended_std": (
                stdev(extended_groups[system]) if len(extended_groups[system]) > 1 else 0.0
            ),
            "extended_min": min(extended_groups[system]),
            "extended_max": max(extended_groups[system]),
        }
    return out


def rank_systems(results: list[AggregateResult]) -> list[tuple[str, float]]:
    comparison = compare_systems(results)
    ranked = sorted(comparison.items(), key=lambda x: x[1]["common_mean"], reverse=True)
    return [(name, stats["common_mean"]) for name, stats in ranked]
