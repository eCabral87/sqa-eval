from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sqa_eval.aggregator import AggregateResult, compare_systems, rank_systems


class Reporter:
    def __init__(self, results: list[AggregateResult]):
        self._results = sorted(results, key=lambda r: r.common_score, reverse=True)

    @property
    def results(self) -> list[AggregateResult]:
        return self._results

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self._results:
            row = {
                "file_name": r.file_name,
                "system": r.system,
                "model_used": r.model_used,
                "common_score": r.common_score,
                "extended_score": r.extended_score,
            }
            row.update(r.raw_scores)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path):
        df = self.to_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def summary_table(self) -> pd.DataFrame:
        comp = compare_systems(self._results)
        rows = []
        for system, stats in comp.items():
            rows.append(
                {
                    "system": system,
                    "count": stats["count"],
                    "common_mean": stats["common_mean"],
                    "common_std": stats["common_std"],
                    "common_min": stats["common_min"],
                    "common_max": stats["common_max"],
                    "extended_mean": stats["extended_mean"],
                    "extended_std": stats["extended_std"],
                    "extended_min": stats["extended_min"],
                    "extended_max": stats["extended_max"],
                }
            )
        return pd.DataFrame(rows)

    def summary_to_csv(self, path: str | Path):
        df = self.summary_table()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def ranking_table(self) -> pd.DataFrame:
        ranked = rank_systems(self._results)
        rows = []
        for rank, (system, score) in enumerate(ranked, start=1):
            rows.append({"rank": rank, "system": system, "mean_common_score": score})
        return pd.DataFrame(rows)

    def to_json(self, path: str | Path):
        comparison = compare_systems(self._results)
        ranked = rank_systems(self._results)

        data = {
            "per_file": [
                {
                    "file_name": r.file_name,
                    "system": r.system,
                    "model_used": r.model_used,
                    "common_score": r.common_score,
                    "extended_score": r.extended_score,
                    "raw_scores": r.raw_scores,
                }
                for r in self._results
            ],
            "aggregates": comparison,
            "ranking": [
                {"rank": i + 1, "system": name, "mean_common_score": score}
                for i, (name, score) in enumerate(ranked)
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def merge_reports(reports: list[Reporter]) -> Reporter:
        merged: list[AggregateResult] = []
        for report in reports:
            merged.extend(report.results)
        return Reporter(merged)
