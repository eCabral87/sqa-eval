from __future__ import annotations

import warnings
from pathlib import Path

from sqa_eval.aggregator import AggregateResult, ScoreAggregator
from sqa_eval.engine import InferenceEngine
from sqa_eval.io import match_experiment_refs, resolve_experiment, scan_audio
from sqa_eval.metrics import METRICS_5, METRICS_22
from sqa_eval.plotter import Plotter
from sqa_eval.reporter import Reporter


class Evaluator:
    def __init__(self, model: str = "5metric", weights: dict[str, float] | None = None):
        self._model = model
        self._weights = weights

        self._engine_5 = None
        self._engine_22 = None

        if model == "both":
            self._engine_5 = InferenceEngine("5metric")
            self._engine_22 = InferenceEngine("22metric")
        else:
            self._engine_5 = InferenceEngine(model)

    def _get_aggregator(self, model_name: str) -> ScoreAggregator:
        if "22metric" in model_name:
            return ScoreAggregator(METRICS_22, self._weights)
        return ScoreAggregator(METRICS_5, self._weights)

    def evaluate_file(
        self,
        audio_path: str | Path,
        ref_path: str | Path | None = None,
        system: str = "default",
    ) -> AggregateResult:
        audio_path = Path(audio_path)

        if self._engine_22 is not None and ref_path is not None:
            scores_22 = self._engine_22.predict(audio_path, ref_path)
            agg = ScoreAggregator(METRICS_22, self._weights)
            return agg.evaluate(audio_path.name, system, "22metric", scores_22)

        if self._engine_22 is not None:
            scores_5 = self._engine_5.predict(audio_path)
            agg = ScoreAggregator(METRICS_5, self._weights)
            return agg.evaluate(audio_path.name, system, "5metric", scores_5)

        if ref_path is not None:
            scores = self._engine_5.predict(audio_path, ref_path)
            agg = self._get_aggregator(self._engine_5.model_name)
            return agg.evaluate(audio_path.name, system, self._engine_5.model_name, scores)
        else:
            scores = self._engine_5.predict(audio_path)
            agg = self._get_aggregator(self._engine_5.model_name)
            return agg.evaluate(audio_path.name, system, self._engine_5.model_name, scores)

    def evaluate_directory(
        self,
        audio_dir: str | Path,
        ref_dir: str | Path | None = None,
        recursive: bool = False,
    ) -> list[AggregateResult]:
        files = scan_audio(audio_dir, recursive=recursive)
        if not files:
            return []

        from sqa_eval.io import match_references

        ref_map = {}
        if ref_dir is not None:
            ref_map = match_references(files, ref_dir)

        results = []
        for f in files:
            ref = ref_map.get(f)
            result = self.evaluate_file(f, ref)
            results.append(result)
        return results

    def to_csv(self, results: list[AggregateResult], path: str | Path):
        Reporter(results).to_csv(path)

    def to_json(self, results: list[AggregateResult], path: str | Path):
        Reporter(results).to_json(path)


class Experiment:
    def __init__(
        self,
        name: str,
        base_dir: str | Path,
        systems: list[str],
        ref_dir: str | Path | None = None,
        model: str = "both",
        weights: dict[str, float] | None = None,
        output_dir: str | Path | None = None,
    ):
        self.name = name
        self.base_dir = Path(base_dir)
        self.systems = systems
        self.ref_dir = Path(ref_dir) if ref_dir else None
        self.model = model
        self.weights = weights

        if output_dir is None:
            output_dir = self.base_dir.parent / "results" / name
        self.output_dir = Path(output_dir)
        self._results: list[AggregateResult] = []

    @property
    def results(self) -> list[AggregateResult]:
        return self._results

    def run(self) -> list[AggregateResult]:
        system_files = resolve_experiment(self.base_dir, self.systems)
        for sys_name, files in system_files.items():
            if not files:
                warnings.warn(f"System '{sys_name}' has no audio files. Skipping.")

        if self.ref_dir:
            ref_mapping = match_experiment_refs(system_files, self.ref_dir)
        else:
            ref_mapping = {s: {f: None for f in files} for s, files in system_files.items()}

        evaluator = Evaluator(model=self.model, weights=self.weights)

        all_files = []
        for sys_name, files in system_files.items():
            all_files.extend(files)

        total = sum(len(v) for v in system_files.values())
        scored = 0

        for sys_name, files in system_files.items():
            refs = ref_mapping.get(sys_name, {})
            for f in files:
                ref = refs.get(f)
                result = evaluator.evaluate_file(f, ref, system=sys_name)
                scored += 1
                if scored % 10 == 0 or scored == total:
                    print(f"Scored {scored}/{total} files in {sys_name}...")
                self._results.append(result)

        return self._results

    def report(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        reporter = Reporter(self._results)
        reporter.to_csv(self.output_dir / "scores.csv")
        reporter.summary_to_csv(self.output_dir / "summary.csv")
        reporter.ranking_table().to_csv(self.output_dir / "ranking.csv", index=False)
        reporter.to_json(self.output_dir / "results.json")

        plotter = Plotter(self._results)
        plotter.all_plots(self.output_dir)
