from __future__ import annotations

import math
import warnings
from pathlib import Path

from sqa_eval.aggregator import AggregateResult, ScoreAggregator
from sqa_eval.engine import InferenceEngine
from sqa_eval.io import match_experiment_refs, resolve_experiment, scan_audio
from sqa_eval.metrics import METRICS_5, METRICS_22
from sqa_eval.plotter import Plotter
from sqa_eval.preprocess import Preprocessor
from sqa_eval.reporter import Reporter


class Evaluator:
    def __init__(
        self,
        model: str = "5metric",
        weights: dict[str, float] | None = None,
        preprocess: bool = False,
    ):
        """Score a single file, a directory, or export results.

        Parameters
        ----------
        model : str, default "5metric"
            Model alias or HF repo ID. One of ``"5metric"``, ``"22metric"``,
            ``"both"``, or any HuggingFace repo ID (e.g.
            ``"org/custom-model"``).
        weights : dict[str, float] | None, optional
            Per-metric weight overrides, e.g. ``{"sdr": 2.0, "mos": 0.5}``.
            ``None`` uses each metric's default weight (1.0).
        preprocess : bool, default False
            If ``True``, run VAD-based speech extraction on test files before
            scoring to remove silence gaps. Only applies to no-reference
            evaluation (no ``ref_path``).
        """
        self._model = model
        self._weights = weights
        self._preprocess = preprocess
        self._preprocessor: Preprocessor | None = Preprocessor() if preprocess else None

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
        """Score a single audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file to evaluate.
        ref_path : str | Path | None, optional
            Path to a clean reference file. Required for ``22metric``
            metrics that need a reference (SDR, PESQ, MCD, LSD, etc.).
        system : str, default "default"
            System label used in reports and plots.

        Returns
        -------
        AggregateResult
            Aggregated scores (common, extended) plus raw per-metric values.
        """
        audio_path = Path(audio_path)

        if self._engine_22 is not None and ref_path is not None:
            scores_22 = self._engine_22.predict(audio_path, ref_path)
            agg = ScoreAggregator(METRICS_22, self._weights)
            return agg.evaluate(audio_path.name, system, "22metric", scores_22)

        if self._engine_22 is not None:
            if self._preprocess:
                return self._evaluate_preprocessed(audio_path, system, self._engine_5)
            scores_5 = self._engine_5.predict(audio_path)
            agg = ScoreAggregator(METRICS_5, self._weights)
            return agg.evaluate(audio_path.name, system, "5metric", scores_5)

        if ref_path is not None:
            scores = self._engine_5.predict(audio_path, ref_path)
            agg = self._get_aggregator(self._engine_5.model_name)
            return agg.evaluate(audio_path.name, system, self._engine_5.model_name, scores)
        else:
            if self._preprocess:
                return self._evaluate_preprocessed(audio_path, system, self._engine_5)
            scores = self._engine_5.predict(audio_path)
            agg = self._get_aggregator(self._engine_5.model_name)
            return agg.evaluate(audio_path.name, system, self._engine_5.model_name, scores)

    def _evaluate_preprocessed(
        self, audio_path: Path, system: str, engine: InferenceEngine
    ) -> AggregateResult:
        assert self._preprocessor is not None
        chunks = self._preprocessor.process_file(audio_path)

        all_raw: list[dict[str, float]] = []
        for chunk, sr in chunks:
            raw = engine.predict_from_tensor(chunk, sr)
            all_raw.append(raw)

        avg_raw = self._average_scores(all_raw)
        agg = self._get_aggregator(engine.model_name)
        return agg.evaluate(audio_path.name, system, engine.model_name, avg_raw)

    @staticmethod
    def _average_scores(scores_list: list[dict[str, float]]) -> dict[str, float]:
        if not scores_list:
            return {}
        if len(scores_list) == 1:
            return scores_list[0]
        keys = set.union(*[set(s.keys()) for s in scores_list])
        averaged: dict[str, float] = {}
        for key in keys:
            vals = [s.get(key, float("nan")) for s in scores_list]
            valid = [v for v in vals if not math.isnan(v)]
            averaged[key] = sum(valid) / len(valid) if valid else float("nan")
        return averaged

    def evaluate_directory(
        self,
        audio_dir: str | Path,
        ref_dir: str | Path | None = None,
        recursive: bool = False,
    ) -> list[AggregateResult]:
        """Score every audio file in a directory.

        Parameters
        ----------
        audio_dir : str | Path
            Directory containing audio files to evaluate.
        ref_dir : str | Path | None, optional
            Directory with ``REF_*`` reference files. References are
            matched by stem to files in ``audio_dir``.
        recursive : bool, default False
            If ``True``, recurse into subdirectories.

        Returns
        -------
        list[AggregateResult]
            One result per audio file found.
        """
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
        """Export results to CSV.

        Parameters
        ----------
        results : list[AggregateResult]
            Results from :meth:`evaluate_file` or :meth:`evaluate_directory`.
        path : str | Path
            Output CSV file path.
        """
        Reporter(results).to_csv(path)

    def to_json(self, results: list[AggregateResult], path: str | Path):
        """Export results to JSON.

        Parameters
        ----------
        results : list[AggregateResult]
            Results from :meth:`evaluate_file` or :meth:`evaluate_directory`.
        path : str | Path
            Output JSON file path.
        """
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
        preprocess: bool = False,
    ):
        """Compare multiple systems in a single run.

        Parameters
        ----------
        name : str
            Experiment name (used as output subdirectory name).
        base_dir : str | Path
            Parent directory containing one subdirectory per system.
        systems : list[str]
            System subdirectory names, e.g. ``["dnn_v1", "dnn_v2"]``.
        ref_dir : str | Path | None, optional
            Directory with ``REF_*`` reference files.
        model : str, default "both"
            Model alias or HF repo ID. ``"both"`` runs both
            ``5metric`` and ``22metric`` models.
        weights : dict[str, float] | None, optional
            Per-metric weight overrides.
        output_dir : str | Path | None, optional
            Output directory for reports and plots. Defaults to
            ``<base_dir>/../results/<name>/``.
        preprocess : bool, default False
            If ``True``, run VAD-based speech extraction on test files
            before scoring.
        """
        self.name = name
        self.base_dir = Path(base_dir)
        self.systems = systems
        self.ref_dir = Path(ref_dir) if ref_dir else None
        self.model = model
        self.weights = weights
        self.preprocess = preprocess

        if output_dir is None:
            output_dir = self.base_dir.parent / "results" / name
        self.output_dir = Path(output_dir)
        self._results: list[AggregateResult] = []

    @property
    def results(self) -> list[AggregateResult]:
        return self._results

    def run(self) -> list[AggregateResult]:
        """Run the experiment: score all files across all systems.

        Iterates over every audio file in each system directory, scores
        it via :class:`Evaluator`, and collects the results.

        Returns
        -------
        list[AggregateResult]
            All per-file results from every system.
        """
        system_files = resolve_experiment(self.base_dir, self.systems)
        for sys_name, files in system_files.items():
            if not files:
                warnings.warn(f"System '{sys_name}' has no audio files. Skipping.")

        if self.ref_dir:
            ref_mapping = match_experiment_refs(system_files, self.ref_dir)
        else:
            ref_mapping = {s: {f: None for f in files} for s, files in system_files.items()}

        evaluator = Evaluator(model=self.model, weights=self.weights, preprocess=self.preprocess)

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
        """Export all results to CSV, JSON, and plots.

        Writes to :attr:`output_dir`:
          - ``scores.csv`` — per-file scores
          - ``summary.csv`` — per-system statistics
          - ``ranking.csv`` — systems ordered by mean common score
          - ``results.json`` — full data
          - ``bar_common_score.png``, ``box_common_score.png``,
            ``radar.png``, ``scatter_*.png`` — visualisations
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        reporter = Reporter(self._results)
        reporter.to_csv(self.output_dir / "scores.csv")
        reporter.summary_to_csv(self.output_dir / "summary.csv")
        reporter.ranking_table().to_csv(self.output_dir / "ranking.csv", index=False)
        reporter.to_json(self.output_dir / "results.json")

        plotter = Plotter(self._results)
        plotter.all_plots(self.output_dir)
