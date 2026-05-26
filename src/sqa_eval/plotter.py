from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sqa_eval.aggregator import AggregateResult

STYLE = "seaborn-v0_8-darkgrid"


def _use_style():
    available = plt.style.available
    if STYLE in available:
        plt.style.use(STYLE)
    elif "dark_background" in available:
        plt.style.use("dark_background")


def _resolve_metric(r: AggregateResult, metric: str) -> float | None:
    if metric in ("common_score", "extended_score"):
        return getattr(r, metric, None)
    return r.raw_scores.get(metric)


class Plotter:
    def __init__(self, results: list[AggregateResult]):
        self._results = results

    @staticmethod
    def _group_by_system(results: list[AggregateResult]) -> dict[str, list[float]]:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in results:
            groups[r.system].append(r.common_score)
        return dict(groups)

    @staticmethod
    def _get_scores(results: list[AggregateResult], metric: str) -> dict[str, list[float]]:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in results:
            if metric in ("common_score", "extended_score"):
                groups[r.system].append(getattr(r, metric))
            else:
                val = r.raw_scores.get(metric)
                if val is not None:
                    groups[r.system].append(val)
        return dict(groups)

    def _save_or_return(self, fig: plt.Figure, path: str | Path | None) -> plt.Figure:
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    def bar_compare(
        self, metric: str = "common_score", path: str | Path | None = None
    ) -> plt.Figure:
        _use_style()
        groups = self._get_scores(self._results, metric)
        systems = list(groups.keys())
        means = [np.mean(groups[s]) for s in systems]
        stds = [np.std(groups[s]) for s in systems]

        fig, ax = plt.subplots(figsize=(10, max(4, len(systems) * 0.6)))
        y_pos = range(len(systems))
        _ = ax.barh(y_pos, means, xerr=stds, capsize=4, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(systems)
        ax.set_xlabel(metric)
        ax.set_title(f"Mean {metric} by System")
        ax.invert_yaxis()

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 0.01, i, f"{m:.3f} ± {s:.3f}", va="center", fontsize=9)

        fig.tight_layout()
        return self._save_or_return(fig, path)

    def box_plot(self, metric: str = "common_score", path: str | Path | None = None) -> plt.Figure:
        _use_style()
        groups = self._get_scores(self._results, metric)
        systems = list(groups.keys())
        data = [groups[s] for s in systems]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, tick_labels=systems, patch_artist=True)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Distribution by System")
        fig.tight_layout()
        return self._save_or_return(fig, path)

    def scatter_pair(
        self,
        system_a: str,
        system_b: str,
        metric: str = "common_score",
        path: str | Path | None = None,
    ) -> plt.Figure:
        _use_style()
        scores_a: dict[str, float] = {}
        scores_b: dict[str, float] = {}
        for r in self._results:
            val = _resolve_metric(r, metric)
            if r.system == system_a and val is not None:
                scores_a[r.file_name] = val
            elif r.system == system_b and val is not None:
                scores_b[r.file_name] = val

        common = sorted(set(scores_a) & set(scores_b))
        if not common:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f"No common files between {system_a} and {system_b}", ha="center", va="center")
            return self._save_or_return(fig, path)

        xs = [scores_a[f] for f in common]
        ys = [scores_b[f] for f in common]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(xs, ys, alpha=0.7)
        all_vals = xs + ys
        lim_min = min(all_vals) - 0.1
        lim_max = max(all_vals) + 0.1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4)
        ax.set_xlabel(f"{system_a} ({metric})")
        ax.set_ylabel(f"{system_b} ({metric})")
        ax.set_title(f"{system_a} vs {system_b} — {metric}")
        fig.tight_layout()
        return self._save_or_return(fig, path)

    def radar_chart(self, path: str | Path | None = None) -> plt.Figure:
        _use_style()
        all_metrics: set[str] = set()
        for r in self._results:
            all_metrics.update(r.raw_scores.keys())
        metric_set_per_system: dict[str, set[str]] = {}
        for r in self._results:
            metric_set_per_system.setdefault(r.system, set()).update(r.raw_scores.keys())

        common_metrics = list(all_metrics)
        for ms in metric_set_per_system.values():
            common_metrics = [m for m in common_metrics if m in ms]

        if not common_metrics:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No common metrics across systems", ha="center", va="center")
            return self._save_or_return(fig, path)

        system_means: dict[str, list[float]] = {}
        for system in metric_set_per_system:
            sys_results = [r for r in self._results if r.system == system]
            means = []
            for m in common_metrics:
                vals = [r.raw_scores[m] for r in sys_results if m in r.raw_scores]
                means.append(np.mean(vals) if vals else 0)
            system_means[system] = means

        num_vars = len(common_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for system, means in system_means.items():
            values = means + means[:1]
            ax.fill(angles, values, alpha=0.15)
            ax.plot(angles, values, linewidth=2, label=system)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(common_metrics, fontsize=8)
        ax.set_title("Mean Scores per Metric per System")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        return self._save_or_return(fig, path)

    def all_plots(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        self.bar_compare(path=output_dir / "bar_common_score.png")
        self.box_plot(path=output_dir / "box_common_score.png")
        self.radar_chart(path=output_dir / "radar.png")
        systems = sorted(set(r.system for r in self._results))
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                self.scatter_pair(
                    systems[i],
                    systems[j],
                    path=output_dir / f"scatter_{systems[i]}_vs_{systems[j]}.png",
                )
