# sqa-eval

**Speech Quality Assessment** — score your audio with neural MOS metrics and rank enhancement algorithms in minutes, not days.

Backed by the [Uni-VERSA-Ext](https://arxiv.org/abs/2506.12260) framework.

- **GPU** when available, **CPU** otherwise — auto-detected
- **Cross-platform** — Windows, Linux, macOS (via `uv`)
- **50 tests** — zero network calls, zero GPU required

---

## Setup

```bash
git clone <repo-url> && cd eval-sqa-speech

# One command, handles Python, virtualenv, and all deps:
uv sync --extra dev
```

That's it. You now have `python`, `pytest`, and the full inference stack ready to go.

**Audio loading** uses `soundfile` (bundles `libsndfile` for all platforms) — no system packages needed.

**Platform note:** `pyproject.toml` registers the [PyTorch CUDA index](https://download.pytorch.org/whl/cu128) alongside PyPI so Linux/Windows users get GPU-capable wheels. On macOS `uv` falls back to PyPI's CPU wheels automatically.

---

## In a Nutshell

```python
from sqa_eval import Evaluator, Experiment

# --- Score a single file ---
evaluator = Evaluator("5metric")          # 5 no-reference MOS metrics
result = evaluator.evaluate_file("sample.wav")
print(result.common_score)                # → 0.72

# --- Pit two denoisers against each other ---
exp = Experiment(
    name="denoiser-shootout",
    base_dir="./recordings",
    systems=["dnn_v1", "dnn_v2"],
    ref_dir="./clean_refs",
    model="22metric",
)
exp.run()
exp.report()                              # CSV, JSON, and plots land in results/
```

Open `results/denoiser-shootout/` and you'll find:

```
results/denoiser-shootout/
├── scores.csv
├── summary.csv
├── ranking.csv
├── results.json
├── bar_common_score.png
├── box_common_score.png
├── radar.png
└── scatter_dnn_v1_vs_dnn_v2.png
```

---

## Models

| Alias | HF Repo | # Metrics | Needs Clean Ref? |
|-------|---------|-----------|------------------|
| `"5metric"` | `vvwangvv/universa-ext_wavlm-base_5metric` | 5 | No |
| `"22metric"` | `vvwangvv/universa-ext_wavlm-base_22metric` | 22 | Yes (for SDR, PESQ, MCD...) |

Or pass any HuggingFace repo ID directly: `Evaluator("org/custom-model")`.

### GPU / CPU

`InferenceEngine` auto-detects CUDA. If it's not available, a warning is printed once and inference falls back to CPU:

```python
>>> from sqa_eval import InferenceEngine
>>> engine = InferenceEngine("5metric")
>>> engine.device
'cuda'   # or 'cpu' if no GPU
```

You can force CPU by unsetting the device variable before running:

```bash
CUDA_VISIBLE_DEVICES="" uv run python my_script.py
```

---

## Score Aggregation

The library implements Equation (2) from the Uni-VERSA-Ext paper:

```
L_score = Σ w_k × direction_k × normalize(s_k)
```

- Each metric is mapped to [0, 1] using its known min/max range
- Lower-is-better metrics (MCD, LSD) are flipped so they contribute positively
- User-configurable weights let you emphasise what matters (default: equal)

---

## API Tour

### `Evaluator` — one file or one folder

```python
from sqa_eval import Evaluator

e = Evaluator(model="5metric")

result = e.evaluate_file("noisy.wav")                       # single file
results = e.evaluate_directory("./speech/", ref_dir="./refs/")  # whole folder

e.to_csv(results, "scores.csv")
e.to_json(results, "results.json")
```

### `Experiment` — multi-system comparison

```python
from sqa_eval import Experiment

exp = Experiment(
    name="my-comparison",
    base_dir="./outputs",           # contains dnn_v1/, dnn_v2/ subdirs
    systems=["dnn_v1", "dnn_v2"],
    ref_dir="./clean",              # REF_file1.wav, REF_file2.wav, ...
    model="both",                   # "5metric" | "22metric" | "both"
)

exp.run()     # prints progress: "Scored 42/50 files in dnn_v1..."
exp.report()  # dumps everything into results/my-comparison/
```

### `InferenceEngine` — raw predictions

```python
from sqa_eval import InferenceEngine

engine = InferenceEngine("5metric")
scores = engine.predict("audio.wav")                  # → {mos: 3.2, ...}

# With reference (22metric)
scores = engine.predict("degraded.wav", ref_path="clean.wav")

# Batch
pairs = [("a.wav", None), ("b.wav", "ref_b.wav")]
all_scores = engine.predict_batch(pairs)
```

---

## File Layout

```
src/sqa_eval/
├── __init__.py      # public API
├── metrics.py       # MetricDef, METRICS_5, METRICS_22
├── engine.py        # InferenceEngine (wraps Uni-VERSA-Ext)
├── aggregator.py    # ScoreAggregator + system ranking
├── io.py            # scan_audio, match_references, resolve_experiment
├── reporter.py      # CSV / JSON / summary table exports
├── plotter.py       # bar, box, scatter, radar charts
└── experiment.py    # Evaluator + Experiment high-level API
```

---

## Testing & Code Quality

```bash
uv run ruff format src/ tests/    # formatter (black-compatible, just faster)
uv run ruff check src/ tests/     # linter
uv run pytest tests/ -v           # 50 tests, no GPU needed
```

Or do it all in one go:

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && uv run pytest tests/ -q
```

Tests mock the framework — no downloads, no GPU, no internet required.

---

## Reference Convention

Put clean reference files in a `ref_dir` with the prefix `REF_`:

```
clean_refs/
├── REF_sample01.wav
└── REF_sample02.wav

outputs/dnn_v1/
├── sample01.wav   → paired with REF_sample01.wav
└── sample02.wav   → paired with REF_sample02.wav
```

Files without a matching reference are scored with no-ref metrics only.
