# sqa-eval

**Speech Quality Assessment** — score your audio with neural MOS metrics and rank enhancement algorithms in minutes, not days.

Backed by the [Uni-VERSA-Ext](https://github.com/urgent-challenge/urgent2026_challenge_track2/tree/main) framework.

📄 [Paper](https://arxiv.org/abs/2506.12260) · 🤗 [Hugging Face Models](https://huggingface.co/vvwangvv/)

- **GPU** when available, **CPU** otherwise — auto-detected
- **Cross-platform** — Windows, Linux, macOS (via `uv`)

---

## Test File Requirements

For reliable and subjectively correlated scores, test files should meet the following guidelines:

- **Duration** — [Uni-VERSA-Ext](https://github.com/urgent-challenge/urgent2026_challenge_track2/tree/main) works best with short files (10–20 s). Longer files do not necessarily yield better results.
- **Content** — Files should contain continuous speech without prolonged silence gaps between speech sections. See the `assets/` folder in the Uni-VERSA-Ext repository for examples. Following this guideline keeps the evaluation focused on speech quality and improves correlation with subjective ratings. Long silence gaps can mislead scores, especially for no-reference (non-intrusive) metrics.
- **Avoid** — Extreme data augmentation can mislead the model and produce unreliable scores.
- **Metric selection** — Review the metric descriptions in the [paper](https://arxiv.org/abs/2506.12260) (Section II-B, particularly Table 1) and choose the metrics that best match your requirements. Give more weight to them in the evaluation. See Score Aggregation for more details.

---

## Setup

```bash
git clone https://github.com/eCabral87/sqa-eval.git && cd sqa-eval

# One command, handles Python, virtualenv, and all deps:
uv sync --extra dev
```

That's it. You now have `python`, `pytest`, and the full inference stack ready to go.

**Audio loading** uses `soundfile` (bundles `libsndfile` for all platforms) — no system packages needed.

**CPU (all platforms):**

```bash
uv sync --extra dev
```

**GPU (Linux/Windows with NVIDIA):**

```bash
uv sync --extra dev --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## In a Nutshell

```python
from sqa_eval import Experiment

# --- Pit two denoisers against each other ---
#
# Directory layout expected:
#   recordings/
#   ├── dnn_v1/
#   │   ├── sample01.wav
#   │   └── sample02.wav
#   └── dnn_v2/
#       ├── sample01.wav
#       └── sample02.wav
#   clean_refs/
#   ├── REF_sample01.wav        (see Reference Convention below)
#   └── REF_sample02.wav
#
exp = Experiment(
    name="denoiser-shootout",
    base_dir="./recordings",          # parent dir containing one subdir per system
    systems=["dnn_v1", "dnn_v2"],     # system subdirectory names
    ref_dir="./clean_refs",           # clean references (REF_ prefix matched by stem)
    model="22metric",                 # "5metric" | "22metric" | "both"
)
exp.run()                             # scores every file across all systems
exp.report()                          # CSV, JSON, and plots land in results/
```

Every result gives you two aggregated scores:
- **`common_score`** — weighted average of only the 5 no-reference MOS metrics (`mos`, `dnsmos_ovrl`, `scoreq`, `utmos`, `nisqa_mos`)
- **`extended_score`** — weighted average of **all** metrics the model produced

With `"5metric"` the two scores are identical (only those 5 metrics exist). With `"22metric"` they diverge because `extended_score` also includes SDR, PESQ, MCD, LSD, speaker similarity, etc.
Use `"both"` when your dataset mixes files with and without references. Files that have a matching reference are scored with `"22metric"` (all metrics), while files without a reference fall back to `"5metric"` (no-reference metrics only).

For a deeper dive into every function, implementation details, and more use cases see the [Wiki](https://github.com/eCabral87/sqa-eval/wiki).

You can also adapt `test_smoke.py` — it is a ready-to-run template you can edit with your own directories and model choice.

```bash
uv run test_smoke.py
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

```python
from sqa_eval import Evaluator

# --- Score a single file ---
evaluator = Evaluator("5metric")      # 5 no-reference MOS metrics
result = evaluator.evaluate_file("sample.wav")
print(result.common_score)            # → 0.72
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

After the model produces raw metric scores (each on its own scale), the library normalises them into a single `[0, 1]` score:

1. **Normalise** — each raw score `s_k` is mapped to `[0, 1]` via min–max over its known range (e.g. MOS `[1,5]` → `(s-1)/4`; SDR `[-30,30]` → `(s+30)/60`)
2. **Flip lower-is-better** — metrics with `direction = -1` (MCD, LSD) become `1.0 - norm_val` so **1.0 is always best**
3. **Weighted average** — the final score is a weighted average, not a sum:

```
score = ( Σ w_k × norm_val_k ) / Σ w_k
```

Dividing by the total weight keeps the result in `[0, 1]` regardless of how many metrics contributed or what weights are set, making scores comparable across experiments. This combined score works similarly to the PRISM score proposed in [this paper](https://arxiv.org/abs/2512.16420).

**Weights**: each metric has a default weight of `1.0`, but you can pass custom per-metric weights to `Evaluator(model, weights={"sdr": 2.0, "mos": 0.5})`.

---

## API Tour

### `Evaluator` — one file or one folder

```python
from sqa_eval import Evaluator

e = Evaluator(model="22metric")

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
all_scores = engine.predict_batch(pairs)  # → [{mos: 3.2, ...}, {mos: 3.5, sdr: 12.1, ...}]

# Each result dict contains only the metrics the model computed for that pair.
# If a model requires a reference (e.g. "22metric") but ref_path is None,
# predict_batch raises ValueError — it does not silently produce garbage.
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
uv run pytest tests/ -v           # 50 unit tests, no GPU needed
```

Or do it all in one go:

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && uv run pytest tests/ -q
```

Tests mock the framework — no downloads, no GPU, no internet required.

### Integration tests

Full-stack tests that load the real model from HuggingFace and run inference on synthetic audio (generated on the fly):

```bash
uv run pytest tests/ -m integration -v
```

These require network access, a GPU (falls back to CPU), and may take several minutes.

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

---
