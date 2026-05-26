"""Smoke test — adapt this file to start using the project."""

from sqa_eval import Experiment

# --- Configuration — edit these to match your setup ---
EXPERIMENT_NAME = "denoiser-shootout"
BASE_DIR = "./recordings"
SYSTEMS = ["dnn_v1", "dnn_v2"]
REF_DIR = "./recordings/clean_refs"
MODEL = "22metric"  # "5metric" | "22metric" | "both"
WEIGHTS = None  # None = all metrics weight 1.0; e.g. {"sdr": 2.0, "mos": 0.5}
OUTPUT_DIR = None  # None defaults to <base_dir>/../results/<name>/
# -----------------------------------------------------

exp = Experiment(
    name=EXPERIMENT_NAME,
    base_dir=BASE_DIR,
    systems=SYSTEMS,
    ref_dir=REF_DIR,
    model=MODEL,
    weights=WEIGHTS,
    output_dir=OUTPUT_DIR,
)
exp.run()
exp.report()
