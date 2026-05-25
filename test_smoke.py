from sqa_eval import Evaluator, Experiment, InferenceEngine


# --- Pit two denoisers against each other ---
exp = Experiment(
    name="denoiser-shootout",
    base_dir="./recordings",
    systems=["dnn_v1", "dnn_v2"],
    ref_dir="./recordings/clean_refs",
    model="22metric",
)
exp.run()
exp.report()
