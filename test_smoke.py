from sqa_eval import Evaluator, Experiment, InferenceEngine


# engine = InferenceEngine("5metric")
# scores = engine.predict("sp03.wav")  # → {mos: 3.2, ...}
# print(scores)

# With reference (22metric)
engine = InferenceEngine("both")
# scores = engine.predict("sp03.wav", ref_path="sp03.wav")
# print(scores)

# Batch
pairs = [("sp03.wav", None), ("sp03.wav", "sp03.wav")]
all_scores = engine.predict_batch(pairs)
print(all_scores)
