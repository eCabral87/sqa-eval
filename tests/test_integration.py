"""Integration tests using the real Uni-VERSA-Ext model with generated audio.

Covers all examples from the "In a Nutshell" and "API Tour" sections of the README.

These tests download the actual model from HuggingFace and run real inference.
They require network access and may be slow.
"""

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from sqa_eval import Evaluator, Experiment, InferenceEngine

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _sine_wav(path, duration=1.0, sr=16000, freq=440):
    n_samples = int(duration * sr)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        for i in range(n_samples):
            sample = int(16384 * math.sin(2 * math.pi * freq * i / sr))
            f.writeframes(struct.pack("<h", sample))
    return path


@pytest.fixture(scope="module")
def degraded_wav(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("audio")
    return _sine_wav(tmp / "degraded.wav", freq=440)


@pytest.fixture(scope="module")
def ref_wav(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("audio_ref")
    return _sine_wav(tmp / "REF_clean.wav", freq=440)


class TestInferenceEngineAPITour:
    """Covers the InferenceEngine section of API Tour."""

    def test_predict_no_ref(self, degraded_wav):
        engine = InferenceEngine("5metric")
        scores = engine.predict(degraded_wav)
        assert isinstance(scores, dict)
        assert "mos" in scores
        assert all(isinstance(v, float) for v in scores.values())

    def test_predict_with_ref(self, degraded_wav, ref_wav):
        engine = InferenceEngine("22metric")
        scores = engine.predict(degraded_wav, ref_path=ref_wav)
        assert isinstance(scores, dict)
        assert "mos" in scores
        assert "pesq" in scores

    def test_predict_batch(self, degraded_wav):
        engine = InferenceEngine("5metric")
        pairs = [(degraded_wav, None), (degraded_wav, None)]
        results = engine.predict_batch(pairs)
        assert len(results) == 2
        assert "mos" in results[0]
        assert "mos" in results[1]


class TestEvaluatorAPITour:
    """Covers the Evaluator section of API Tour."""

    def test_evaluate_file(self, degraded_wav):
        e = Evaluator(model="5metric")
        result = e.evaluate_file(degraded_wav)
        assert result.file_name == degraded_wav.name
        assert isinstance(result.common_score, float)
        assert 0 <= result.common_score <= 1

    def test_evaluate_directory(self, degraded_wav, tmp_path):
        speech_dir = tmp_path / "speech"
        speech_dir.mkdir()
        (speech_dir / "sample01.wav").write_bytes(degraded_wav.read_bytes())
        (speech_dir / "sample02.wav").write_bytes(degraded_wav.read_bytes())
        e = Evaluator(model="5metric")
        results = e.evaluate_directory(speech_dir)
        assert len(results) == 2

    def test_to_csv(self, degraded_wav, tmp_path):
        e = Evaluator(model="5metric")
        result = e.evaluate_file(degraded_wav)
        csv_path = tmp_path / "scores.csv"
        e.to_csv([result], csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "common_score" in content
        assert degraded_wav.name in content

    def test_to_json(self, degraded_wav, tmp_path):
        e = Evaluator(model="5metric")
        result = e.evaluate_file(degraded_wav)
        json_path = tmp_path / "results.json"
        e.to_json([result], json_path)
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "per_file" in data
        assert "aggregates" in data


class TestExperimentNutshell:
    """Covers the Experiment section of In a Nutshell."""

    def test_experiment_run_and_report(self, degraded_wav, tmp_path):
        recordings = tmp_path / "recordings"
        (recordings / "dnn_v1").mkdir(parents=True)
        (recordings / "dnn_v2").mkdir(parents=True)
        (recordings / "dnn_v1" / "sample.wav").write_bytes(degraded_wav.read_bytes())
        (recordings / "dnn_v2" / "sample.wav").write_bytes(degraded_wav.read_bytes())

        exp = Experiment(
            name="denoiser-shootout",
            base_dir=recordings,
            systems=["dnn_v1", "dnn_v2"],
            ref_dir=None,
            model="5metric",
            output_dir=tmp_path / "results" / "denoiser-shootout",
        )
        exp.run()
        exp.report()

        out = exp.output_dir
        expected = [
            "scores.csv",
            "summary.csv",
            "ranking.csv",
            "results.json",
            "bar_common_score.png",
            "box_common_score.png",
            "radar.png",
            "scatter_dnn_v1_vs_dnn_v2.png",
        ]
        for f in expected:
            assert (out / f).exists(), f"Missing {f}"

    def test_experiment_with_ref(self, degraded_wav, ref_wav, tmp_path):
        recordings = tmp_path / "recordings"
        (recordings / "dnn_v1").mkdir(parents=True)
        (recordings / "dnn_v1" / "sample.wav").write_bytes(degraded_wav.read_bytes())
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        (ref_dir / "REF_sample.wav").write_bytes(ref_wav.read_bytes())

        exp = Experiment(
            name="denoiser-shootout",
            base_dir=recordings,
            systems=["dnn_v1"],
            ref_dir=ref_dir,
            model="22metric",
            output_dir=tmp_path / "results" / "denoiser-shootout",
        )
        results = exp.run()
        assert len(results) == 1
        assert results[0].model_used == "22metric"
