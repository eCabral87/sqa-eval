from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from sqa_eval.preprocess import (
    FADE_MS,
    GUARD_MS,
    MAX_DURATION_S,
    TARGET_SR,
    Preprocessor,
)

SP03_PATH = Path(__file__).resolve().parent.parent / "sp03.wav"


@pytest.fixture(scope="session")
def sp03_duration() -> float:
    data, sr = sf.read(str(SP03_PATH))
    return len(data) / sr


class TestPreprocessorUnit:
    def test_init_no_vad(self):
        p = Preprocessor(use_vad=False)
        assert p._vad_model is None

    def test_init_with_vad(self):
        p = Preprocessor(use_vad=True)
        assert p._vad_model is not None

    def test_fade_in(self):
        audio = torch.ones(1, 100)
        faded = Preprocessor._fade_in(audio, 20)
        assert faded.shape == audio.shape
        assert faded[0, 0].item() < 0.01
        assert faded[0, -1].item() == 1.0

    def test_fade_out(self):
        audio = torch.ones(1, 100)
        faded = Preprocessor._fade_out(audio, 20)
        assert faded.shape == audio.shape
        assert faded[0, -1].item() < 0.01
        assert faded[0, 0].item() == 1.0

    def test_fade_too_short(self):
        audio = torch.ones(1, 5)
        result = Preprocessor._fade_in(audio, 20)
        assert result is audio

    def test_split_audio_under_limit(self):
        sr = TARGET_SR
        audio = torch.randn(1, int(30 * sr))
        chunks = Preprocessor._split_audio(audio, sr)
        assert len(chunks) == 1

    def test_split_audio_over_limit(self):
        sr = TARGET_SR
        audio = torch.randn(1, int(130 * sr))
        chunks = Preprocessor._split_audio(audio, sr)
        assert len(chunks) == 3
        for chunk, _ in chunks:
            assert chunk.shape[-1] <= int(MAX_DURATION_S * sr)

    def test_collect_noise_samples(self):
        audio = torch.randn(1, 1000)
        segments = [{"start": 100, "end": 300}, {"start": 500, "end": 700}]
        noise = Preprocessor()._collect_noise_samples(audio, segments)
        assert noise.shape[-1] == 600

    def test_collect_noise_no_gaps(self):
        audio = torch.randn(1, 1000)
        segments = [{"start": 0, "end": 1000}]
        noise = Preprocessor()._collect_noise_samples(audio, segments)
        assert noise.shape[-1] == 1

    def test_get_noise_guard_shorter_than_pool(self):
        noise_pool = torch.randn(1, 1000)
        guard = Preprocessor()._get_noise_guard(noise_pool, 50, 10)
        assert guard.shape[-1] == 50
        assert guard[0, 0].abs().item() < 0.1

    def test_get_noise_guard_longer_than_pool(self):
        noise_pool = torch.randn(1, 10)
        guard = Preprocessor()._get_noise_guard(noise_pool, 50, 5)
        assert guard.shape[-1] == 50

    def test_average_scores_single(self):
        from sqa_eval.experiment import Evaluator

        scores = Evaluator._average_scores([{"mos": 3.0, "sdr": 12.0}])
        assert scores["mos"] == 3.0
        assert scores["sdr"] == 12.0

    def test_average_scores_multiple(self):
        from sqa_eval.experiment import Evaluator

        scores = Evaluator._average_scores([{"mos": 3.0}, {"mos": 4.0}])
        assert scores["mos"] == 3.5

    def test_average_scores_missing_keys(self):
        from sqa_eval.experiment import Evaluator

        scores = Evaluator._average_scores([{"mos": 3.0}, {"mos": 4.0, "sdr": 10.0}])
        assert scores["mos"] == 3.5
        assert scores["sdr"] == 10.0

    def test_average_scores_empty(self):
        from sqa_eval.experiment import Evaluator

        assert Evaluator._average_scores([]) == {}

    def test_concatenate_segments_multiple(self):
        sr = TARGET_SR
        audio_a = torch.ones(1, sr)
        audio_b = torch.ones(1, sr)
        full_audio = torch.cat([audio_a, audio_b], dim=-1)
        segments = [{"start": 0, "end": sr}, {"start": sr, "end": 2 * sr}]

        p = Preprocessor(use_vad=False)
        concat = p._concatenate_segments(full_audio, segments, sr)
        guard_len = int(sr * GUARD_MS / 1000)
        expected_len = 2 * sr + guard_len
        assert concat.shape[-1] == expected_len
        assert concat[0, sr + guard_len - 1].abs().item() >= 0

    def test_concatenate_segments_single(self):
        sr = TARGET_SR
        audio = torch.ones(1, sr)
        segments = [{"start": 0, "end": sr}]
        p = Preprocessor(use_vad=False)
        concat = p._concatenate_segments(audio, segments, sr)
        assert concat.shape[-1] == sr


class TestPreprocessorNoVAD:
    def test_process_file_no_vad(self, tmp_path):
        path = tmp_path / "test.wav"
        sf.write(str(path), np.random.randn(TARGET_SR), TARGET_SR)
        p = Preprocessor(use_vad=False)
        chunks = p.process_file(path)
        assert len(chunks) == 1
        chunk, sr = chunks[0]
        assert sr == TARGET_SR
        assert chunk.shape[0] == 1


class TestPreprocessorWithRealSpeech:
    def test_process_file_short_speech(self):
        p = Preprocessor(use_vad=True)
        chunks = p.process_file(SP03_PATH)
        assert len(chunks) == 1
        chunk, sr = chunks[0]
        assert sr == TARGET_SR
        assert chunk.shape[0] == 1
        assert chunk.shape[-1] > 0

    def test_process_file_removes_silence(self):
        p = Preprocessor(use_vad=True)
        chunks = p.process_file(SP03_PATH)

        original, _ = p._read_audio(SP03_PATH)
        original_dur = original.shape[-1] / TARGET_SR

        concat_dur = sum(c.shape[-1] for c, _ in chunks) / TARGET_SR
        assert concat_dur <= original_dur

    def test_long_file_splits_into_chunks(self, tmp_path):
        data, sr = sf.read(str(SP03_PATH))
        data_t = torch.from_numpy(data).float().unsqueeze(0)
        silence = torch.zeros(1, int(0.5 * sr))
        long_parts = []
        for _ in range(20):
            long_parts.append(data_t)
            long_parts.append(silence)
        long_audio = torch.cat(long_parts, dim=-1).squeeze(0).numpy()
        long_path = tmp_path / "long_test.wav"
        sf.write(str(long_path), long_audio, sr)

        p = Preprocessor(use_vad=True)
        chunks = p.process_file(long_path)

        for chunk, chunk_sr in chunks:
            assert chunk_sr == TARGET_SR
            assert chunk.shape[-1] <= int(MAX_DURATION_S * TARGET_SR)

    def test_guard_sections_nonzero(self):
        p = Preprocessor(use_vad=True)
        audio, sr = p._read_audio(SP03_PATH)
        audio, sr = p._resample(audio, sr, TARGET_SR)
        segments = p._get_speech_segments(audio, sr)

        if len(segments) > 1:
            concat = p._concatenate_segments(audio, segments, sr)
            speech_only = sum((s["end"] - s["start"]) for s in segments)
            guard_total = concat.shape[-1] - speech_only
            expected_guard = (len(segments) - 1) * int(GUARD_MS / 1000 * sr)
            assert abs(guard_total - expected_guard) <= sr * GUARD_MS // 1000

    def test_process_file_empty_audio(self, tmp_path):
        path = tmp_path / "empty.wav"
        sf.write(str(path), np.zeros(100), TARGET_SR)
        p = Preprocessor(use_vad=True)
        chunks = p.process_file(path)
        assert len(chunks) == 1


class TestPreprocessorFadeTransitions:
    def test_fade_boundary_smooth(self):
        sr = TARGET_SR
        fade_len = int(sr * FADE_MS / 1000)

        audio_a = torch.ones(1, sr)
        audio_b = torch.ones(1, sr)
        full_audio = torch.cat([audio_a, audio_b], dim=-1)
        segments = [{"start": 0, "end": sr}, {"start": sr, "end": 2 * sr}]

        p = Preprocessor(use_vad=False)
        concat = p._concatenate_segments(full_audio, segments, sr)
        guard_len = int(sr * GUARD_MS / 1000)

        assert concat.shape[-1] == 2 * sr + guard_len

        val_a = concat[0, sr - fade_len // 2].abs().item()
        val_b = concat[0, sr + guard_len + fade_len // 2].abs().item()
        assert val_a > 0
        assert val_b > 0
