from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch

TARGET_SR = 16000
GUARD_MS = 50
MAX_DURATION_S = 60.0
FADE_MS = 10


class Preprocessor:
    def __init__(self, use_vad: bool = True):
        self._use_vad = use_vad
        self._vad_model: Any = None
        if use_vad:
            self._load_vad()

    def _load_vad(self):
        import silero_vad

        self._vad_model = silero_vad.load_silero_vad()

    @staticmethod
    def _read_audio(path: str | Path) -> tuple[torch.Tensor, int]:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True)
        data = data.mean(axis=1, keepdims=True).T
        return torch.from_numpy(data).float(), sr

    @staticmethod
    def _resample(audio: torch.Tensor, sr: int, target_sr: int) -> tuple[torch.Tensor, int]:
        if sr == target_sr:
            return audio, sr
        import torchaudio.functional as F

        return F.resample(audio, sr, target_sr), target_sr

    def process_file(self, audio_path: str | Path) -> list[tuple[torch.Tensor, int]]:
        audio, sr = self._read_audio(audio_path)
        audio, sr = self._resample(audio, sr, TARGET_SR)

        if not self._use_vad:
            return [(audio, sr)]

        segments = self._get_speech_segments(audio, sr)
        if not segments or len(segments) <= 1:
            return [(audio, sr)]

        concatenated = self._concatenate_segments(audio, segments, sr)
        return self._split_audio(concatenated, sr)

    def _get_speech_segments(self, audio: torch.Tensor, sr: int) -> list[dict[str, Any]]:
        import silero_vad

        audio_1d = audio.squeeze(0)
        return silero_vad.get_speech_timestamps(
            audio_1d,
            self._vad_model,
            sampling_rate=sr,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=30,
        )

    def _concatenate_segments(
        self, audio: torch.Tensor, segments: list[dict[str, Any]], sr: int
    ) -> torch.Tensor:
        fade_len = int(sr * FADE_MS / 1000)
        guard_len = int(sr * GUARD_MS / 1000)

        non_speech_noise = self._collect_noise_samples(audio, segments)

        chunks: list[torch.Tensor] = []
        for i, seg in enumerate(segments):
            chunk = audio[:, seg["start"] : seg["end"]].clone()
            if chunk.shape[-1] > fade_len * 2:
                chunk = self._fade_in(chunk, fade_len)
                chunk = self._fade_out(chunk, fade_len)
            chunks.append(chunk)

            if i < len(segments) - 1:
                guard = self._get_noise_guard(non_speech_noise, guard_len, fade_len)
                chunks.append(guard)

        if not chunks:
            return audio
        return torch.cat(chunks, dim=-1)

    def _collect_noise_samples(
        self, audio: torch.Tensor, segments: list[dict[str, Any]]
    ) -> torch.Tensor:
        non_speech: list[torch.Tensor] = []
        prev_end = 0
        for seg in segments:
            start = seg["start"]
            if start > prev_end:
                region = audio[:, prev_end:start]
                if region.shape[-1] > 0:
                    non_speech.append(region)
            prev_end = seg["end"]
        if prev_end < audio.shape[-1]:
            region = audio[:, prev_end:]
            if region.shape[-1] > 0:
                non_speech.append(region)

        if non_speech:
            return torch.cat(non_speech, dim=-1)
        return torch.zeros((1, 1), dtype=audio.dtype)

    def _get_noise_guard(
        self, noise_pool: torch.Tensor, guard_len: int, fade_len: int
    ) -> torch.Tensor:
        if noise_pool.shape[-1] > guard_len:
            start = random.randint(0, noise_pool.shape[-1] - guard_len)
            guard = noise_pool[:, start : start + guard_len].clone()
        else:
            guard = noise_pool.clone()
            if guard.shape[-1] < guard_len:
                repeats = (guard_len // guard.shape[-1]) + 1
                guard = guard.repeat(1, repeats)[:, :guard_len]

        guard = self._fade_in(guard, min(fade_len, guard.shape[-1] // 2))
        guard = self._fade_out(guard, min(fade_len, guard.shape[-1] // 2))
        return guard

    @staticmethod
    def _fade_in(audio: torch.Tensor, fade_len: int) -> torch.Tensor:
        if fade_len <= 0 or audio.shape[-1] < fade_len * 2:
            return audio
        window = torch.cos(torch.linspace(math.pi / 2, 0, fade_len)) ** 2
        window = window.to(audio.device, audio.dtype)
        result = audio.clone()
        result[:, :fade_len] *= window
        return result

    @staticmethod
    def _fade_out(audio: torch.Tensor, fade_len: int) -> torch.Tensor:
        if fade_len <= 0 or audio.shape[-1] < fade_len * 2:
            return audio
        window = torch.cos(torch.linspace(0, math.pi / 2, fade_len)) ** 2
        window = window.to(audio.device, audio.dtype)
        result = audio.clone()
        result[:, -fade_len:] *= window
        return result

    @staticmethod
    def _split_audio(audio: torch.Tensor, sr: int) -> list[tuple[torch.Tensor, int]]:
        max_samples = int(MAX_DURATION_S * sr)
        if audio.shape[-1] <= max_samples:
            return [(audio, sr)]

        chunks: list[tuple[torch.Tensor, int]] = []
        offset = 0
        while offset < audio.shape[-1]:
            chunk = audio[:, offset : offset + max_samples]
            chunks.append((chunk, sr))
            offset += max_samples
        return chunks
