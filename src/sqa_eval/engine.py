from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import torch

from sqa_eval.metrics import MODEL_ALIASES

TARGET_SR = 16000


class InferenceEngine:
    def __init__(self, model: str = "5metric"):
        """Load a Uni-VERSA-Ext model for inference.

        Parameters
        ----------
        model : str, default "5metric"
            Model alias (``"5metric"`` or ``"22metric"``) or a full
            HuggingFace repo ID (e.g. ``"org/custom-model"``). Auto-detects
            CUDA; falls back to CPU.
        """
        self._resolve_model(model)
        self._device = self._detect_device()
        self._model, self._config = self._load()

    def _resolve_model(self, model: str) -> None:
        if model in MODEL_ALIASES:
            self._repo_id = MODEL_ALIASES[model]
            self._model_name = model
        elif "/" in model or "\\" in model:
            self._repo_id = model
            self._model_name = model
        else:
            raise ValueError(
                f"Unknown model '{model}'. Use one of {list(MODEL_ALIASES)} "
                "or a full HuggingFace repo ID."
            )

    def _detect_device(self) -> str:
        try:
            import torch
        except Exception:
            return "cpu"

        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> tuple[Any, Any]:
        try:
            from urgent2026_sqa.infer import load_model
        except Exception as e:
            raise ImportError(
                "urgent2026_sqa is not available. "
                "Install it with: pip install git+https://github.com/urgent-challenge/urgent2026_challenge_track2\n"
                f"Underlying error: {e}"
            ) from e

        if self._device == "cuda" and not self._has_cuda():
            warnings.warn("GPU requested but CUDA unavailable. Falling back to CPU.")
            self._device = "cpu"

        model, config = load_model(self._repo_id)
        return model, config

    def _has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _needs_ref(self) -> bool:
        if self._model_name == "22metric":
            return True
        if self._model_name not in MODEL_ALIASES:
            return True
        return False

    @staticmethod
    def _read_audio(path: str | Path) -> tuple[Any, int]:
        import soundfile as sf
        import torch

        data, sr = sf.read(str(path), always_2d=True)
        data = data.mean(axis=1, keepdims=True).T
        return torch.from_numpy(data).float(), sr

    @staticmethod
    def _resample(audio: Any, sr: int, target_sr: int) -> tuple[Any, int]:
        if sr == target_sr:
            return audio, sr
        import torchaudio.functional as F

        return F.resample(audio, sr, target_sr), target_sr

    def _get_infer_single(self):
        from urgent2026_sqa.infer import infer_single

        return infer_single

    def predict_from_tensor(
        self,
        audio: torch.Tensor,
        sr: int,
        ref_audio: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run inference on a pre-loaded audio tensor.

        Parameters
        ----------
        audio : torch.Tensor
            Audio tensor of shape ``(1, samples)`` at the model's target
            sample rate (``TARGET_SR``).
        sr : int
            Sample rate of ``audio``.
        ref_audio : torch.Tensor | None, optional
            Reference audio tensor of shape ``(1, samples)``. Required when
            the model needs reference-based metrics.

        Returns
        -------
        dict[str, float]
            Raw per-metric scores.
        """
        if audio.shape[-1] == 0:
            raise ValueError("Empty audio tensor")

        if ref_audio is not None:
            if ref_audio.shape[-1] == 0:
                raise ValueError("Empty reference audio tensor")
            min_len = min(audio.shape[-1], ref_audio.shape[-1])
            audio = audio[..., :min_len]
            ref_audio = ref_audio[..., :min_len]
            combined = torch.cat([audio, ref_audio], dim=0)
            infer_single = self._get_infer_single()
            scores = infer_single(self._model, self._config, combined, audio_sr=sr)
        else:
            infer_single = self._get_infer_single()
            scores = infer_single(self._model, self._config, audio, audio_sr=sr)

        return {k: float(v) for k, v in scores.items()}

    def predict(
        self, audio_path: str | Path, ref_path: str | Path | None = None
    ) -> dict[str, float]:
        """Run inference on a single audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to the degraded / test audio file.
        ref_path : str | Path | None, optional
            Path to a clean reference file. Required when the model
            needs reference-based metrics (``"22metric"`` or custom).

        Returns
        -------
        dict[str, float]
            Raw per-metric scores keyed by metric name (e.g.
            ``{"mos": 3.2, "sdr": 12.5, ...}``).

        Raises
        ------
        FileNotFoundError
            If ``audio_path`` or ``ref_path`` doesn't exist.
        ValueError
            If the model requires a reference but none was given.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if ref_path is not None:
            ref_path = Path(ref_path)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference file not found: {ref_path}")
            return self._predict_with_ref(str(audio_path), str(ref_path))

        if self._needs_ref():
            raise ValueError(
                f"Model '{self._model_name}' requires reference audio for some metrics. "
                "Pass a ref_path or use '5metric' for no-reference evaluation."
            )

        audio, sr = self._read_audio(audio_path)
        audio, sr = self._resample(audio, sr, TARGET_SR)

        infer_single = self._get_infer_single()
        scores = infer_single(self._model, self._config, audio, audio_sr=sr)
        return {k: float(v) for k, v in scores.items()}

    def _predict_with_ref(self, audio_path: str, ref_path: str) -> dict[str, float]:
        import torch

        test_audio, test_sr = self._read_audio(audio_path)
        ref_audio, ref_sr = self._read_audio(ref_path)

        test_audio, test_sr = self._resample(test_audio, test_sr, TARGET_SR)
        ref_audio, _ = self._resample(ref_audio, ref_sr, TARGET_SR)

        min_len = min(test_audio.shape[-1], ref_audio.shape[-1])
        test_audio = test_audio[..., :min_len]
        ref_audio = ref_audio[..., :min_len]

        combined = torch.cat([test_audio, ref_audio], dim=0)
        infer_single = self._get_infer_single()
        scores = infer_single(self._model, self._config, combined, audio_sr=TARGET_SR)
        return {k: float(v) for k, v in scores.items()}

    def predict_batch(
        self, pairs: list[tuple[str | Path, str | Path | None]]
    ) -> list[dict[str, float]]:
        """Run inference on multiple audio files.

        Parameters
        ----------
        pairs : list[tuple[str | Path, str | Path | None]]
            List of ``(audio_path, ref_path_or_None)`` tuples.

        Returns
        -------
        list[dict[str, float]]
            Raw per-metric scores, one dict per input pair.
        """
        results: list[dict[str, float]] = []

        for audio_path, ref_path in pairs:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            if ref_path is not None:
                ref_path = Path(ref_path)
                if not ref_path.exists():
                    raise FileNotFoundError(f"Reference file not found: {ref_path}")
                results.append(self._predict_with_ref(str(audio_path), str(ref_path)))
            else:
                if self._needs_ref():
                    raise ValueError(
                        f"Model '{self._model_name}' requires reference audio for some metrics. "
                        "Pass a ref_path or use '5metric' for no-reference evaluation."
                    )
                audio, sr = self._read_audio(audio_path)
                audio, sr = self._resample(audio, sr, TARGET_SR)
                infer_single = self._get_infer_single()
                scores = infer_single(self._model, self._config, audio, audio_sr=sr)
                results.append({k: float(v) for k, v in scores.items()})

        return results

    @property
    def device(self) -> str:
        """Device the model is running on: ``"cuda"`` or ``"cpu"``."""
        return self._device

    @property
    def model_name(self) -> str:
        """Model alias or HF repo ID that was passed at init."""
        return self._model_name

    @property
    def loaded_metrics(self) -> list[str]:
        return list(self._model.metrics) if hasattr(self._model, "metrics") else []
