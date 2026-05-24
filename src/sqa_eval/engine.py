from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from sqa_eval.metrics import MODEL_ALIASES


class InferenceEngine:
    def __init__(self, model: str = "5metric"):
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

    def _get_infer_single(self):
        from urgent2026_sqa.infer import infer_single

        return infer_single

    def _get_infer_list(self):
        from urgent2026_sqa.infer import infer_list

        return infer_list

    def predict(
        self, audio_path: str | Path, ref_path: str | Path | None = None
    ) -> dict[str, float]:
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

        infer_single = self._get_infer_single()
        scores = infer_single(self._model, self._config, str(audio_path))
        return {k: float(v) for k, v in scores.items()}

    def _predict_with_ref(self, audio_path: str, ref_path: str) -> dict[str, float]:
        import torch
        import torchaudio

        test_audio, test_sr = torchaudio.load(audio_path)
        ref_audio, ref_sr = torchaudio.load(ref_path)

        target_sr = 16000
        if test_sr != target_sr:
            test_audio = torchaudio.functional.resample(test_audio, test_sr, target_sr)
        if ref_sr != target_sr:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, target_sr)

        min_len = min(test_audio.shape[-1], ref_audio.shape[-1])
        test_audio = test_audio[..., :min_len]
        ref_audio = ref_audio[..., :min_len]

        combined = torch.cat([test_audio, ref_audio], dim=0)
        infer_single = self._get_infer_single()
        scores = infer_single(self._model, self._config, combined, audio_sr=target_sr)
        return {k: float(v) for k, v in scores.items()}

    def predict_batch(
        self, pairs: list[tuple[str | Path, str | Path | None]]
    ) -> list[dict[str, float]]:
        no_ref_paths: list[str] = []
        no_ref_indices: list[int] = []
        ref_results: list[dict[str, float] | None] = [None] * len(pairs)

        for i, (audio_path, ref_path) in enumerate(pairs):
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            if ref_path is not None:
                ref_path = Path(ref_path)
                if not ref_path.exists():
                    raise FileNotFoundError(f"Reference file not found: {ref_path}")
                ref_results[i] = self._predict_with_ref(str(audio_path), str(ref_path))
            else:
                no_ref_paths.append(str(audio_path))
                no_ref_indices.append(i)

        if no_ref_paths:
            infer_list = self._get_infer_list()
            no_ref_scores = infer_list(self._model, self._config, no_ref_paths)
            for idx, scores in zip(no_ref_indices, no_ref_scores):
                ref_results[idx] = {k: float(v) for k, v in scores.items()}

        return ref_results

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded_metrics(self) -> list[str]:
        return list(self._model.metrics) if hasattr(self._model, "metrics") else []
