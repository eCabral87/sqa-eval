from unittest.mock import MagicMock, patch

import pytest

from sqa_eval.engine import InferenceEngine

MOCK_5_SCORES = {
    "mos": 3.2,
    "dnsmos_ovrl": 2.8,
    "scoreq": 3.5,
    "utmos": 3.0,
    "nisqa_mos": 3.1,
}


@patch.object(InferenceEngine, "_load")
class TestInferenceEngine:
    def test_init_loads_model(self, mock_load):
        mock_load.return_value = (MagicMock(), {})
        engine = InferenceEngine("5metric")
        assert engine.model_name == "5metric"

    def test_init_unknown_model_raises(self, mock_load):
        mock_load.return_value = (MagicMock(), {})
        with pytest.raises(ValueError, match="Unknown model"):
            InferenceEngine("bogus_model")

    def test_predict_no_ref(self, mock_load, sample_audio):
        mock_model = MagicMock()
        mock_config = {}
        mock_load.return_value = (mock_model, mock_config)

        with patch.object(InferenceEngine, "_get_infer_single") as mock_get:
            mock_infer = MagicMock(return_value=dict(MOCK_5_SCORES))
            mock_get.return_value = mock_infer

            engine = InferenceEngine("5metric")
            scores = engine.predict(sample_audio)
            assert isinstance(scores, dict)
            assert "mos" in scores

    def test_predict_file_not_found(self, mock_load):
        mock_load.return_value = (MagicMock(), {})
        engine = InferenceEngine("5metric")
        with pytest.raises(FileNotFoundError):
            engine.predict("/nonexistent/audio.wav")

    def test_predict_22metric_no_ref_raises(self, mock_load, tmp_path):
        mock_load.return_value = (MagicMock(), {})
        engine = InferenceEngine("22metric")
        dummy = tmp_path / "dummy.wav"
        dummy.touch()
        with pytest.raises(ValueError):
            engine.predict(dummy)

    def test_predict_with_ref(self, mock_load, sample_audio, sample_ref_audio):
        mock_load.return_value = (MagicMock(), {})
        engine = InferenceEngine("22metric")
        engine._predict_with_ref = MagicMock(return_value=dict(MOCK_5_SCORES))
        scores = engine.predict(sample_audio, sample_ref_audio)
        assert isinstance(scores, dict)
        engine._predict_with_ref.assert_called_once()

    def test_predict_with_ref_file_not_found(self, mock_load, sample_audio):
        mock_load.return_value = (MagicMock(), {})
        engine = InferenceEngine("22metric")
        with pytest.raises(FileNotFoundError, match="Reference"):
            engine.predict(sample_audio, "/nonexistent/ref.wav")

    def test_predict_batch(self, mock_load, sample_audio):
        mock_load.return_value = (MagicMock(), {})

        with patch.object(InferenceEngine, "_get_infer_single") as mock_get:
            mock_infer = MagicMock(return_value=dict(MOCK_5_SCORES))
            mock_get.return_value = mock_infer

            engine = InferenceEngine("5metric")
            results = engine.predict_batch([(sample_audio, None)])
            assert len(results) == 1
            assert "mos" in results[0]

    def test_device_property(self, mock_load):
        mock_load.return_value = (MagicMock(), {})
        engine = InferenceEngine("5metric")
        assert engine.device in ("cpu", "cuda")

    def test_model_name_property(self, mock_load):
        mock_load.return_value = (MagicMock(), {})
        engine = InferenceEngine("5metric")
        assert engine.model_name == "5metric"

    def test_loaded_metrics(self, mock_load):
        mock_model = MagicMock()
        mock_model.metrics = ["mos", "dnsmos_ovrl"]
        mock_load.return_value = (mock_model, {})
        engine = InferenceEngine("5metric")
        assert isinstance(engine.loaded_metrics, list)
