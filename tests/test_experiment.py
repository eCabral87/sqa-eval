from unittest.mock import MagicMock, patch

import pytest

from sqa_eval.experiment import Evaluator, Experiment

MOCK_5_SCORES = {
    "mos": 3.2,
    "dnsmos_ovrl": 2.8,
    "scoreq": 3.5,
    "utmos": 3.0,
    "nisqa_mos": 3.1,
}

MOCK_22_SCORES = {**MOCK_5_SCORES, "sdr": 12.5, "pesq": 2.8}


@pytest.fixture
def mock_engine(mock_raw_scores_5):
    with patch("sqa_eval.experiment.InferenceEngine") as mock_cls:
        instance = MagicMock()
        instance.predict.return_value = mock_raw_scores_5
        instance.model_name = "5metric"
        mock_cls.return_value = instance
        yield mock_cls


class TestEvaluator:
    def test_evaluator_file(self, mock_engine, mock_raw_scores_5, sample_audio):
        evaluator = Evaluator("5metric")
        result = evaluator.evaluate_file(sample_audio)
        assert isinstance(result, type(result))
        assert result.file_name == sample_audio.name
        assert result.system == "default"
        assert result.model_used == "5metric"

    def test_evaluator_directory(self, mock_engine, mock_raw_scores_5, experiment_layout):
        evaluator = Evaluator("5metric")
        results = evaluator.evaluate_directory(experiment_layout / "sys_a")
        assert len(results) == 2


class TestExperiment:
    def test_experiment_run_scans_systems(self, mock_engine, mock_raw_scores_5, experiment_layout):
        exp = Experiment("test_exp", experiment_layout, ["sys_a", "sys_b"], model="5metric")
        results = exp.run()
        assert len(results) == 4
        systems = {r.system for r in results}
        assert "sys_a" in systems
        assert "sys_b" in systems

    def test_experiment_report_creates_files(
        self, mock_engine, mock_raw_scores_5, experiment_layout, tmp_path
    ):
        output_dir = tmp_path / "results" / "test_exp"
        exp = Experiment(
            "test_exp",
            experiment_layout,
            ["sys_a", "sys_b"],
            model="5metric",
            output_dir=output_dir,
        )
        exp.run()
        exp.report()
        assert (output_dir / "scores.csv").exists()
        assert (output_dir / "summary.csv").exists()
        assert (output_dir / "results.json").exists()

    def test_experiment_missing_system_raises(self, mock_engine, experiment_layout):
        exp = Experiment("test_exp", experiment_layout, ["no_such_system"], model="5metric")
        with pytest.raises(ValueError, match="does not exist"):
            exp.run()

    def test_experiment_ref_matching(self, experiment_layout):
        mock_22_scores = {
            "mos": 3.5,
            "dnsmos_ovrl": 3.0,
            "scoreq": 3.8,
            "utmos": 3.2,
            "nisqa_mos": 3.4,
            "sdr": 12.5,
            "pesq": 2.8,
        }

        with patch("sqa_eval.experiment.InferenceEngine") as mock_cls:
            instance = MagicMock()
            instance.predict.return_value = mock_22_scores
            instance.model_name = "22metric"
            mock_cls.return_value = instance

            exp = Experiment(
                "test_exp",
                experiment_layout,
                ["sys_a"],
                ref_dir=experiment_layout / "ref",
                model="22metric",
            )
            results = exp.run()
            assert len(results) == 2
            for call in instance.predict.call_args_list:
                args, kwargs = call
                assert len(args) >= 2
