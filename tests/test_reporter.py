import json

import pytest

from sqa_eval.aggregator import ScoreAggregator
from sqa_eval.metrics import METRICS_5
from sqa_eval.reporter import Reporter


@pytest.fixture
def sample_results(mock_raw_scores_5):
    agg = ScoreAggregator(METRICS_5)
    sys_a_2 = {
        "mos": 3.0,
        "dnsmos_ovrl": 2.5,
        "scoreq": 3.0,
        "utmos": 2.8,
        "nisqa_mos": 3.0,
    }
    sys_b_1 = {
        "mos": 3.5,
        "dnsmos_ovrl": 3.0,
        "scoreq": 3.8,
        "utmos": 3.2,
        "nisqa_mos": 3.4,
    }
    sys_b_2 = {
        "mos": 3.8,
        "dnsmos_ovrl": 3.2,
        "scoreq": 3.9,
        "utmos": 3.5,
        "nisqa_mos": 3.7,
    }
    return [
        agg.evaluate("file1.wav", "sys_a", "5metric", mock_raw_scores_5),
        agg.evaluate("file2.wav", "sys_a", "5metric", sys_a_2),
        agg.evaluate("file1.wav", "sys_b", "5metric", sys_b_1),
        agg.evaluate("file2.wav", "sys_b", "5metric", sys_b_2),
    ]


class TestReporter:
    def test_to_dataframe_shape(self, sample_results):
        reporter = Reporter(sample_results)
        df = reporter.to_dataframe()
        assert df.shape[0] == 4
        assert "mos" in df.columns
        assert "common_score" in df.columns

    def test_to_csv_writes_file(self, sample_results, tmp_path):
        reporter = Reporter(sample_results)
        path = tmp_path / "scores.csv"
        reporter.to_csv(path)
        assert path.exists()
        content = path.read_text()
        assert "file_name" in content

    def test_summary_has_expected_stats(self, sample_results):
        reporter = Reporter(sample_results)
        df = reporter.summary_table()
        for col in ["common_mean", "common_std", "common_min", "common_max", "count"]:
            assert col in df.columns

    def test_ranking_order(self, sample_results):
        reporter = Reporter(sample_results)
        df = reporter.ranking_table()
        assert df.iloc[0]["system"] == "sys_b"

    def test_to_json_is_valid(self, sample_results, tmp_path):
        reporter = Reporter(sample_results)
        path = tmp_path / "results.json"
        reporter.to_json(path)
        data = json.loads(path.read_text())
        assert "per_file" in data
        assert "aggregates" in data
        assert "ranking" in data

    def test_merge_reports(self, sample_results):
        r1 = Reporter(sample_results[:2])
        r2 = Reporter(sample_results[2:])
        merged = Reporter.merge_reports([r1, r2])
        assert len(merged.results) == 4
