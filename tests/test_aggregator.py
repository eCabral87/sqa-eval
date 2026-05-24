import pytest

from sqa_eval.aggregator import (
    AggregateResult,
    ScoreAggregator,
    compare_systems,
    rank_systems,
)
from sqa_eval.metrics import METRICS_5, METRICS_22, MetricDef


class TestNormalize:
    def test_normalize_bounded(self):
        agg = ScoreAggregator(METRICS_5)
        raw = {"mos": 3.0, "dnsmos_ovrl": 3.0, "scoreq": 3.0, "utmos": 3.0, "nisqa_mos": 3.0}
        norm = agg.normalize(raw)
        for name in raw:
            assert 0.0 <= norm[name] <= 1.0
        assert norm["mos"] == pytest.approx(0.5)

    def test_normalize_clamps(self):
        agg = ScoreAggregator(METRICS_5)
        raw = {"mos": 10.0, "dnsmos_ovrl": -5.0, "scoreq": 3.0, "utmos": 3.0, "nisqa_mos": 3.0}
        norm = agg.normalize(raw)
        assert norm["mos"] == 1.0
        assert norm["dnsmos_ovrl"] == 0.0

    def test_normalize_unbounded_sdr(self):
        agg = ScoreAggregator(METRICS_22)
        assert agg.normalize({"sdr": -30.0})["sdr"] == 0.0
        assert agg.normalize({"sdr": 30.0})["sdr"] == 1.0
        assert 0.0 < agg.normalize({"sdr": 0.0})["sdr"] < 1.0
        assert agg.normalize({"sdr": -100.0})["sdr"] == 0.0
        assert agg.normalize({"sdr": 100.0})["sdr"] == 1.0


class TestCompute:
    def test_compute_all_higher_better(self):
        agg = ScoreAggregator(METRICS_5)
        raw = {"mos": 5.0, "dnsmos_ovrl": 5.0, "scoreq": 5.0, "utmos": 5.0, "nisqa_mos": 5.0}
        assert agg.compute(raw) == 1.0

        raw_low = {"mos": 1.0, "dnsmos_ovrl": 1.0, "scoreq": 1.0, "utmos": 1.0, "nisqa_mos": 1.0}
        assert agg.compute(raw_low) == 0.0

    def test_compute_with_lower_better(self):
        mcd_metric = MetricDef("mcd", 0.0, float("inf"), -1, True)
        m = {"mcd": mcd_metric}
        agg = ScoreAggregator(m)
        high_mcd = agg.compute({"mcd": 0.0})
        low_mcd = agg.compute({"mcd": 20.0})
        assert high_mcd > low_mcd

    def test_compute_with_weights(self):
        agg = ScoreAggregator(METRICS_5, weights={"mos": 10.0})
        raw = {"mos": 5.0, "dnsmos_ovrl": 1.0, "scoreq": 1.0, "utmos": 1.0, "nisqa_mos": 1.0}
        weighted = agg.compute(raw)
        unweighted = ScoreAggregator(METRICS_5).compute(raw)
        assert weighted > unweighted

    def test_compute_common_uses_only_5(self, mock_raw_scores_5, mock_raw_scores_22):
        agg_22 = ScoreAggregator(METRICS_22)
        common_5 = agg_22.compute_common(mock_raw_scores_5)
        common_22 = agg_22.compute_common(mock_raw_scores_22)
        assert common_5 == pytest.approx(common_22)


class TestAggregateResult:
    def test_aggregate_result_fields(self, mock_raw_scores_5):
        agg = ScoreAggregator(METRICS_5)
        result = agg.evaluate("file1.wav", "sys_a", "5metric", mock_raw_scores_5)
        assert result.file_name == "file1.wav"
        assert result.system == "sys_a"
        assert result.model_used == "5metric"
        assert isinstance(result.common_score, float)
        assert isinstance(result.extended_score, float)
        assert result.raw_scores == mock_raw_scores_5


class TestCompareSystems:
    def test_compare_systems(self):
        r1 = AggregateResult("f1.wav", "sys_a", "5metric", 0.8, 0.8, {})
        r2 = AggregateResult("f2.wav", "sys_a", "5metric", 0.6, 0.6, {})
        r3 = AggregateResult("f1.wav", "sys_b", "5metric", 0.7, 0.7, {})
        r4 = AggregateResult("f2.wav", "sys_b", "5metric", 0.5, 0.5, {})

        comp = compare_systems([r1, r2, r3, r4])
        assert "sys_a" in comp
        assert "sys_b" in comp
        assert comp["sys_a"]["common_mean"] == pytest.approx(0.7)

    def test_rank_systems(self):
        r1 = AggregateResult("f1.wav", "sys_a", "5metric", 0.6, 0.6, {})
        r2 = AggregateResult("f1.wav", "sys_b", "5metric", 0.9, 0.9, {})
        r3 = AggregateResult("f1.wav", "sys_c", "5metric", 0.3, 0.3, {})

        ranked = rank_systems([r1, r2, r3])
        assert ranked[0][0] == "sys_b"
        assert ranked[1][0] == "sys_a"
        assert ranked[2][0] == "sys_c"
