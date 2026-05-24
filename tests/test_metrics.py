from sqa_eval.metrics import (
    COMMON_METRICS,
    METRICS_5,
    METRICS_22,
    MODEL_ALIASES,
)


def test_metrics_5_count():
    assert len(METRICS_5) == 5


def test_metrics_22_count():
    assert len(METRICS_22) == 22


def test_common_metrics_subset_of_both():
    for name in COMMON_METRICS:
        assert name in METRICS_5
        assert name in METRICS_22


def test_metric_ranges():
    for metrics in (METRICS_5, METRICS_22):
        for m in metrics.values():
            if m.min_val != float("-inf") and m.max_val != float("inf"):
                assert m.min_val < m.max_val, f"{m.name}: min >= max"
            assert m.direction in (+1, -1), f"{m.name}: bad direction"


def test_ref_required_flags():
    for name, m in METRICS_5.items():
        assert not m.ref_required, f"5metric metric {name} should not require ref"

    for name in COMMON_METRICS:
        assert not METRICS_22[name].ref_required, f"Common metric {name} should not require ref"

    ref_required_22 = sum(1 for m in METRICS_22.values() if m.ref_required)
    assert ref_required_22 > 0, "Expected at least some 22metric metrics to require ref"


def test_model_aliases():
    assert MODEL_ALIASES["5metric"].startswith("vvwangvv/universa-ext_wavlm-base_5metric")
    assert MODEL_ALIASES["22metric"].startswith("vvwangvv/universa-ext_wavlm-base_22metric")
