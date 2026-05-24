from dataclasses import dataclass


@dataclass
class MetricDef:
    name: str
    min_val: float
    max_val: float
    direction: int
    ref_required: bool
    weight: float = 1.0


METRICS_5: dict[str, MetricDef] = {
    "mos": MetricDef("mos", 1.0, 5.0, +1, False),
    "dnsmos_ovrl": MetricDef("dnsmos_ovrl", 1.0, 5.0, +1, False),
    "scoreq": MetricDef("scoreq", 1.0, 5.0, +1, False),
    "utmos": MetricDef("utmos", 1.0, 5.0, +1, False),
    "nisqa_mos": MetricDef("nisqa_mos", 1.0, 5.0, +1, False),
}

METRICS_22: dict[str, MetricDef] = {
    "mos": MetricDef("mos", 1.0, 5.0, +1, False),
    "dnsmos_ovrl": MetricDef("dnsmos_ovrl", 1.0, 5.0, +1, False),
    "scoreq": MetricDef("scoreq", 1.0, 5.0, +1, False),
    "utmos": MetricDef("utmos", 1.0, 5.0, +1, False),
    "nisqa_mos": MetricDef("nisqa_mos", 1.0, 5.0, +1, False),
    "distill_mos": MetricDef("distill_mos", 1.0, 5.0, +1, False),
    "sigmos_ovrl": MetricDef("sigmos_ovrl", 1.0, 5.0, +1, False),
    "sigmos_col": MetricDef("sigmos_col", 1.0, 5.0, +1, False),
    "sigmos_disc": MetricDef("sigmos_disc", 1.0, 5.0, +1, False),
    "sigmos_loud": MetricDef("sigmos_loud", 1.0, 5.0, +1, False),
    "sigmos_reverb": MetricDef("sigmos_reverb", 1.0, 5.0, +1, False),
    "sigmos_sig": MetricDef("sigmos_sig", 1.0, 5.0, +1, False),
    "sigmos_noise": MetricDef("sigmos_noise", 1.0, 5.0, +1, False),
    "spksim": MetricDef("spksim", -1.0, 1.0, +1, True),
    "sbert": MetricDef("sbert", 0.0, 1.0, +1, True),
    "mcd": MetricDef("mcd", 0.0, float("inf"), -1, True),
    "sdr": MetricDef("sdr", float("-inf"), float("inf"), +1, True),
    "pesq": MetricDef("pesq", 1.0, 4.5, +1, True),
    "pesqc2": MetricDef("pesqc2", 1.0, 4.5, +1, True),
    "lsd": MetricDef("lsd", 0.0, float("inf"), -1, True),
    "estoi": MetricDef("estoi", 0.0, 1.0, +1, True),
    "lps": MetricDef("lps", 0.0, 1.0, +1, True),
}

COMMON_METRICS: list[str] = ["mos", "dnsmos_ovrl", "scoreq", "utmos", "nisqa_mos"]

MODEL_ALIASES: dict[str, str] = {
    "5metric": "vvwangvv/universa-ext_wavlm-base_5metric",
    "22metric": "vvwangvv/universa-ext_wavlm-base_22metric",
}
