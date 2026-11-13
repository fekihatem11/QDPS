"""
S0 — Tuned QDPS (reference).

The original QDPS before the ablation study: budget-adaptive alpha /
reserve_ratio / feature_mix / quality_power (core.TUNED_DEFAULTS), reserve
penalty 0.5, sum kernel. Run through the ablation engine, which reproduces the
original qdps.select selections exactly.
"""
from qdps.ablation_study.core import TUNED_DEFAULTS, get_tuned_defaults, qdps_select

METHOD_NAME = "QDPS-tuned"
CONFIG = {"source": "core.qdps_select", "tuned_defaults": TUNED_DEFAULTS}


def select(size, index, features, output_probability):
    return qdps_select(size, index, features, output_probability, **get_tuned_defaults(size))
