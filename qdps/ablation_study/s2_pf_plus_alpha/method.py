"""
S2 — QDPS-PF + candidate reduction.

Same as S1 but restores the tuned budget-adaptive alpha (3 / 4 / 3).
Isolates whether truncating the pool to the top alpha*k uncertain inputs is needed.
"""
from qdps.ablation_study.core import get_tuned_defaults, qdps_select

METHOD_NAME = "QDPS-PF+alpha"
CONFIG = {"alpha": "adaptive (3/4/3)", "reserve_ratio": 0.0, "quality_power": 1.0,
          "feature_mix": 0.5, "kernel": "sum"}


def select(size, index, features, output_probability):
    alpha = get_tuned_defaults(size)["alpha"]
    return qdps_select(size, index, features, output_probability,
                       alpha=alpha, reserve_ratio=0.0, quality_power=1.0,
                       feature_mix=0.5, kernel="sum")
