"""
S3 — QDPS-PF + uncertainty reserve.

Same as S1 but restores the tuned budget-adaptive reserve_ratio (0.2 / 0.1 / 0.1)
together with its reserved-similarity penalty (strength 0.5, VGG cosine, max).
Isolates whether the reserve is needed (watch k=100 on MNIST / SVHN).
"""
from qdps.ablation_study.core import get_tuned_defaults, qdps_select

METHOD_NAME = "QDPS-PF+reserve"
CONFIG = {"alpha": None, "reserve_ratio": "adaptive (0.2/0.1/0.1)",
          "penalty_strength": 0.5, "quality_power": 1.0,
          "feature_mix": 0.5, "kernel": "sum"}


def select(size, index, features, output_probability):
    reserve_ratio = get_tuned_defaults(size)["reserve_ratio"]
    return qdps_select(size, index, features, output_probability,
                       alpha=None, reserve_ratio=reserve_ratio, penalty_strength=0.5,
                       quality_power=1.0, feature_mix=0.5, kernel="sum")
