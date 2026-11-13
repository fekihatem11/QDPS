"""
S4 — QDPS-PF with behavioural kernel only.

Same as S1 but feature_mix = 0: K = K_prob (cosine over log-probability vectors).
No VGG16 features at all — tests whether the behavioural view alone carries
the diversity signal (and gives a cheaper, fully output-only variant).
"""
from qdps.ablation_study.core import qdps_select

METHOD_NAME = "QDPS-PF-behavioral"
CONFIG = {"alpha": None, "reserve_ratio": 0.0, "quality_power": 1.0,
          "feature_mix": 0.0, "kernel": "sum"}


def select(size, index, features, output_probability):
    return qdps_select(size, index, None, output_probability, **CONFIG)
