"""
S1 — Parameter-free QDPS (QDPS-PF).

- alpha:          dropped  -> candidate pool = all valid test inputs
- reserve_ratio:  dropped  -> 0 (no reserve, hence no reserve penalty either)
- quality_power:  fixed at 1 (standard quality/diversity L-ensemble, q_i = u_i)
- feature_mix:    fixed at 0.5 (== cosine over concatenated unit-normalised
                  [VGG || log-prob] features, i.e. one kernel, equal view weight)
"""
from qdps.ablation_study.core import qdps_select

METHOD_NAME = "QDPS-PF"
CONFIG = {"alpha": None, "reserve_ratio": 0.0, "quality_power": 1.0,
          "feature_mix": 0.5, "kernel": "sum"}


def select(size, index, features, output_probability):
    return qdps_select(size, index, features, output_probability, **CONFIG)
