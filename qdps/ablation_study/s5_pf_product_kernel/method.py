"""
S5 — QDPS-PF with product kernel.

Same as S1 but K = K_vgg * K_prob (elementwise / Schur product, still PSD).
Parameter-free alternative to the 0.5 mix: two inputs are redundant only if
they are similar both visually and behaviourally.
"""
from qdps.ablation_study.core import qdps_select

METHOD_NAME = "QDPS-PF-product"
CONFIG = {"alpha": None, "reserve_ratio": 0.0, "quality_power": 1.0,
          "kernel": "product"}


def select(size, index, features, output_probability):
    return qdps_select(size, index, features, output_probability, **CONFIG)
