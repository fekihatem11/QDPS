"""
S6 — Product kernel + single fixed alpha.

Combines the two strongest ablation results: the product kernel (S5) and
candidate reduction (S2), but with ONE alpha value for every budget instead of
the budget-adaptive 3/4/3. No reserve, q_i = u_i (p = 1).
"""
from qdps.ablation_study.core import qdps_select

METHOD_NAME = "QDPS-PF-product-a3"
CONFIG = {"alpha": 3, "reserve_ratio": 0.0, "quality_power": 1.0,
          "kernel": "product"}


def select(size, index, features, output_probability):
    return qdps_select(size, index, features, output_probability, **CONFIG)
