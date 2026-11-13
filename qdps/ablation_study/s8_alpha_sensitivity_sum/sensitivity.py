"""
S8 — Alpha sensitivity of QDPS-PF with Gram (sum) aggregation.

Same sweep as S7 but the two Gram matrices are added instead of multiplied:
K = 0.5*K_vgg + 0.5*K_prob, no reserve, q_i = u_i (p = 1).
Compare with S7 (product kernel) to choose the kernel aggregation.

Usage (from the repo root):
    python -m qdps.ablation_study.s8_alpha_sensitivity_sum.sensitivity [n_runs] [alphas]
"""
import os

from qdps.ablation_study.s7_alpha_sensitivity_product.sensitivity import main

METHOD_NAME = "QDPS-PF-sum"
FIXED = {"reserve_ratio": 0.0, "quality_power": 1.0, "kernel": "sum", "feature_mix": 0.5}
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    main(STUDY_DIR, "s8_alpha_sensitivity_sum", METHOD_NAME, FIXED,
         "sum kernel, no reserve, p = 1")
