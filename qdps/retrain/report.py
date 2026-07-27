"""Aggregation, Wilcoxon test, and result writers for the retraining experiment.

Mirrors the JSON/summary pattern of ``qdps/experiments/run_experiment.py`` and the
Wilcoxon usage of ``qdps/analysis/statistical_test.py``.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from qdps.io.paths import RETRAIN_SUBJECTS_DIR


def aggregate(acc_res, improvements):
    """Per-method aggregate over the runs."""
    runs = [
        {"acc_re": float(a), "improvement": float(i)}
        for a, i in zip(acc_res, improvements)
    ]
    return {
        "selected_size": None,  # filled in by the runner
        "runs": runs,
        "mean_improvement": float(np.mean(improvements)),
        "std_improvement": float(np.std(improvements)),
        "mean_acc_re": float(np.mean(acc_res)),
    }


def wilcoxon_qdps_vs_sets(qdps_imps, sets_imps):
    """Two-sided Wilcoxon signed-rank on the paired improvement lists (alpha=0.05)."""
    try:
        _stat, p = wilcoxon(qdps_imps, sets_imps, alternative="two-sided")
        p = float(p)
    except ValueError:
        # raised when all paired differences are zero
        p = float("nan")
    mean_delta = float(np.mean(qdps_imps) - np.mean(sets_imps))
    winner = "QDPS" if mean_delta > 0 else ("SETS" if mean_delta < 0 else "tie")
    significant = bool(p < 0.05) if p == p else False  # p==p guards NaN
    return {
        "wilcoxon_p": p,
        "significant": significant,
        "winner": winner,
        "mean_delta_improvement": mean_delta,
    }


def save_subject_result(subject_key, subject_result, meta):
    """Persist one subject's result to the durable accumulating store.

    Overwrites ``RETRAIN_SUBJECTS_DIR/<subject>.json`` so re-running a subject
    refreshes it; other subjects' files are untouched. The comparison table is
    built from whatever subject files exist.
    """
    os.makedirs(RETRAIN_SUBJECTS_DIR, exist_ok=True)
    payload = {
        "subject": subject_key,
        "meta": meta,
        **subject_result,   # acc_ori + per-budget {QDPS,SETS,stats}
    }
    path = os.path.join(RETRAIN_SUBJECTS_DIR, f"{subject_key}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_results(exp_dir, meta, results):
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    lines = []
    lines.append(f"RQ4 Retraining Experiment ({meta['n_runs']} runs/method)")
    lines.append("=" * 88)
    lines.append(
        f"{'Subject':<26}{'k':>5}{'method':>8}{'mean_imp':>12}{'std_imp':>10}{'mean_acc_re':>13}"
    )
    lines.append("-" * 88)
    for subject, sub in results.items():
        acc_ori = sub.get("acc_ori")
        for k in meta["budgets"]:
            kb = sub.get(str(k), {})
            for method in meta["methods"]:
                m = kb.get(method)
                if not m:
                    continue
                lines.append(
                    f"{subject:<26}{k:>5}{method:>8}"
                    f"{m['mean_improvement']*100:>11.2f}%{m['std_improvement']*100:>9.2f}%"
                    f"{m['mean_acc_re']:>13.4f}"
                )
            stats = kb.get("stats")
            if stats:
                star = " *" if stats["significant"] else ""
                lines.append(
                    f"{'  -> QDPS vs SETS':<26}{k:>5}{'':>8}"
                    f"{stats['mean_delta_improvement']*100:>11.2f}%"
                    f"   p={stats['wilcoxon_p']:.4f} winner={stats['winner']}{star}"
                )
        if acc_ori is not None:
            lines.append(f"  (acc_ori on V = {acc_ori:.4f})")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write(summary)
    return summary
