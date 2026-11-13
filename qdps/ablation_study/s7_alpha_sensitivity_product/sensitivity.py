"""
S7 — Alpha sensitivity of QDPS-PF-product.

Sweeps alpha (candidate pool = top alpha*k by uncertainty) with the S6 setup:
product kernel K_vgg ⊙ K_prob, no reserve, q_i = u_i (p = 1).

Writes to `s7_alpha_sensitivity_product/results/<timestamp>/`:
    results.json         per alpha / subject / budget FDR and time
    alpha_sensitivity.png  time (left) and FDR (right) vs alpha, one row per budget
    tables.md            FDR and time per subject for every alpha

Usage (from the repo root):
    python -m qdps.ablation_study.s7_alpha_sensitivity_product.sensitivity [n_runs] [alphas]
    python -m qdps.ablation_study.s7_alpha_sensitivity_product.sensitivity 3 2,3,4,5,6,7,8,9,10
"""
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qdps.ablation_study.core import qdps_select
from qdps.io.loader import DATA_MODEL_PAIRS, load_subject, compute_fdr

METHOD_NAME = "QDPS-PF-product"
FIXED = {"reserve_ratio": 0.0, "quality_power": 1.0, "kernel": "product"}
ALPHAS = list(range(2, 11))
BUDGETS = [100, 300, 500]
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# Categorical slots 1-8 in fixed order; markers give a second, non-colour cue
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def run_sweep(n_runs, alphas, fixed=FIXED):
    results = {}
    for data_name, model_name in DATA_MODEL_PAIRS:
        key = f"{data_name}_{model_name}"
        print(f"Subject: {key}")
        subject = load_subject(data_name, model_name)
        results[key] = {}
        for k in BUDGETS:
            results[key][str(k)] = {}
            for alpha in alphas:
                fdrs, times = [], []
                for _ in range(n_runs):
                    start = time.time()
                    selected = qdps_select(k, subject["index"], subject["features"],
                                           subject["output_probability"], alpha=alpha, **fixed)
                    times.append(time.time() - start)
                    fdr, _ = compute_fdr(selected, subject["mis_index"], subject["cluster_labels"],
                                         k, subject["total_faults"])
                    fdrs.append(fdr)
                results[key][str(k)][str(alpha)] = {
                    "fdrs": fdrs, "times": times,
                    "mean_fdr": float(np.mean(fdrs)), "mean_time": float(np.mean(times)),
                }
            row = " ".join(f"a{a}={results[key][str(k)][str(a)]['mean_fdr']:.3f}" for a in alphas)
            print(f"  k={k}: {row}")
    return results


def plot(results, alphas, out, method_name=METHOD_NAME, subtitle="no reserve, p = 1"):
    subjects = list(results)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED})
    fig, axes = plt.subplots(len(BUDGETS), 2, figsize=(11, 4 * len(BUDGETS)), facecolor="white")
    letters = iter("abcdefghijkl")
    for row, k in enumerate(BUDGETS):
        for col, (metric, ylabel) in enumerate((("mean_time", "Selection time (s)"),
                                                ("mean_fdr", "FDR"))):
            ax = axes[row, col]
            for i, s in enumerate(subjects):
                ys = [results[s][str(k)][str(a)][metric] for a in alphas]
                ax.plot(alphas, ys, color=COLORS[i], marker=MARKERS[i], markersize=6,
                        linewidth=2, label=s, markeredgecolor="white", markeredgewidth=0.8)
            ax.set_xticks(alphas)
            ax.set_xlabel("α")
            ax.set_ylabel(ylabel)
            ax.grid(color=GRID, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"({next(letters)}) k = {k}", y=-0.28, color=INK, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{method_name} — sensitivity to α ({subtitle})", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=3)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_tables(results, alphas, out, method_name=METHOD_NAME, fixed=FIXED, study="S7"):
    lines = [f"# {study} — {method_name} sensitivity to α", "",
             f"Fixed: {fixed}. Bold = best FDR / fastest time in the row.", ""]
    for metric, title, better in (("mean_fdr", "FDR (higher is better)", max),
                                  ("mean_time", "Selection time in seconds (lower is better)", min)):
        lines += [f"## {title}", "",
                  "| Subject | k | " + " | ".join(f"α={a}" for a in alphas) + " |",
                  "|---|---|" + "---|" * len(alphas)]
        sep = "| " + " | ".join(["—"] * (len(alphas) + 2)) + " |"
        for i_s, s in enumerate(results):
            if i_s > 0:
                lines.append(sep)
            for i_k, k in enumerate(BUDGETS):
                vals = [results[s][str(k)][str(a)][metric] for a in alphas]
                best = better(vals)
                cells = [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in vals]
                label = f"**{s}**" if i_k == 0 else ""
                lines.append(f"| {label} | {k} | " + " | ".join(cells) + " |")
        lines.append(sep)
        for k in BUDGETS:
            means = [np.mean([results[s][str(k)][str(a)][metric] for s in results]) for a in alphas]
            lines.append(f"| **Mean** | {k} | " + " | ".join(f"{m:.3f}" for m in means) + " |")
        means = [np.mean([results[s][str(k)][str(a)][metric] for s in results for k in BUDGETS])
                 for a in alphas]
        lines.append("| **Mean** | all | " + " | ".join(f"{m:.3f}" for m in means) + " |")
        lines.append("")
    with open(out, "w") as f:
        f.write("\n".join(lines))


def main(study_dir, study, method_name, fixed, subtitle):
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    alphas = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else ALPHAS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(study_dir, "results", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    results = run_sweep(n_runs, alphas, fixed)
    meta = {"method": method_name, "study": study, "fixed": fixed,
            "alphas": alphas, "budgets": BUDGETS, "n_runs": n_runs, "timestamp": timestamp,
            "subjects": list(results)}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    plot(results, alphas, os.path.join(run_dir, "alpha_sensitivity.png"), method_name, subtitle)
    write_tables(results, alphas, os.path.join(run_dir, "tables.md"), method_name, fixed,
                 study.split("_")[0].upper())
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main(STUDY_DIR, "s7_alpha_sensitivity_product", METHOD_NAME, FIXED, "no reserve, p = 1")
