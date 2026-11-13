"""
Plot FDR and selection time per subject for one ablation run:
SETS vs QDPS (initial, tuned) vs the study's QDPS variant.

Writes `fdr_time.png` (chart) and `tables.md` (FDR and time tables) into the run folder.

Usage (from the repo root):
    python -m qdps.ablation_study.plot_run <study> [timestamp]   # latest run if omitted
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qdps.io.paths import FDR_RESULTS_DIR, SETS_RESULTS_DIR

STUDY_ROOT = os.path.dirname(os.path.abspath(__file__))

# Reference runs the variant is compared against
QDPS_INITIAL_RUN = FDR_RESULTS_DIR / "QDPS_20260623_163421" / "results.json"
SETS_RUN = SETS_RESULTS_DIR / "sets_results.json"

# Categorical slots 1-3 (validated all-pairs, CVD-safe)
COLORS = {"QDPS initial": "#2a78d6", "variant": "#eb6834", "SETS": "#1baf7a"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SHORT_NAMES = {
    "mnist_LeNet1": "MNIST\nLeNet1", "mnist_LeNet5": "MNIST\nLeNet5",
    "Fashion_mnist_LeNet4": "Fashion\nLeNet4", "cifar10_12Conv": "CIFAR10\n12Conv",
    "cifar10_ResNet20": "CIFAR10\nResNet20", "SVHN_LeNet5": "SVHN\nLeNet5",
    "Fruit360_ResNet50": "Fruit360\nResNet50", "TinyImageNet_ResNet101": "TinyImgNet\nResNet101",
}


def load_sets():
    raw = json.load(open(SETS_RUN))
    return {s: {k: {"fdr": float(np.mean(v["fdrs"])), "time": float(np.mean(v["times"]))}
                for k, v in per_k.items()} for s, per_k in raw.items()}


def load_qdps_run(path):
    raw = json.load(open(path))
    res = {s: {str(k): {"fdr": v["mean_fdr"], "time": v["mean_time"]} for k, v in per_k.items()}
           for s, per_k in raw["results"].items()}
    return raw["meta"], res


def load_series(run_dir):
    meta, variant = load_qdps_run(os.path.join(run_dir, "results.json"))
    _, initial = load_qdps_run(QDPS_INITIAL_RUN)
    series = [("QDPS initial", initial, COLORS["QDPS initial"]),
              (meta["method"], variant, COLORS["variant"]),
              ("SETS", load_sets(), COLORS["SETS"])]
    return meta, series


def _metric_table(metric, series, subjects, budgets, higher_is_better, fmt):
    """Per-subject table; the last series is the reference (SETS) for the diff columns."""
    names = [name for name, _, _ in series]
    ref_name = names[-1]
    diff_names = [f"{n} − {ref_name}" for n in names[:-1]]
    n_cols = len(names) + len(diff_names) + 2
    header = "| Subject | k | " + " | ".join(names + diff_names) + " |"
    lines = [header, "|---|---|" + "---|" * (n_cols - 2)]
    totals = {name: [] for name in names}
    separator = "| " + " | ".join(["—"] * n_cols) + " |"
    for i_s, s in enumerate(subjects):
        if i_s > 0:
            lines.append(separator)
        for i_k, k in enumerate(budgets):
            vals = [data.get(s, {}).get(k, {}).get(metric, np.nan) for _, data, _ in series]
            for name, v in zip(names, vals):
                totals[name].append(v)
            best = np.nanmax(vals) if higher_is_better else np.nanmin(vals)
            cells = [f"**{fmt(v)}**" if v == best else fmt(v) for v in vals]
            diffs = [f"{v - vals[-1]:+.3f}" for v in vals[:-1]]
            label = f"**{s}**" if i_k == 0 else ""
            lines.append(f"| {label} | {k} | " + " | ".join(cells + diffs) + " |")
    lines.append(separator)
    means = [np.nanmean(totals[n]) for n in names]
    mean_diffs = [f"{m - means[-1]:+.3f}" for m in means[:-1]]
    lines.append("| **Mean** | all | " + " | ".join([fmt(m) for m in means] + mean_diffs) + " |")
    return "\n".join(lines)


def _win_tie_loss(variant, reference, subjects, budgets, band=0.005):
    w = t = l = 0
    for s in subjects:
        for k in budgets:
            d = variant[s][k]["fdr"] - reference[s][k]["fdr"]
            w, t, l = (w + 1, t, l) if d > band else (w, t, l + 1) if d < -band else (w, t + 1, l)
    return w, t, l


def write_tables(run_dir):
    meta, series = load_series(run_dir)
    subjects = meta["subjects"]
    budgets = [str(b) for b in meta["budgets"]]
    (_, initial, _), (variant_name, variant, _), (_, sets, _) = series

    lines = [f"# {meta['study']} — FDR and selection time per subject", "",
             f"- Variant: **{variant_name}**, runs per configuration: {meta['n_runs']}",
             f"- QDPS initial: `{QDPS_INITIAL_RUN.parent.name}` (tuned QDPS)",
             f"- SETS: local baseline run `{SETS_RUN.name}`",
             "- Bold = best value in the row. Diff columns = method − SETS (FDR: positive is better; time: negative is faster).",
             "- Ties use a ±0.005 FDR band in the W/T/L counts.", ""]
    lines.append("## Win / tie / loss on FDR")
    lines.append("")
    lines.append("| Comparison | Win | Tie | Loss |")
    lines.append("|---|---|---|---|")
    for label, a, b in ((f"{variant_name} vs SETS", variant, sets),
                        (f"QDPS initial vs SETS", initial, sets),
                        (f"{variant_name} vs QDPS initial", variant, initial)):
        w, t, l = _win_tie_loss(a, b, subjects, budgets)
        lines.append(f"| {label} | {w} | {t} | {l} |")
    lines += ["", "## FDR (higher is better)", "",
              _metric_table("fdr", series, subjects, budgets, True, lambda v: f"{v:.3f}"),
              "", "## Selection time in seconds (lower is better)", "",
              _metric_table("time", series, subjects, budgets, False, lambda v: f"{v:.3f}"), ""]
    out = os.path.join(run_dir, "tables.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    return out


def plot_run(run_dir):
    meta, series = load_series(run_dir)
    subjects = meta["subjects"]
    budgets = [str(b) for b in meta["budgets"]]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED})
    fig, axes = plt.subplots(2, len(budgets), figsize=(5.2 * len(budgets), 7.5),
                             squeeze=False, facecolor="white")
    x = np.arange(len(subjects))
    width = 0.27

    for col, k in enumerate(budgets):
        for row, metric in enumerate(("fdr", "time")):
            ax = axes[row, col]
            for i, (name, data, color) in enumerate(series):
                vals = [data.get(s, {}).get(k, {}).get(metric, np.nan) for s in subjects]
                ax.bar(x + (i - 1) * width, vals, width * 0.9, color=color, label=name,
                       edgecolor="white", linewidth=1, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT_NAMES.get(s, s) for s in subjects], fontsize=7)
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            if metric == "fdr":
                ax.set_title(f"k = {k}", color=INK, fontsize=11, fontweight="bold")
                ax.set_ylim(0, 1)
                if col == 0:
                    ax.set_ylabel("FDR (higher is better)")
            else:
                ax.set_yscale("log")
                ax.set_ylim(0.01, 10)
                ticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:g}" for t in ticks])
                ax.minorticks_off()
                if col == 0:
                    ax.set_ylabel("Selection time, s (log scale, lower is better)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{meta['study']} — FDR and selection time per subject",
                 color=INK, fontsize=12, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(run_dir, "fdr_time.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: plot_run <study> [timestamp]")
    study = sys.argv[1]
    if len(sys.argv) > 2:
        run_dir = os.path.join(STUDY_ROOT, study, "results", sys.argv[2])
    else:
        run_dir = sorted(glob.glob(os.path.join(STUDY_ROOT, study, "results", "*/")))[-1]
    print(f"Saved: {plot_run(run_dir)}")
    print(f"Saved: {write_tables(run_dir)}")
