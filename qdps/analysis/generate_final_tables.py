"""
Generate the final FDR and execution-time tables (QDPS vs SETS) as Markdown.

Inputs (latest run of each, or the paths given on the command line):
    results/fdr/QDPS_<ts>/results.json        QDPS FDR (run_experiment)
    results/sets_baseline/sets_results.json   SETS FDR (our local SETS run)
    results/timing/TIMING_<ts>/results.json   QDPS and SETS times, interleaved (run_timing)

Outputs:
    results/final_results/TABLE_FDR.md
    results/final_results/TABLE_TIME.md

Usage (from the repo root):
    python -m qdps.analysis.generate_final_tables [fdr_run_dir] [timing_run_dir]
"""
import glob
import json
import os
import sys

import numpy as np

from qdps.io.paths import FDR_RESULTS_DIR, RESULTS_DIR, SETS_RESULTS_DIR

FINAL_DIR = RESULTS_DIR / "final_results"
TIMING_DIR = RESULTS_DIR / "timing"
BUDGETS = [100, 300, 500]
TIE_BAND = 0.005

SUBJECTS_DISPLAY = [
    # (subject key, dataset label, model label)
    ("mnist_LeNet1", "MNIST", "LeNet1"),
    ("mnist_LeNet5", "", "LeNet5"),
    ("Fashion_mnist_LeNet4", "Fashion", "LeNet4"),
    ("cifar10_12Conv", "CIFAR-10", "12Conv"),
    ("cifar10_ResNet20", "", "ResNet20"),
    ("SVHN_LeNet5", "SVHN", "LeNet5"),
    ("Fruit360_ResNet50", "Fruit-360", "ResNet50"),
    ("TinyImageNet_ResNet101", "TinyImageNet", "ResNet101"),
]
SHORT = {
    "mnist_LeNet1": "MNIST LeNet1", "mnist_LeNet5": "MNIST LeNet5",
    "Fashion_mnist_LeNet4": "Fashion LeNet4", "cifar10_12Conv": "CIFAR-10 12Conv",
    "cifar10_ResNet20": "CIFAR-10 ResNet20", "SVHN_LeNet5": "SVHN LeNet5",
    "Fruit360_ResNet50": "Fruit-360 ResNet50", "TinyImageNet_ResNet101": "TinyImageNet ResNet101",
}


def latest(pattern):
    runs = sorted(glob.glob(pattern))
    if not runs:
        raise SystemExit(f"No run found matching {pattern}")
    return runs[-1]


def header(title):
    cols = " | ".join(f"QDPS k={k} | SETS k={k}" for k in BUDGETS)
    return [f"| Dataset | DNN Model | {cols} |", "|---|---|" + "---|" * (2 * len(BUDGETS))]


def fdr_table(fdr_run, sets_path):
    qdps = json.load(open(os.path.join(fdr_run, "results.json")))
    sets = json.load(open(sets_path))
    meta = qdps["meta"]
    wins, ties, losses, loss_list = 0, 0, 0, []

    lines = ["# Fault Detection Rates (FDRs) of QDPS and SETS Approaches", "",
             f"QDPS: final configuration (product kernel, α = 3), run `{os.path.basename(fdr_run)}`, "
             f"{meta['n_runs']} runs per configuration. SETS: our local run.",
             f"Bold = better method (difference > {TIE_BAND * 100:.1f}%).", ""]
    lines += header("FDR")
    for key, ds, model in SUBJECTS_DISPLAY:
        cells = []
        for k in BUDGETS:
            q = qdps["results"][key][str(k)]["mean_fdr"]
            s = float(np.mean(sets[key][str(k)]["fdrs"]))
            d = q - s
            if d > TIE_BAND:
                wins += 1
            elif d < -TIE_BAND:
                losses += 1
                loss_list.append(f"{SHORT[key]} k={k}")
            else:
                ties += 1
            cells += [f"**{q:.0%}**" if d > TIE_BAND else f"{q:.0%}",
                      f"**{s:.0%}**" if d < -TIE_BAND else f"{s:.0%}"]
        lines.append(f"| {ds} | {model} | " + " | ".join(cells) + " |")

    mean_q = np.mean([qdps["results"][key][str(k)]["mean_fdr"] for key, _, _ in SUBJECTS_DISPLAY for k in BUDGETS])
    mean_s = np.mean([np.mean(sets[key][str(k)]["fdrs"]) for key, _, _ in SUBJECTS_DISPLAY for k in BUDGETS])
    lines += ["", f"**Summary: QDPS wins {wins} / ties {ties} / loses {losses} out of {wins + ties + losses} configurations.**",
              "", f"Mean FDR over the 24 configurations: QDPS {mean_q:.1%}, SETS {mean_s:.1%}.",
              "", "QDPS losses: " + (", ".join(loss_list) if loss_list else "none") + "."]
    return "\n".join(lines) + "\n"


def time_table(timing_run):
    data = json.load(open(os.path.join(timing_run, "results.json")))
    meta, res = data["meta"], data["results"]
    faster, speedups = 0, {k: [] for k in BUDGETS}

    lines = ["# Execution Time Costs (in Seconds) of QDPS and SETS Approaches", "",
             f"Run `{os.path.basename(timing_run)}`: QDPS and SETS timed alternately in the same process "
             f"({meta['n_warmup']} warm-up runs discarded, {meta['n_runs']} timed runs averaged) on "
             f"{meta['machine']}.", "Bold = faster method.", ""]
    lines += header("time")
    for key, ds, model in SUBJECTS_DISPLAY:
        cells = []
        for k in BUDGETS:
            q = res[key][str(k)]["QDPS"]["mean_time"]
            s = res[key][str(k)]["SETS"]["mean_time"]
            faster += q < s
            speedups[k].append((s / q, key))
            cells += [f"**{q:.3f}**" if q < s else f"{q:.3f}", f"**{s:.3f}**" if s < q else f"{s:.3f}"]
        lines.append(f"| {ds} | {model} | " + " | ".join(cells) + " |")

    n = len(SUBJECTS_DISPLAY) * len(BUDGETS)
    lines += ["", f"**Summary: QDPS is faster in {faster}/{n} configurations.**", "",
              "| Budget | Speedup range (SETS time / QDPS time) | Best case |", "|---|---|---|"]
    for k in BUDGETS:
        vals = [v for v, _ in speedups[k]]
        best_v, best_key = max(speedups[k])
        lines.append(f"| k={k} | {min(vals):.1f}–{max(vals):.1f}x | {SHORT[best_key]}: {best_v:.1f}x |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    fdr_run = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(latest(str(FDR_RESULTS_DIR / "QDPS_*" / "results.json")))
    timing_run = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(latest(str(TIMING_DIR / "TIMING_*" / "results.json")))
    os.makedirs(FINAL_DIR, exist_ok=True)
    with open(FINAL_DIR / "TABLE_FDR.md", "w") as f:
        f.write(fdr_table(fdr_run, SETS_RESULTS_DIR / "sets_results.json"))
    with open(FINAL_DIR / "TABLE_TIME.md", "w") as f:
        f.write(time_table(timing_run))
    print(f"FDR table from {fdr_run}\nTime table from {timing_run}\nWritten to {FINAL_DIR}")
