"""Tables of the DeepGD-protocol retraining experiment (DeepGD Tables 6 and 7 layout).

Reads RESULTS_DIR/retrain_deepgd/subjects/*.json and writes
RESULTS_DIR/retrain_deepgd/TABLES.md:
- Table A: accuracy improvement on T and G (mean over runs), QDPS vs SETS.
- Table B: optimization effectiveness on T and G (improvement / max improvement).
- Wilcoxon signed-rank test across subjects (as in DeepGD), QDPS vs SETS, per metric.

Usage (repo root): python -m qdps.retrain.retrain_deepgd_version.report
"""
import json

import numpy as np
from scipy.stats import wilcoxon

from qdps.io.paths import RESULTS_DIR
from qdps.retrain.retrain_deepgd_version.protocol import METHODS
from qdps.retrain.retrain_deepgd_version.run_retrain_deepgd import SUBJECTS_DIR

SUBJECT_ORDER = ["mnist_LeNet1", "mnist_LeNet5", "Fashion_mnist_LeNet4", "cifar10_12Conv",
                 "cifar10_ResNet20", "SVHN_LeNet5", "Fruit360_ResNet50", "TinyImageNet_ResNet101"]


def mean_improvement(store, condition, split):
    runs = store["conditions"].get(condition, {}).get("runs")
    if not runs:
        return None
    return float(np.mean([r[f"acc_{split}"] for r in runs])) - store["original"][f"acc_{split}"]


def collect():
    rows = {}
    for subject in SUBJECT_ORDER:
        path = SUBJECTS_DIR / f"{subject}.json"
        if not path.exists():
            continue
        s = json.load(open(path))
        row = {"orig_T": s["original"]["acc_T"], "orig_G": s["original"]["acc_G"]}
        for split in ("T", "G"):
            max_imp = mean_improvement(s, f"FULL_{split}", split)
            row[f"max_{split}"] = max_imp
            for m in METHODS:
                imp = mean_improvement(s, m, split)
                row[f"{m}_imp_{split}"] = imp
                # A ratio to a non-positive maximum is meaningless: reported as n/a.
                ok = imp is not None and max_imp is not None and max_imp > 0
                row[f"{m}_eff_{split}"] = (imp / max_imp) if ok else None
        rows[subject] = row
    return rows


def _pct(v):
    return "–" if v is None else f"{v:.2%}"


def _bold_pair(a, b):
    """Bold the higher value, only when the difference shows at the displayed precision."""
    if a is None or b is None or _pct(a) == _pct(b):
        return _pct(a), _pct(b)
    return (f"**{_pct(a)}**", _pct(b)) if a > b else (_pct(a), f"**{_pct(b)}**")


def _wilcoxon_line(rows, key_a, key_b, label):
    pairs = [(r[key_a], r[key_b]) for r in rows.values() if r[key_a] is not None and r[key_b] is not None]
    if len(pairs) < 2:
        return f"- {label}: not enough subjects ({len(pairs)})."
    a, b = np.array(pairs).T
    wins = int(np.sum(a > b))
    try:
        p = wilcoxon(a, b).pvalue
        return f"- {label}: QDPS higher on {wins}/{len(a)} subjects, Wilcoxon p = {p:.4f}."
    except ValueError:
        return f"- {label}: QDPS higher on {wins}/{len(a)} subjects (all differences zero)."


def write_tables():
    rows = collect()
    lines = ["# Retraining, DeepGD protocol: QDPS vs SETS", "",
             "k = 300 inputs selected from the whole test set, added to the training set, model "
             "retrained (5 runs, mean). T = original test set, G = combined generated set "
             "(5 transformations). Bold = higher.", "",
             "## Accuracy improvement", "",
             "| Subject | Original T | QDPS T | SETS T | Original G | QDPS G | SETS G |",
             "|---|---|---|---|---|---|---|"]
    for subject, r in rows.items():
        qt, st = _bold_pair(r["QDPS_imp_T"], r["SETS_imp_T"])
        qg, sg = _bold_pair(r["QDPS_imp_G"], r["SETS_imp_G"])
        lines.append(f"| {subject} | {_pct(r['orig_T'])} | {qt} | {st} | {_pct(r['orig_G'])} | {qg} | {sg} |")
    lines += ["", "## Optimization effectiveness (improvement / improvement with 100% of T or G)", "",
              "| Subject | Max imp. T | QDPS T | SETS T | Max imp. G | QDPS G | SETS G |",
              "|---|---|---|---|---|---|---|"]
    for subject, r in rows.items():
        qt, st = _bold_pair(r["QDPS_eff_T"], r["SETS_eff_T"])
        qg, sg = _bold_pair(r["QDPS_eff_G"], r["SETS_eff_G"])
        lines.append(f"| {subject} | {_pct(r['max_T'])} | {qt} | {st} | {_pct(r['max_G'])} | {qg} | {sg} |")
    lines += ["", "– = not run, or the maximum improvement is not positive (ratio undefined)."]
    lines += ["", "## Wilcoxon signed-rank test across subjects (QDPS vs SETS)", "",
              _wilcoxon_line(rows, "QDPS_imp_T", "SETS_imp_T", "Improvement on T"),
              _wilcoxon_line(rows, "QDPS_imp_G", "SETS_imp_G", "Improvement on G"),
              _wilcoxon_line(rows, "QDPS_eff_T", "SETS_eff_T", "Optimization effectiveness on T"),
              _wilcoxon_line(rows, "QDPS_eff_G", "SETS_eff_G", "Optimization effectiveness on G")]
    out = RESULTS_DIR / "retrain_deepgd" / "TABLES.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    print(f"Written {write_tables()}")
