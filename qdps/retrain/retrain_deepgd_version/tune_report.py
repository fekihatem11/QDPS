"""Pick each subject's retraining setting from the tuning runs (tune_settings.py).

Rule, fixed before the tuning runs were read (2026-10-28):
- candidates, in this fixed order: CANDIDATES (current setting first);
- best = the highest score (mean gain on T_val and G_val, accuracy points);
- chosen = the first candidate in that order whose score is within TOLERANCE of best,
  so a setting closer to the original protocol wins a near-tie.
QDPS and SETS results are not used.

Fixed without tuning: MNIST LeNet1/LeNet5 (DeepGD's published setting) and
TinyImageNet ResNet101 (PyTorch, SETS's setting). Fruit-360 always uses the logits loss
(its last layer is linear), only its learning rate is tuned.

Writes RESULTS_DIR/retrain_deepgd/tuning/TUNING.md and chosen_settings.json.
Usage (repo root): python -m qdps.retrain.retrain_deepgd_version.tune_report
"""
import json

from qdps.retrain.retrain_deepgd_version.tune_settings import TUNING_DIR, tag_of

TOLERANCE = 0.5
CANDIDATES = [("adadelta", 0.005), ("adadelta", 0.05), ("adam", 0.0001), ("adadelta", 1.0), ("adam", 0.001)]
TUNED = {
    "Fashion_mnist_LeNet4": (CANDIDATES, False),
    "cifar10_12Conv": (CANDIDATES, False),
    "cifar10_ResNet20": (CANDIDATES, False),
    "SVHN_LeNet5": (CANDIDATES, False),
    "Fruit360_ResNet50": ([("adam", 0.00001), ("adam", 0.0001)], True),
}


def write():
    lines = ["# Retraining settings (step 2)", "",
             f"Score = mean gain (points) on T_val and G_val after retraining on training set + "
             f"T_fit + G_fit. Chosen = first candidate (fixed order) within {TOLERANCE} of the best.", ""]
    chosen = {}
    for subject, (cands, logits) in TUNED.items():
        rows = []
        for opt, lr in cands:
            f = TUNING_DIR / f"{subject}__{tag_of(opt, lr, logits)}.json"
            rows.append((opt, lr, json.load(open(f)) if f.exists() else None))
        done = [r for r in rows if r[2] is not None]
        lines += [f"## {subject}", "", "| Optimizer | lr | Gain T_val | Gain G_val | Score |", "|---|---|---|---|---|"]
        for opt, lr, r in rows:
            lines.append(f"| {opt} | {lr:g} | " + (" | ".join(f"{r[k]:+.2f}" for k in ("gain_T_val", "gain_G_val", "score"))
                                                    if r else "– | – | –") + " |")
        if len(done) == len(rows):
            best = max(r["score"] for _, _, r in done)
            opt, lr, r = next(x for x in done if x[2]["score"] >= best - TOLERANCE)
            chosen[subject] = {"optimizer": opt, "lr": lr, "from_logits": logits, "score": r["score"]}
            lines += ["", f"Chosen: **{opt} lr={lr:g}**" + (" (logits loss)" if logits else "") + f", score {r['score']:+.2f}.", ""]
        else:
            lines += ["", f"Incomplete: {len(done)}/{len(rows)} runs.", ""]
    (TUNING_DIR / "TUNING.md").write_text("\n".join(lines) + "\n")
    json.dump(chosen, open(TUNING_DIR / "chosen_settings.json", "w"), indent=2)
    return chosen


if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
