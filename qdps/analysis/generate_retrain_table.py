"""Build the QDPS-vs-SETS retraining comparison table.

Reads the durable per-subject results in ``results/retrain/subjects/*.json`` and
prints/saves the comparison table (mean improvement per method, delta, Wilcoxon
p, winner; ``*`` = significant at p<0.05).

Usage:
    python -m qdps.analysis.generate_retrain_table [k]      # default k=500
"""
import json
import os
import sys

from qdps.io.paths import RETRAIN_SUBJECTS_DIR, RETRAIN_TABLES_DIR

# Canonical subject order + short column labels (dataset/model).
SUBJECT_ORDER = [
    "mnist_LeNet1", "mnist_LeNet5", "Fashion_mnist_LeNet4",
    "cifar10_12Conv", "cifar10_ResNet20", "SVHN_LeNet5",
    "Fruit360_ResNet50", "TinyImageNet_ResNet101",
]
def load_all():
    """Return {subject_key: payload} for every result file present."""
    out = {}
    if not RETRAIN_SUBJECTS_DIR.exists():
        return out
    for fname in os.listdir(RETRAIN_SUBJECTS_DIR):
        if fname.endswith(".json"):
            with open(RETRAIN_SUBJECTS_DIR / fname) as f:
                payload = json.load(f)
            out[payload.get("subject", fname[:-5])] = payload
    return out


def _cell(payload, k):
    """Return (qdps_imp%, sets_imp%, winner, significant, p) or None if k absent."""
    kb = payload.get(str(k))
    if not kb or "QDPS" not in kb or "SETS" not in kb:
        return None
    q = kb["QDPS"]["mean_improvement"] * 100
    s = kb["SETS"]["mean_improvement"] * 100
    stats = kb.get("stats", {})
    return q, s, stats.get("winner"), bool(stats.get("significant")), stats.get("wilcoxon_p")


def build(k=500):
    data = load_all()
    subjects = [s for s in SUBJECT_ORDER if s in data]
    subjects += [s for s in data if s not in SUBJECT_ORDER]  # any unknown extras
    rows = []
    for s in subjects:
        cell = _cell(data[s], k)
        if cell is None:
            continue
        q, sv, winner, sig, p = cell
        rows.append((s, q, sv, winner, sig, p))
    return rows


def console_table(rows, k):
    lines = [f"\nQDPS vs SETS — retraining accuracy improvement (k={k})",
             "=" * 78,
             f"{'Subject':<26}{'QDPS':>9}{'SETS':>9}{'Delta':>9}{'p':>8}  winner",
             "-" * 78]
    for s, q, sv, winner, sig, p in rows:
        star = "*" if sig else " "
        p_txt = f"{p:.4f}" if isinstance(p, (int, float)) and p == p else "n/a"
        lines.append(f"{s:<26}{q:>8.2f}%{sv:>8.2f}%{q-sv:>+8.2f}%"
                     f"{p_txt:>8}  {winner}{star}")
    if not rows:
        lines.append("(no subject results found — run `make retrain` first)")
    return "\n".join(lines) + "\n"


def main(argv):
    k = int(argv[0]) if argv else 500
    rows = build(k)
    console = console_table(rows, k)
    print(console)
    os.makedirs(RETRAIN_TABLES_DIR, exist_ok=True)
    out_path = os.path.join(RETRAIN_TABLES_DIR, "table_retrain_qdps_vs_sets.txt")
    with open(out_path, "w") as f:
        f.write(console)
    print(f"Saved -> {out_path}  ({len(rows)} subjects, k={k})")


if __name__ == "__main__":
    main(sys.argv[1:])
