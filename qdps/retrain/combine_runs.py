"""Merge the RQ4 run stores into one N-run store, without touching the inputs.

subjects/ holds runs 1-5 (seeds 0-4); subjects_runs5-9/ holds runs 6-10 (seeds 5-9).
This writes subjects_10runs/<subject>.json with the runs concatenated and the means,
stds and the per-subject Wilcoxon test recomputed. Checks that both stores used the
same original accuracy and the same selections.

Usage (repo root):
    python -m qdps.retrain.combine_runs subjects_runs5-9 subjects_10runs
    python -m qdps.analysis.generate_retrain_table 500 subjects_10runs
"""
import json
import sys

import numpy as np

from qdps.io.paths import RETRAIN_RESULTS_DIR, RETRAIN_SUBJECTS_DIR
from qdps.retrain.report import wilcoxon_qdps_vs_sets


def combine(extra_name, out_name, k="500"):
    out_dir = RETRAIN_RESULTS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for extra_path in sorted((RETRAIN_RESULTS_DIR / extra_name).glob("*.json")):
        base = json.load(open(RETRAIN_SUBJECTS_DIR / extra_path.name))
        extra = json.load(open(extra_path))
        assert abs(base["acc_ori"] - extra["acc_ori"]) < 1e-12, f"{extra_path.stem}: acc_ori differs"
        merged = {"subject": base["subject"], "acc_ori": base["acc_ori"],
                  "retrain_config": base["retrain_config"],
                  "sources": [str(RETRAIN_SUBJECTS_DIR.name), extra_name], k: {}}
        for method in ("QDPS", "SETS"):
            b, e = base[k][method], extra[k][method]
            assert b.get("selected") == e.get("selected"), f"{extra_path.stem} {method}: selection differs"
            runs = [dict(r, seed=r.get("seed", i)) for i, r in enumerate(b["runs"])] + e["runs"]
            imps = [r["improvement"] for r in runs]
            merged[k][method] = {**{x: b[x] for x in ("selected_size", "selection_source", "selected") if x in b},
                                 "runs": runs, "mean_improvement": float(np.mean(imps)),
                                 "std_improvement": float(np.std(imps)),
                                 "mean_acc_re": float(np.mean([r["acc_re"] for r in runs]))}
        q = [r["improvement"] for r in merged[k]["QDPS"]["runs"]]
        s = [r["improvement"] for r in merged[k]["SETS"]["runs"]]
        merged[k]["stats"] = wilcoxon_qdps_vs_sets(q, s)
        json.dump(merged, open(out_dir / extra_path.name, "w"), indent=2)
        print(f"{extra_path.stem}: {len(q)} runs per method")


if __name__ == "__main__":
    combine(sys.argv[1], sys.argv[2])
