"""
Composition of the QDPS and SETS selections (supports the Discussion of the evaluation).

For every subject x budget, reports for each method:
  - mis:     number of selected inputs that the model mispredicts
  - faults:  distinct faults exposed (1noisy convention, as in compute_fdr)
  - per_mis: faults / mis  (share of mispredicted picks that expose a *new* fault)

Results saved to: qdps/results/composition/composition.json
"""
import os
import json

from qdps import select as qdps_select
from qdps.baselines.sets import sets_select
from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, RESULTS_DIR

BUDGETS = [100, 300, 500]
METHODS = {"QDPS": qdps_select, "SETS": sets_select}


def main():
    out = {}
    for dn, mn in DATA_MODEL_PAIRS:
        key = f"{dn}_{mn}"
        s = load_subject(dn, mn, BASE_DATA)
        mis_set = set(s["mis_index"])
        out[key] = {}
        for k in BUDGETS:
            row = {}
            for m, fn in METHODS.items():
                sel = fn(k, s["index"], s["features"], s["output_probability"])
                n_mis = sum(1 for i in sel if i in mis_set)
                fdr, faults = compute_fdr(sel, s["mis_index"], s["cluster_labels"], k, s["total_faults"])
                row[m] = {"mis": n_mis, "faults": faults, "fdr": fdr,
                          "per_mis": faults / n_mis if n_mis else 0.0}
            out[key][str(k)] = row
            q, t = row["QDPS"], row["SETS"]
            print(f"{key:24s} k={k}: QDPS mis={q['mis']:3d} faults={q['faults']:3d} | "
                  f"SETS mis={t['mis']:3d} faults={t['faults']:3d}")
    out_dir = os.path.join(RESULTS_DIR, "composition")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "composition.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
