"""
RQ2 timing: QDPS vs SETS selection time, measured in the same process and interleaved
(QDPS run, then SETS run, repeated 30 times) so both see identical machine conditions.
Two warm-up runs per configuration are discarded.

Results saved to: qdps/results/timing/TIMING_<timestamp>/results.json
"""
import os
import json
import time
import platform
import numpy as np

from qdps import select as qdps_select
from qdps.baselines.sets import sets_select
from qdps.io.loader import load_subject, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, RESULTS_DIR

N_RUNS = 30
N_WARMUP = 2
BUDGETS = [100, 300, 500]
METHODS = {"QDPS": qdps_select, "SETS": sets_select}


def timed(fn, subject, k):
    start = time.perf_counter()
    fn(k, subject["index"], subject["features"], subject["output_probability"])
    return time.perf_counter() - start


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, "timing", f"TIMING_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for dn, mn in DATA_MODEL_PAIRS:
        key = f"{dn}_{mn}"
        subject = load_subject(dn, mn, BASE_DATA)
        results[key] = {}
        for k in BUDGETS:
            for _ in range(N_WARMUP):
                for fn in METHODS.values():
                    timed(fn, subject, k)
            times = {m: [] for m in METHODS}
            for _ in range(N_RUNS):
                for m, fn in METHODS.items():
                    times[m].append(timed(fn, subject, k))
            results[key][str(k)] = {
                m: {"times": t, "mean_time": float(np.mean(t)), "std_time": float(np.std(t, ddof=1))}
                for m, t in times.items()
            }
            q, s = results[key][str(k)]["QDPS"]["mean_time"], results[key][str(k)]["SETS"]["mean_time"]
            print(f"{key:28s} k={k}: QDPS={q:.4f}s  SETS={s:.4f}s  speedup={s / q:.1f}x", flush=True)
    meta = {"experiment": "RQ2_timing", "timestamp": ts, "n_runs": N_RUNS, "n_warmup": N_WARMUP,
            "budgets": BUDGETS, "methods": list(METHODS), "machine": platform.platform(),
            "processor": platform.processor(), "python": platform.python_version(),
            "numpy": np.__version__}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
