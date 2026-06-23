"""RQ4 retraining experiment runner: QDPS vs SETS accuracy improvement.

Faithful replica of the SETS paper's RQ4 (``SETS/Source_code/retrain_four.py``),
comparing QDPS-selected vs SETS-selected inputs. For each subject:
select k inputs from the pool T (live, via the package selectors), augment the
training set, retrain a fresh pretrained model, and measure accuracy improvement
on V = test \\ T. Both methods select from the *same* pool; the selected inputs
(subset of T) never overlap V.

Usage:
    python -m qdps.retrain.run_retrain <subject|all-mnist> [budgets] [n_runs]
    e.g. python -m qdps.retrain.run_retrain mnist_LeNet1 500 5
         python -m qdps.retrain.run_retrain all-mnist 500 5
"""
import sys
from datetime import datetime

from qdps.io.loader import load_subject
from qdps.io.paths import RETRAIN_RESULTS_DIR
from qdps.experiments.run_single_subject import SUBJECT_MAP
from qdps.retrain.raw_data import load_raw_subject, load_pretrained_model
from qdps.retrain.splits import load_T, make_V, selection_pool
from qdps.retrain.selectors import SELECTORS
from qdps.retrain.retrain_step import RetrainConfig, original_accuracy, retrain_and_eval
from qdps.retrain.report import (
    aggregate, wilcoxon_qdps_vs_sets, write_results, save_subject_result,
)

# Subjects runnable today (Phase 1). Extended as backends/data are added.
MNIST_SUBJECTS = ["mnist_LeNet1", "mnist_LeNet5"]


def run_subject(subject_key, budgets, methods, n_runs, seed, cfg):
    dn, mn = SUBJECT_MAP[subject_key]
    sel = load_subject(dn, mn)                  # selection artifacts (features, probs, index)
    raw = load_raw_subject(subject_key)         # raw images + labels
    T = load_T(subject_key)
    n_test = len(raw.x_test)
    V = make_V(T, n_test)
    pool = selection_pool(T, sel["index"])

    def model_loader():
        return load_pretrained_model(subject_key)

    acc_ori = original_accuracy(raw, model_loader, V)
    print(f"\n{'='*70}\n  {subject_key}  |  |T|={len(T)} |V|={len(V)} |pool|={len(pool)}"
          f"  acc_ori(V)={acc_ori:.4f}\n{'='*70}")

    sub_result = {"acc_ori": acc_ori}
    for k in budgets:
        k_result = {}
        method_imps = {}
        for method in methods:
            selected = SELECTORS[method](k, pool, sel["features"], sel["output_probability"])
            acc_res, imps = [], []
            for run in range(n_runs):
                run_seed = seed * 1000 + run
                r = retrain_and_eval(raw, model_loader, selected, V, cfg, run_seed)
                imp = r["acc_re"] - acc_ori
                acc_res.append(r["acc_re"])
                imps.append(imp)
                print(f"  k={k} {method} run {run+1}/{n_runs}: "
                      f"acc_re={r['acc_re']:.4f} imp={imp*100:+.2f}%")
            agg = aggregate(acc_res, imps)
            agg["selected_size"] = len(selected)
            k_result[method] = agg
            method_imps[method] = imps
            print(f"  => {method} mean imp = {agg['mean_improvement']*100:+.2f}% "
                  f"(+/- {agg['std_improvement']*100:.2f}%)")
        if "QDPS" in method_imps and "SETS" in method_imps:
            k_result["stats"] = wilcoxon_qdps_vs_sets(method_imps["QDPS"], method_imps["SETS"])
        sub_result[str(k)] = k_result
    return sub_result


def run_retrain_experiment(subjects, budgets=(500,), methods=("QDPS", "SETS"),
                           n_runs=5, seed=0, cfg=None):
    cfg = cfg or RetrainConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "experiment": "RQ4_retrain",
        "timestamp": timestamp,
        "n_runs": n_runs,
        "budgets": list(budgets),
        "methods": list(methods),
        "subjects": list(subjects),
        "retrain_config": vars(cfg),
        "seed": seed,
    }
    results = {}
    for subject_key in subjects:
        sub_result = run_subject(subject_key, budgets, methods, n_runs, seed, cfg)
        results[subject_key] = sub_result
        # Persist immediately so a long multi-subject (or crashed) run keeps
        # every completed subject in the durable comparison store.
        store_path = save_subject_result(subject_key, sub_result, meta)
        print(f"  saved -> {store_path}")

    exp_dir = RETRAIN_RESULTS_DIR / f"RETRAIN_{timestamp}"
    summary = write_results(str(exp_dir), meta, results)
    print(f"\n{summary}\nSaved to: {exp_dir}/")
    return results, exp_dir


def _parse_subjects(arg):
    if arg in ("all-mnist", "mnist"):
        return MNIST_SUBJECTS
    keys = arg.split(",") if "," in arg else [arg]
    unknown = [k for k in keys if k not in SUBJECT_MAP]
    if unknown:
        print(f"Unknown subject(s): {unknown}\nAvailable: {', '.join(SUBJECT_MAP)} | all-mnist")
        sys.exit(1)
    return keys


def main(argv):
    subjects = _parse_subjects(argv[0]) if argv else MNIST_SUBJECTS
    budgets = [int(b) for b in argv[1].split(",")] if len(argv) > 1 else [500]
    n_runs = int(argv[2]) if len(argv) > 2 else 5
    run_retrain_experiment(subjects, budgets=budgets, n_runs=n_runs)


if __name__ == "__main__":
    main(sys.argv[1:])
