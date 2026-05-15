"""Per-instance fault clustering for the robustness pipeline.

Ports `SETS/Source_code/cluster.py` to operate per (subject, seed). For one
trained instance:

    total_features = vstack(mis_features_test, mis_features_train_subset)

where `mis_features_*` are VGG16 block5_conv3 features for this instance's
misclassifications. Then 2-stage UMAP -> augment with scaled (truth, pred)
columns -> HDBSCAN(min_cluster_size=5). The test slice is contiguous at
the start so cluster_results.npy[0:n_mis_test] is the test-side clustering
that downstream FDR code consumes.

Inputs (must exist):
    instances/seed_<n>/predictions/{output_probability_train.npy,
                                     output_probability_test.npy,
                                     mis_index_train.npy,
                                     mis_index_test.npy}
    features_for_clustering/<dataset>/features_{train,test}_raw.npy

Outputs (in instances/seed_<n>/clusters/):
    cluster_results.npy   int64  (n_mis_total,)  HDBSCAN label per row, -1=noise
    cluster_meta.json                              config + cluster count + provenance

Two modes:

    python cluster.py --subject mnist_LeNet1 --seed 0
        Use LOCKED_CONFIG (set after a one-time sweep), single clustering run.

    python cluster.py --subject mnist_LeNet1 --seed 0 --sweep
        Run the SETS 80-config grid, save sweep_results.json (not
        cluster_results.npy). Mirrors SETS's selection logic:
          (silhouette_umap >= 0.1 OR silhouette_features >= 0.1)
          AND (n_clusters >= --min-clusters if given)
        then rank survivors by silhouette_umap descending. The winning
        config goes into LOCKED_CONFIG; rerun without --sweep for the
        per-instance clustering. --reference-clusters adds an informational
        secondary view of closeness to a SETS-published cluster count.

Determinism: random_state=42 for both UMAP stages; UMAP and HDBSCAN use
n_jobs=1 / core_dist_n_jobs=1 (single-threaded execution is required to
guarantee bit-identical clusterings across reruns).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FEATURES_ROOT = HERE / "features_for_clustering"

# Locked clustering config -- updated after a one-time --sweep on seed 0
# whose result is closest to the SETS reference cluster count (~137 for
# mnist_LeNet1). Before that initial sweep, this is a placeholder taken
# from SETS's first sweep tuple. Re-running predict + cluster with a
# different LOCKED_CONFIG silently changes what the experiment measures,
# so any change here is a methodology change.
LOCKED_CONFIG = {
    "n_components_1": 500,
    "n_components_2": 450,
    "n_neighbors_1":  15,
    "n_neighbors_2":  10,
    "min_dist_1":     0.1,
    "min_dist_2":     0.1,
    "min_cluster_size": 5,
    "random_state":   42,
}

# Same hyperparameter grid that the SETS reference cluster.py sweeps over.
SWEEP_COMPONENT_PAIRS = [(500, 450), (400, 350), (300, 250), (250, 200)]
SWEEP_NEIGHBOR_PAIRS  = [(5, 3), (10, 5), (15, 10), (20, 15), (25, 20)]
SWEEP_MIN_DIST_1      = [0.03, 0.1, 0.25, 0.5]
SWEEP_MIN_DIST_2      = 0.1   # SETS always uses 0.1 for the 2nd stage

# Sweep selection mirrors SETS/Source_code/cluster.py lines 189-201:
#   if (silhouette_umap >= 0.1 OR silhouette_features >= 0.1)
#      AND labels.max() + 2 >= MIN_CLUSTERS_FLOOR:
#       record as candidate
#   then "select the one config clustering that has best Silhouette score"
SILHOUETTE_FLOOR = 0.1

# Reference numbers from SETS for the corresponding fault_clusters/<subject>/
# files. INFORMATIONAL ONLY -- not a selection criterion. We use them in the
# sweep report to show how close each config lands to SETS's published output.
SETS_REFERENCE_N_CLUSTERS = {
    "mnist_LeNet1": 137,
    "mnist_LeNet5": None,
}


def scale_one(x: np.ndarray) -> np.ndarray:
    """SETS's min-max to [0, 1]. Defined on a flat 1D array."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_mnist_labels():
    """Return (y_train_full, y_test) as int64. Same source as predict.py."""
    from tensorflow.keras.datasets import mnist
    (_, y_train_full), (_, y_test) = mnist.load_data()
    return y_train_full.astype(np.int64), y_test.astype(np.int64)


def build_total_matrix(subject: str, seed: int):
    """Assemble the (n_mis_total, 4608) feature matrix + (n_mis_total,) truth/pred
    arrays from the predict.py outputs and the per-dataset VGG features.

    Returns: dict with keys
        total_features    (n_total, 4608) float32
        y_truth           (n_total,)      int64    raw truth labels
        y_pred            (n_total,)      int64    argmax of softmax
        y_truth_scaled    (n_total,)      float64  SETS scale_one over the FULL
                                                   train + test arrays, then sliced
        y_pred_scaled     (n_total,)      float64  same recipe
        n_mis_test, n_mis_train_subset    -- so consumers can split the slices
        meta              -- the predict-side meta.json
    """
    inst_dir = HERE / subject / "instances" / f"seed_{seed}"
    pred_dir = inst_dir / "predictions"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"predictions/ not found: {pred_dir}. "
                                f"Run predict.py first.")
    pred_meta = json.loads((pred_dir / "meta.json").read_text())

    prob_train = np.load(pred_dir / "output_probability_train.npy")  # (60000, 10)
    prob_test  = np.load(pred_dir / "output_probability_test.npy")   # (10000, 10)
    mis_index_train = np.load(pred_dir / "mis_index_train.npy")      # global into 60k, pool n mis
    mis_index_test  = np.load(pred_dir / "mis_index_test.npy")       # global into 10k

    pred_train = np.argmax(prob_train, axis=1).astype(np.int64)  # (60000,)
    pred_test  = np.argmax(prob_test,  axis=1).astype(np.int64)  # (10000,)

    y_train_full, y_test = load_mnist_labels()

    # SETS scaling: min-max over the FULL truth and pred arrays, then slice
    # with the mis indices. This is a no-op for 10-class MNIST where both
    # truth and pred span 0..9 -- but match the recipe exactly.
    yt_train_scaled = scale_one(y_train_full)
    yp_train_scaled = scale_one(pred_train)
    yt_test_scaled  = scale_one(y_test)
    yp_test_scaled  = scale_one(pred_test)

    # Pick the dataset folder from the subject name (mnist_LeNet1 -> mnist).
    dataset = subject.split("_", 1)[0].lower()
    feat_train_path = FEATURES_ROOT / dataset / "features_train_raw.npy"
    feat_test_path  = FEATURES_ROOT / dataset / "features_test_raw.npy"
    if not feat_train_path.exists() or not feat_test_path.exists():
        raise FileNotFoundError(
            f"VGG features missing under {FEATURES_ROOT / dataset}. "
            f"Run extract_vgg_features.py --dataset {dataset} first.")

    # mmap to avoid loading the 1.1 GB train file into memory; the slice we
    # actually use is small (~7000 rows -> ~130 MB).
    features_train_mm = np.load(feat_train_path, mmap_mode="r")
    features_test_mm  = np.load(feat_test_path,  mmap_mode="r")

    mis_features_test  = np.asarray(features_test_mm[mis_index_test],  dtype=np.float32)
    mis_features_train = np.asarray(features_train_mm[mis_index_train], dtype=np.float32)

    # SETS order: test rows first, then train rows.
    total_features = np.vstack([mis_features_test, mis_features_train])
    y_truth        = np.concatenate([y_test[mis_index_test], y_train_full[mis_index_train]])
    y_pred         = np.concatenate([pred_test[mis_index_test], pred_train[mis_index_train]])
    y_truth_scaled = np.concatenate([yt_test_scaled[mis_index_test], yt_train_scaled[mis_index_train]])
    y_pred_scaled  = np.concatenate([yp_test_scaled[mis_index_test], yp_train_scaled[mis_index_train]])

    return {
        "total_features": total_features,
        "y_truth": y_truth,
        "y_pred": y_pred,
        "y_truth_scaled": y_truth_scaled,
        "y_pred_scaled":  y_pred_scaled,
        "n_mis_test":          int(len(mis_index_test)),
        "n_mis_train_subset":  int(len(mis_index_train)),
        "feat_train_path": str(feat_train_path.relative_to(REPO_ROOT)),
        "feat_test_path":  str(feat_test_path.relative_to(REPO_ROOT)),
        "pred_meta": pred_meta,
        "inst_dir": inst_dir,
    }


def cluster_once(total_features: np.ndarray,
                 y_truth_scaled: np.ndarray,
                 y_pred_scaled:  np.ndarray,
                 config: dict):
    """Run 2-stage UMAP + HDBSCAN with `config`. Returns (labels, u_aug, timing)."""
    import umap.umap_ as umap_  # noqa: import-during-call is intentional
    import hdbscan

    random_state = int(config["random_state"])

    t_u1 = time.time()
    u1 = umap_.UMAP(
        n_components=int(config["n_components_1"]),
        n_neighbors=int(config["n_neighbors_1"]),
        min_dist=float(config["min_dist_1"]),
        random_state=random_state,
        n_jobs=1,
    ).fit_transform(total_features)
    t_u1 = time.time() - t_u1

    t_u2 = time.time()
    u2 = umap_.UMAP(
        n_components=int(config["n_components_2"]),
        n_neighbors=int(config["n_neighbors_2"]),
        min_dist=float(config["min_dist_2"]),
        random_state=random_state,
        n_jobs=1,
    ).fit_transform(u1)
    t_u2 = time.time() - t_u2

    u_aug = np.column_stack([u2, y_truth_scaled, y_pred_scaled])

    t_hd = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(config["min_cluster_size"]),
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(u_aug)
    t_hd = time.time() - t_hd

    return labels, u_aug, {"umap1_s": t_u1, "umap2_s": t_u2, "hdbscan_s": t_hd}


def cluster_summary(labels: np.ndarray) -> dict:
    n_total = int(len(labels))
    n_noise = int((labels == -1).sum())
    real_labels = labels[labels >= 0]
    n_clusters = int(len(np.unique(real_labels))) if real_labels.size else 0
    return {
        "n_total": n_total,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_frac": n_noise / n_total if n_total else 0.0,
    }


def maybe_silhouette(X: np.ndarray, labels: np.ndarray, sample: int = 3000):
    """Sampled silhouette to keep cost bounded. Returns float or None.

    Pass `X` = the data the silhouette is computed against (u_aug for
    silhouette_umap; total_features for silhouette_features).
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None
    real = labels >= 0
    if real.sum() < 10 or len(np.unique(labels[real])) < 2:
        return None
    if real.sum() > sample:
        idx = np.random.default_rng(0).choice(np.where(real)[0], size=sample, replace=False)
    else:
        idx = np.where(real)[0]
    try:
        return float(silhouette_score(X[idx], labels[idx]))
    except Exception:
        return None


def run_locked(subject: str, seed: int, config: dict, force: bool) -> int:
    """Single clustering run with `config`. Writes cluster_results.npy + meta."""
    bundle = build_total_matrix(subject, seed)
    out_dir = bundle["inst_dir"] / "clusters"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "cluster_results.npy"
    meta_path    = out_dir / "cluster_meta.json"
    if results_path.exists() and meta_path.exists() and not force:
        print(f"[cluster] {subject} seed={seed}: outputs exist, skipping (use --force)")
        return 0

    n = len(bundle["total_features"])
    print(f"[cluster] {subject} seed={seed}: "
          f"input matrix {bundle['total_features'].shape}  "
          f"(test={bundle['n_mis_test']}, train={bundle['n_mis_train_subset']})")
    print(f"[cluster] config: {config}")

    labels, u_aug, timing = cluster_once(
        bundle["total_features"],
        bundle["y_truth_scaled"],
        bundle["y_pred_scaled"],
        config,
    )

    summary = cluster_summary(labels)
    silhouette_umap     = maybe_silhouette(u_aug, labels)
    silhouette_features = maybe_silhouette(bundle["total_features"], labels)

    print(f"[cluster] result: {summary['n_clusters']} clusters, "
          f"{summary['n_noise']} noise ({summary['noise_frac']:.1%}), "
          f"silh_umap={silhouette_umap}, silh_feat={silhouette_features}  "
          f"(umap1 {timing['umap1_s']:.1f}s, umap2 {timing['umap2_s']:.1f}s, "
          f"hdbscan {timing['hdbscan_s']:.1f}s)")

    np.save(results_path, labels.astype(np.int64))

    cluster_meta = {
        "subject": subject,
        "seed": seed,
        "config": config,
        "summary": summary,
        "silhouette_umap_sampled":     silhouette_umap,
        "silhouette_features_sampled": silhouette_features,
        "timing_s": timing,
        "n_mis_test":         bundle["n_mis_test"],
        "n_mis_train_subset": bundle["n_mis_train_subset"],
        "vstack_order": "test slice first, then train_subset slice",
        "test_slice": [0, bundle["n_mis_test"]],
        "train_slice": [bundle["n_mis_test"],
                        bundle["n_mis_test"] + bundle["n_mis_train_subset"]],
        "feat_train_path": bundle["feat_train_path"],
        "feat_test_path":  bundle["feat_test_path"],
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        import umap
        import hdbscan
        cluster_meta["umap_learn"] = umap.__version__
        cluster_meta["hdbscan"]    = hdbscan.__version__
    except ImportError:
        pass
    meta_path.write_text(json.dumps(cluster_meta, indent=2))

    print(f"[cluster] wrote {results_path.name} + {meta_path.name}")
    return 0


def run_sweep(subject: str, seed: int, min_clusters: int | None,
              reference_clusters: int | None, force: bool) -> int:
    """Sweep SETS's 80-config grid. Writes sweep_results.json. Does NOT
    write cluster_results.npy -- pick a config, set LOCKED_CONFIG, rerun.

    Selection mirrors SETS/Source_code/cluster.py:
      pass = (silhouette_umap >= 0.1 OR silhouette_features >= 0.1)
             AND (min_clusters is None OR n_clusters >= min_clusters)
      then rank survivors by silhouette_umap descending and pick best.

    `reference_clusters` is informational only -- a secondary view in the
    printed report showing each config's distance to a SETS-published count.
    """
    bundle = build_total_matrix(subject, seed)
    out_dir = bundle["inst_dir"] / "clusters"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = out_dir / "sweep_results.json"
    if sweep_path.exists() and not force:
        print(f"[cluster] sweep_results.json exists at {sweep_path}, "
              f"skipping (use --force)")
        return 0

    configs = []
    for (nc1, nc2) in SWEEP_COMPONENT_PAIRS:
        for (nn1, nn2) in SWEEP_NEIGHBOR_PAIRS:
            for md1 in SWEEP_MIN_DIST_1:
                configs.append({
                    "n_components_1": nc1, "n_components_2": nc2,
                    "n_neighbors_1":  nn1, "n_neighbors_2":  nn2,
                    "min_dist_1":     md1, "min_dist_2":     SWEEP_MIN_DIST_2,
                    "min_cluster_size": 5, "random_state":    42,
                })
    print(f"[cluster] sweep: {len(configs)} configs, "
          f"silhouette_floor={SILHOUETTE_FLOOR}, "
          f"min_clusters={min_clusters}, "
          f"reference_clusters={reference_clusters}")

    results = []
    t_total = time.time()
    for idx, cfg in enumerate(configs, 1):
        print(f"[cluster] [{idx}/{len(configs)}] {cfg}")
        try:
            labels, u_aug, timing = cluster_once(
                bundle["total_features"],
                bundle["y_truth_scaled"],
                bundle["y_pred_scaled"],
                cfg,
            )
            summary = cluster_summary(labels)
            silhouette_umap     = maybe_silhouette(u_aug, labels)
            silhouette_features = maybe_silhouette(bundle["total_features"], labels)

            # SETS criterion:
            #   (silhouette_umap >= 0.1 OR silhouette_features >= 0.1)
            #   AND optional min_clusters floor
            passes_silhouette = (
                (silhouette_umap is not None and silhouette_umap >= SILHOUETTE_FLOOR)
                or
                (silhouette_features is not None and silhouette_features >= SILHOUETTE_FLOOR)
            )
            passes_min_clusters = (
                min_clusters is None or summary["n_clusters"] >= min_clusters
            )
            passes_filter = passes_silhouette and passes_min_clusters

            distance_to_reference = (
                abs(summary["n_clusters"] - reference_clusters)
                if reference_clusters is not None else None
            )

            print(f"  -> {summary['n_clusters']:4d} clusters, "
                  f"{summary['n_noise']:4d} noise ({summary['noise_frac']:.1%}), "
                  f"silh_umap={silhouette_umap}, silh_feat={silhouette_features}, "
                  f"passes={passes_filter}")

            results.append({
                "config": cfg,
                "summary": summary,
                "silhouette_umap_sampled":     silhouette_umap,
                "silhouette_features_sampled": silhouette_features,
                "passes_silhouette_floor":     passes_silhouette,
                "passes_min_clusters":         passes_min_clusters,
                "passes_filter":               passes_filter,
                "distance_to_reference":       distance_to_reference,
                "timing_s": timing,
                "status": "ok",
            })
        except Exception as e:  # noqa: broad-except is fine for a sweep
            print(f"  -> FAILED: {type(e).__name__}: {e}")
            results.append({
                "config": cfg,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            })

    total_elapsed = time.time() - t_total

    # Primary ranking: SETS criterion -- among configs passing the filter,
    # sort by silhouette_umap descending. Configs that fail the filter sort
    # to the bottom but are still recorded for transparency.
    def primary_key(r):
        if r["status"] != "ok":
            return (1, -10.0)
        if not r["passes_filter"]:
            return (1, -(r["silhouette_umap_sampled"] or -10.0))
        return (0, -(r["silhouette_umap_sampled"] or 0.0))
    ranked_primary = sorted(results, key=primary_key)

    # Secondary (informational) view: closest to SETS reference cluster count.
    if reference_clusters is not None:
        def secondary_key(r):
            if r["status"] != "ok":
                return (10**9, 0.0)
            return (r["distance_to_reference"] or 10**9,
                    -(r["silhouette_umap_sampled"] or 0.0))
        ranked_by_reference = sorted(results, key=secondary_key)
    else:
        ranked_by_reference = None

    survivors = [r for r in ranked_primary if r["status"] == "ok" and r["passes_filter"]]
    winner = survivors[0] if survivors else None

    def fmt_silh(x):
        return f"{x:.3f}" if x is not None else "  None"

    def fmt_line(r):
        s = r["summary"]
        line = (f"  clusters={s['n_clusters']:4d}  "
                f"noise={s['n_noise']:4d}({s['noise_frac']:.1%})  "
                f"silh_umap={fmt_silh(r.get('silhouette_umap_sampled'))}  "
                f"silh_feat={fmt_silh(r.get('silhouette_features_sampled'))}  ")
        if r.get("distance_to_reference") is not None:
            line += f"dist_ref={r['distance_to_reference']:3d}  "
        line += f"pass={'Y' if r.get('passes_filter') else 'N'}  config={r['config']}"
        return line

    print(f"[cluster] sweep complete in {total_elapsed:.1f}s "
          f"({len(survivors)}/{len(results)} survive the SETS filter)")
    print(f"[cluster] top 5 BY SETS CRITERION (filter passing, "
          f"ranked by silhouette_umap):")
    n_shown = 0
    for r in ranked_primary:
        if r["status"] != "ok" or not r["passes_filter"]:
            continue
        print(fmt_line(r))
        n_shown += 1
        if n_shown >= 5:
            break
    if n_shown == 0:
        print("  (none -- no configs passed the filter; "
              "consider lowering --min-clusters or inspect sweep_results.json)")
    if winner is not None:
        print(f"[cluster] WINNER: {winner['config']}  "
              f"(silh_umap={fmt_silh(winner['silhouette_umap_sampled'])}, "
              f"clusters={winner['summary']['n_clusters']})")

    if ranked_by_reference is not None:
        print(f"[cluster] top 5 BY DISTANCE TO REFERENCE "
              f"({reference_clusters} clusters; informational only):")
        for r in ranked_by_reference[:5]:
            if r["status"] != "ok":
                continue
            print(fmt_line(r))

    payload = {
        "subject": subject,
        "seed": seed,
        "selection_criterion": "SETS: (silh_umap>=0.1 OR silh_feat>=0.1) "
                                "AND (n_clusters>=min_clusters if set); "
                                "rank survivors by silh_umap desc",
        "silhouette_floor":   SILHOUETTE_FLOOR,
        "min_clusters":       min_clusters,
        "reference_clusters": reference_clusters,
        "n_survivors":        len(survivors),
        "winner":             (winner["config"] if winner else None),
        "total_elapsed_s":    total_elapsed,
        "n_configs":          len(configs),
        "results_ranked":     ranked_primary,
        "git_commit":         git_commit(),
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python":             sys.version.split()[0],
        "numpy":              np.__version__,
        "platform":           platform.platform(),
    }
    sweep_path.write_text(json.dumps(payload, indent=2))
    print(f"[cluster] wrote {sweep_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Per-instance fault clustering.")
    p.add_argument("--subject", required=True, choices=["mnist_LeNet1", "mnist_LeNet5"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sweep", action="store_true",
                   help="Run the SETS 80-config sweep; write sweep_results.json only")
    p.add_argument("--min-clusters", type=int, default=None,
                   help="Sweep filter floor on n_clusters (default: no floor). "
                        "SETS's example uses 200, but that's Fruit-360-specific.")
    p.add_argument("--reference-clusters", type=int, default=None,
                   help="Informational only: SETS-published cluster count for "
                        "this subject; the sweep report shows distance to it. "
                        "(default: SETS_REFERENCE_N_CLUSTERS for the subject)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing cluster_results.npy / sweep_results.json")
    # Optional overrides of individual LOCKED_CONFIG fields (rare; mostly for
    # validating a hand-picked config without editing source).
    p.add_argument("--n-components-1", type=int)
    p.add_argument("--n-components-2", type=int)
    p.add_argument("--n-neighbors-1",  type=int)
    p.add_argument("--n-neighbors-2",  type=int)
    p.add_argument("--min-dist-1",     type=float)
    p.add_argument("--min-dist-2",     type=float)
    p.add_argument("--min-cluster-size", type=int)
    p.add_argument("--random-state",   type=int)
    args = p.parse_args()

    if args.sweep:
        reference = args.reference_clusters
        if reference is None:
            reference = SETS_REFERENCE_N_CLUSTERS.get(args.subject)
        return run_sweep(args.subject, args.seed,
                         min_clusters=args.min_clusters,
                         reference_clusters=reference,
                         force=args.force)

    config = dict(LOCKED_CONFIG)
    overrides = {
        "n_components_1":   args.n_components_1,
        "n_components_2":   args.n_components_2,
        "n_neighbors_1":    args.n_neighbors_1,
        "n_neighbors_2":    args.n_neighbors_2,
        "min_dist_1":       args.min_dist_1,
        "min_dist_2":       args.min_dist_2,
        "min_cluster_size": args.min_cluster_size,
        "random_state":     args.random_state,
    }
    for k, v in overrides.items():
        if v is not None:
            config[k] = v
    return run_locked(args.subject, args.seed, config, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
