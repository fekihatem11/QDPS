"""Central path constants for the QDPS package, anchored at the package dir."""
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]          # .../TCP/qdps

DATASETS = PKG_ROOT / "datasets"
FAULT_CLUSTERS = DATASETS / "fault_clusters"
FEATURES_FOR_SELECTION = DATASETS / "features_for_selection"
BASELINE_RESULTS = DATASETS / "baseline_results"

# ONE results root; each experiment gets its own subfolder.
RESULTS_DIR = PKG_ROOT / "results"
FDR_RESULTS_DIR = RESULTS_DIR / "fdr"                # QDPS FDR experiment runs
SETS_RESULTS_DIR = RESULTS_DIR / "sets_baseline"     # SETS baseline FDR runs
DOCS_DIR = PKG_ROOT / "docs"

# RQ4 retraining experiment assets (vendored from the SETS replication package).
# Override the root via QDPS_RETRAIN_DIR (e.g. Narval $SCRATCH) without code changes.
RETRAIN_DIR = Path(os.environ.get("QDPS_RETRAIN_DIR", DATASETS / "retrain"))
PRETRAINED_MODELS = RETRAIN_DIR / "pretrained"     # model_{data}_{model}.h5
RETRAIN_SPLITS = RETRAIN_DIR / "splits"            # {data}_{model}.pkl  (the test pool T)
RAW_DATA_DIR = RETRAIN_DIR / "raw"                 # SVHN .mat / Fruit .npy / TinyImageNet images

# Retraining-experiment outputs, one subfolder per part:
#   subjects/  durable per-subject store (<subject>.json, accumulated across runs)
#   runs/      timestamped full-run snapshots (RETRAIN_<ts>/)
#   tables/    generated comparison tables
RETRAIN_RESULTS_DIR = RESULTS_DIR / "retrain"
RETRAIN_SUBJECTS_DIR = RETRAIN_RESULTS_DIR / "subjects"
RETRAIN_RUNS_DIR = RETRAIN_RESULTS_DIR / "runs"
RETRAIN_TABLES_DIR = RETRAIN_RESULTS_DIR / "tables"
