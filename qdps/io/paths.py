"""Central path constants for the QDPS package, anchored at the package dir."""
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]          # .../TCP/qdps

DATASETS = PKG_ROOT / "datasets"
FAULT_CLUSTERS = DATASETS / "fault_clusters"
FEATURES_FOR_SELECTION = DATASETS / "features_for_selection"
BASELINE_RESULTS = DATASETS / "baseline_results"

RESULTS_DIR = PKG_ROOT / "results"
SETS_RESULTS_DIR = PKG_ROOT / "sets_results"
DOCS_DIR = PKG_ROOT / "docs"
