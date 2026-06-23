# QDPS Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `qdps/` from a flat pile of scripts into the importable, editable-installed Python package `qdps`, with `qdps.py` the only loose source module at root.

**Architecture:** `qdps/` becomes the package. Code is grouped into `io/` (data access), `experiments/` (runners), `analysis/` (result readers/tables); data assets into `datasets/`; markdown into `docs/`. All paths are centralized in `qdps/io/paths.py` anchored at the package dir. Scripts run as modules (`python -m qdps.experiments.run_experiment`) after a one-time `pip install -e .`.

**Tech Stack:** Python 3.10+, numpy/scipy/scikit-learn, setuptools editable install, GNU Make.

## Global Constraints

- This is a **pure reorganization** — no algorithm, FDR convention, or baseline-number changes. Post-refactor FDRs must be numerically identical.
- All file moves use `git mv` to preserve history (renames show as `R`, not delete+add).
- Absolute package imports only: `from qdps import select`, `from qdps.io.loader import …`, `from qdps.io.paths import …`. No flat `import data_loader` / `import qdps` as top-level modules.
- Path constants live ONLY in `qdps/io/paths.py`, anchored via `Path(__file__).resolve().parents[1]`. No new `os.path.dirname(__file__)` data-path logic anywhere else.
- Avoid folder names reserved by `.gitignore`: `lib/`, `build/`, `dist/`.
- Work in the project venv: `source /Users/artem/Desktop/TCP/.venv/bin/activate`.
- `load_subject` returns a **dict** with keys including `features`, `total_faults`, `index`, `output_probability`.
- Out of scope, do not touch: `robustness/`, `methods/`, `scripts/`, `SETS/`. Do not restore the missing `datasets/baseline_results/`.

---

### Task 1: Package skeleton, pyproject, editable install

Make `qdps/` an importable package and install it editable so `from qdps import select` works. `qdps.py` is unchanged; only a new `__init__.py` and root `pyproject.toml` are added.

**Files:**
- Create: `/Users/artem/Desktop/TCP/pyproject.toml`
- Create: `/Users/artem/Desktop/TCP/qdps/__init__.py`

**Interfaces:**
- Produces: package `qdps` exporting `select`, `METHOD_NAME`, `ADAPTIVE_DEFAULTS`, `get_adaptive_defaults`, `maxp_score` (re-exported from `qdps.qdps`).

- [ ] **Step 1: Create `pyproject.toml` at TCP root**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "qdps"
version = "0.1.0"
description = "QDPS — Quality-Diversity DPP Selection for DNN test case prioritization"
requires-python = ">=3.10"
dependencies = ["numpy", "scipy", "scikit-learn"]

[tool.setuptools.packages.find]
include = ["qdps*"]
```

- [ ] **Step 2: Create `qdps/__init__.py`**

```python
"""QDPS — Quality-Diversity DPP Selection (importable package)."""
from .qdps import (
    select,
    METHOD_NAME,
    ADAPTIVE_DEFAULTS,
    get_adaptive_defaults,
    maxp_score,
)

__all__ = [
    "select",
    "METHOD_NAME",
    "ADAPTIVE_DEFAULTS",
    "get_adaptive_defaults",
    "maxp_score",
]
```

- [ ] **Step 3: Editable install into the venv**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && pip install -e .
```
Expected: ends with `Successfully installed qdps-0.1.0`.

- [ ] **Step 4: Verify the package imports**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && python -c "from qdps import select, METHOD_NAME; print(METHOD_NAME)"
```
Expected: prints `QDPS`.

- [ ] **Step 5: Commit**

```bash
cd /Users/artem/Desktop/TCP && git add pyproject.toml qdps/__init__.py && git commit -m "Make qdps an editable-installed package"
```

---

### Task 2: Create `io/` layer — `paths.py` and `loader.py`

Centralize all paths and move `data_loader.py` into the package, dropping the `../..` features fallback hack.

**Files:**
- Create: `/Users/artem/Desktop/TCP/qdps/io/__init__.py`
- Create: `/Users/artem/Desktop/TCP/qdps/io/paths.py`
- Move: `qdps/data_loader.py` → `qdps/io/loader.py`
- Modify: `qdps/io/loader.py` (imports + path logic)

**Interfaces:**
- Produces: `qdps.io.paths` exposing `PKG_ROOT, FAULT_CLUSTERS, FEATURES_FOR_SELECTION, BASELINE_RESULTS, RESULTS_DIR, SETS_RESULTS_DIR, DOCS_DIR` (all `pathlib.Path`).
- Produces: `qdps.io.loader` exposing `load_subject(data_name, model_name, base_path=FAULT_CLUSTERS) -> dict`, `compute_fdr(...)`, `DATA_MODEL_PAIRS`, `FOLDER_MAP`, `FEATURES_FILE_MAP`.

- [ ] **Step 1: Create `qdps/io/__init__.py`** (empty marker)

```python
```

- [ ] **Step 2: Create `qdps/io/paths.py`**

```python
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
```

- [ ] **Step 3: Move `data_loader.py` into `io/`**

```bash
cd /Users/artem/Desktop/TCP && git mv qdps/data_loader.py qdps/io/loader.py
```

- [ ] **Step 4: Update the loader's import header**

In `qdps/io/loader.py`, change the top import block from:
```python
import numpy as np
import pickle
import os
```
to:
```python
import numpy as np
import pickle
import os

from qdps.io.paths import FAULT_CLUSTERS, FEATURES_FOR_SELECTION
```

- [ ] **Step 5: Default `base_path` and drop the `../..` hack**

In `qdps/io/loader.py`, change the signature:
```python
def load_subject(data_name, model_name, base_path):
```
to:
```python
def load_subject(data_name, model_name, base_path=FAULT_CLUSTERS):
```

Then replace the features-fallback block:
```python
            # Selection-kernel features live in qdps/features_for_selection/
            # (one level up from base_path = qdps/fault_clusters/<subject>/).
            alt_path = os.path.join(
                os.path.dirname(base_path), "..", "features_for_selection", feat_filename
            )
            if os.path.exists(alt_path):
                features = np.load(alt_path)
```
with:
```python
            # Selection-kernel features live in qdps/datasets/features_for_selection/.
            alt_path = FEATURES_FOR_SELECTION / feat_filename
            if alt_path.exists():
                features = np.load(alt_path)
```

- [ ] **Step 6: Verify the loader imports (no disk access yet)**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && python -c "from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS; print(len(DATA_MODEL_PAIRS))"
```
Expected: prints `8`.

- [ ] **Step 7: Commit**

```bash
cd /Users/artem/Desktop/TCP && git add qdps/io && git commit -m "Add qdps.io layer: centralized paths + loader"
```

---

### Task 3: Move data assets into `datasets/`

**Files:**
- Move: `qdps/fault_clusters/` → `qdps/datasets/fault_clusters/`
- Move: `qdps/features_for_selection/` → `qdps/datasets/features_for_selection/`

**Interfaces:**
- Consumes: `qdps.io.paths.FAULT_CLUSTERS`, `FEATURES_FOR_SELECTION` (now resolve to these moved dirs).

- [ ] **Step 1: Create `datasets/` and move both asset dirs**

```bash
cd /Users/artem/Desktop/TCP/qdps && mkdir -p datasets && git mv fault_clusters datasets/fault_clusters && git mv features_for_selection datasets/features_for_selection
```

- [ ] **Step 2: Verify a real load works end-to-end (exercises the features fallback)**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && python -c "
from qdps.io.loader import load_subject
s = load_subject('mnist', 'LeNet1')
print('features:', None if s['features'] is None else s['features'].shape)
print('total_faults:', s['total_faults'], 'n_valid:', len(s['index']))
"
```
Expected: `features:` shows a non-None shape (e.g. `(10000, …)`); `total_faults:` is a positive integer.

- [ ] **Step 3: Commit**

```bash
cd /Users/artem/Desktop/TCP && git add -A qdps/datasets && git commit -m "Move data assets under qdps/datasets/"
```

---

### Task 4: Move runners into `experiments/`

**Files:**
- Create: `qdps/experiments/__init__.py`
- Move: `run_experiment.py`, `run_single_subject.py`, `run_sets_baseline.py` → `qdps/experiments/`
- Modify: each moved file (imports + path constants)

**Interfaces:**
- Consumes: `qdps.select`, `qdps.METHOD_NAME`, `qdps.io.loader.{load_subject,compute_fdr,DATA_MODEL_PAIRS}`, `qdps.io.paths.{RESULTS_DIR,SETS_RESULTS_DIR}`.

- [ ] **Step 1: Create the dir, marker, and move the three runners**

```bash
cd /Users/artem/Desktop/TCP/qdps && mkdir -p experiments && : > experiments/__init__.py && git mv run_experiment.py experiments/ && git mv run_single_subject.py experiments/ && git mv run_sets_baseline.py experiments/ && git add experiments/__init__.py
```

- [ ] **Step 2: Fix imports/paths in `experiments/run_experiment.py`**

Change:
```python
from data_loader import DATA_MODEL_PAIRS, load_subject, compute_fdr
from qdps import select as qdps_select, METHOD_NAME
```
to:
```python
from qdps import select as qdps_select, METHOD_NAME
from qdps.io.loader import DATA_MODEL_PAIRS, load_subject, compute_fdr
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA_PATH, RESULTS_DIR
```
Then delete the now-redundant local definitions:
```python
BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "fault_clusters")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
```

- [ ] **Step 3: Fix imports/paths in `experiments/run_single_subject.py`**

Change:
```python
from data_loader import load_subject, compute_fdr
from qdps import select as qdps_select
```
to:
```python
from qdps import select as qdps_select
from qdps.io.loader import load_subject, compute_fdr
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA
```
Then delete:
```python
BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")
```

- [ ] **Step 4: Fix imports/paths in `experiments/run_sets_baseline.py`**

Change:
```python
from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
```
to:
```python
from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, SETS_RESULTS_DIR as RESULTS_DIR
```
Then delete:
```python
BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "sets_results")
```
(Note: `run_sets_baseline.py` may import its SETS selector from elsewhere; leave that import untouched.)

- [ ] **Step 5: Smoke-test a single-subject run (1 run, k=100)**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && python -m qdps.experiments.run_single_subject mnist_LeNet1 1 100
```
Expected: prints per-budget FDR/time output for `mnist_LeNet1` at k=100 without error.

- [ ] **Step 6: Verify the other two import cleanly**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && python -c "import qdps.experiments.run_experiment, qdps.experiments.run_sets_baseline; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 7: Commit**

```bash
cd /Users/artem/Desktop/TCP && git add -A qdps/experiments && git commit -m "Move runners under qdps/experiments/"
```

---

### Task 5: Move result readers/tables into `analysis/`

**Files:**
- Create: `qdps/analysis/__init__.py`
- Move: `compare_results.py`, `statistical_test.py`, `generate_fdr_table.py`, `generate_time_table.py` → `qdps/analysis/`
- Modify: each moved file (imports + path constants; `.tex` outputs → `DOCS_DIR`)

**Interfaces:**
- Consumes: `qdps.select`, `qdps.io.loader.{load_subject,compute_fdr,DATA_MODEL_PAIRS}`, `qdps.io.paths.{FAULT_CLUSTERS,BASELINE_RESULTS,SETS_RESULTS_DIR,DOCS_DIR}`.

- [ ] **Step 1: Create the dir, marker, and move the four files**

```bash
cd /Users/artem/Desktop/TCP/qdps && mkdir -p analysis && : > analysis/__init__.py && git mv compare_results.py analysis/ && git mv statistical_test.py analysis/ && git mv generate_fdr_table.py analysis/ && git mv generate_time_table.py analysis/ && git add analysis/__init__.py
```

- [ ] **Step 2: Fix `analysis/compare_results.py`**

Change:
```python
from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps import select as qdps_select
```
to:
```python
from qdps import select as qdps_select
from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA
```
Then delete:
```python
BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")
```

- [ ] **Step 3: Fix `analysis/statistical_test.py`**

Change:
```python
from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps import select as qdps_select
```
to:
```python
from qdps import select as qdps_select
from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, BASELINE_RESULTS as BASELINE_DIR
```
Then delete:
```python
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baseline_results")
BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")
```

- [ ] **Step 4: Fix `analysis/generate_fdr_table.py`**

Change:
```python
from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps import select as qdps_select
```
to:
```python
from qdps import select as qdps_select
from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, DOCS_DIR
```
Then delete:
```python
BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")
```
And change the output path:
```python
    output_path = os.path.join(os.path.dirname(__file__), "table_qdps_vs_sets.tex")
```
to:
```python
    output_path = os.path.join(DOCS_DIR, "table_qdps_vs_sets.tex")
```

- [ ] **Step 5: Fix `analysis/generate_time_table.py`**

Change:
```python
from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps import select as qdps_select
```
to:
```python
from qdps import select as qdps_select
from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, SETS_RESULTS_DIR, DOCS_DIR
```
Then delete:
```python
BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")
SETS_RESULTS = os.path.join(os.path.dirname(__file__), "sets_results", "sets_results.json")
```
And add, right after the imports, the file reference the code uses:
```python
SETS_RESULTS = os.path.join(SETS_RESULTS_DIR, "sets_results.json")
```
And change the output path:
```python
    output_path = os.path.join(os.path.dirname(__file__), "table_time_qdps_vs_sets.tex")
```
to:
```python
    output_path = os.path.join(DOCS_DIR, "table_time_qdps_vs_sets.tex")
```

- [ ] **Step 6: Smoke-test the comparison (runs QDPS itself; needs no prior results)**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && python -m qdps.analysis.compare_results
```
Expected: prints the QDPS-vs-baselines comparison table without error.

- [ ] **Step 7: Verify the other three import cleanly**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && python -c "import qdps.analysis.statistical_test, qdps.analysis.generate_fdr_table, qdps.analysis.generate_time_table; print('ok')"
```
Expected: prints `ok`. (Note: `statistical_test` *running* needs `datasets/baseline_results/`, which is absent — importing it is the only check here.)

- [ ] **Step 8: Commit**

```bash
cd /Users/artem/Desktop/TCP && git add -A qdps/analysis && git commit -m "Move result readers/tables under qdps/analysis/"
```

---

### Task 6: Move markdown to `docs/`, rewrite Makefile, remove `__pycache__`

**Files:**
- Move: `BASELINES.md`, `TABLE_FDR.md`, `TABLE_TIME.md` → `qdps/docs/`
- Modify: `qdps/Makefile`
- Delete: `qdps/__pycache__/`

- [ ] **Step 1: Move the three markdown files**

```bash
cd /Users/artem/Desktop/TCP/qdps && mkdir -p docs && git mv BASELINES.md docs/ && git mv TABLE_FDR.md docs/ && git mv TABLE_TIME.md docs/
```

- [ ] **Step 2: Remove the stale `__pycache__`**

```bash
cd /Users/artem/Desktop/TCP/qdps && rm -rf __pycache__
```

- [ ] **Step 3: Rewrite the Makefile recipe bodies**

In `qdps/Makefile`, replace each recipe command (the `$(PYTHON) <script>.py` line) as follows. Keep the `## …` comment lines and target names unchanged.

```make
compare:
	$(PYTHON) -m qdps.analysis.compare_results

run:
	$(PYTHON) -m qdps.experiments.run_experiment 30

quick:
	$(PYTHON) -m qdps.experiments.run_experiment 5

single:
	$(PYTHON) -m qdps.experiments.run_experiment 1

subject:
	$(PYTHON) -m qdps.experiments.run_single_subject $(S) $(or $(N),30) $(if $(K),$(K))

table:
	$(PYTHON) -m qdps.analysis.generate_fdr_table

sets:
	$(PYTHON) -m qdps.experiments.run_sets_baseline

stat:
	$(PYTHON) -m qdps.analysis.statistical_test
```
Leave `PYTHON = python3`, the `help:` recipe, `clean:` (`rm -rf results/`), and `.PHONY` unchanged. (Step also fixes the pre-existing bug where `table:` called a nonexistent `generate_table.py`.)

- [ ] **Step 4: Verify `make compare` runs through the Makefile**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && make -C qdps compare
```
Expected: same comparison output as Task 5 Step 6, no error.

- [ ] **Step 5: Verify `make subject` runs**

Run:
```bash
cd /Users/artem/Desktop/TCP && source .venv/bin/activate && make -C qdps subject S=mnist_LeNet1 N=1 K=100
```
Expected: single-subject FDR/time output for `mnist_LeNet1` at k=100.

- [ ] **Step 6: Commit**

```bash
cd /Users/artem/Desktop/TCP && git add -A qdps/docs qdps/Makefile && git commit -m "Move markdown to qdps/docs/, rewrite Makefile to module invocation"
```

---

### Task 7: Update CLAUDE.md and .gitignore

Bring the docs in line with the new layout so they don't go stale.

**Files:**
- Modify: `/Users/artem/Desktop/TCP/CLAUDE.md`
- Modify: `/Users/artem/Desktop/TCP/.gitignore`

- [ ] **Step 1: Update the "Common commands" block in CLAUDE.md**

The current block documents `make compare/run/quick/single/sets/stat/subject` run "from inside `qdps/`". Replace the surrounding prose so it reads: run `make -C qdps <target>` from the repo root (or `make <target>` from inside `qdps/`) after a one-time `pip install -e .`. Update the generic-runner lines:
```
python scripts/run_experiment.py <method_module> [n_runs]   # e.g. dpp_greedy, qdps, sets_baseline
python scripts/compare_results.py                           # QDPS vs all paper baselines
```
Leave those `scripts/` lines as-is (they belong to the separate `methods/`+`scripts/` framework, unchanged by this work).

- [ ] **Step 2: Update the data-flow paths in CLAUDE.md**

In the "Data flow" section, replace:
- `qdps/fault_clusters/<subject>/` → `qdps/datasets/fault_clusters/<subject>/`
- `qdps/features_for_selection/features_test_<subject>.npy` → `qdps/datasets/features_for_selection/features_test_<subject>.npy`
- `qdps/baseline_results/{SETS,DeepGD,RS}/<subject>/` → `qdps/datasets/baseline_results/{SETS,DeepGD,RS}/<subject>/`
- In the `data_loader.load_subject` fallback sentence, `features_test.npy` lookup now resolves via `qdps/io/paths.py` (`FEATURES_FOR_SELECTION`). Update the wording to reference `qdps.io.loader` / `qdps.io.paths` instead of `data_loader.py`.
- Update the `FOLDER_MAP in data_loader.py` reference → `FOLDER_MAP in qdps/io/loader.py`.

- [ ] **Step 3: Update the BASELINES path and duplication convention in CLAUDE.md**

- Any reference to `qdps/BASELINES.md` → `qdps/docs/BASELINES.md`.
- In the "Conventions" bullet listing the four files that duplicate baseline FDRs, update the two `qdps/` paths: `qdps/run_single_subject.py` → `qdps/experiments/run_single_subject.py`, and `qdps/compare_results.py` → `qdps/analysis/compare_results.py`. (`qdps/BASELINES.md` → `qdps/docs/BASELINES.md`; `scripts/compare_results.py` unchanged.)

- [ ] **Step 4: Add egg-info to .gitignore**

Append to `/Users/artem/Desktop/TCP/.gitignore` under the `# Python` section:
```
*.egg-info/
```

- [ ] **Step 5: Verify no stale paths remain in CLAUDE.md**

Run:
```bash
cd /Users/artem/Desktop/TCP && grep -nE "qdps/(fault_clusters|features_for_selection|baseline_results|BASELINES\.md|run_single_subject\.py|compare_results\.py)" CLAUDE.md || echo "clean"
```
Expected: prints `clean` (the `scripts/compare_results.py` line, if present, is fine — it is not prefixed `qdps/`).

- [ ] **Step 6: Verify git shows renames and the tree is clean**

Run:
```bash
cd /Users/artem/Desktop/TCP && git status --short qdps && echo "--- root files in qdps/ ---" && ls /Users/artem/Desktop/TCP/qdps/*.py
```
Expected: working tree clean for `qdps/` (all prior tasks committed); the only loose `.py` listed is `qdps.py`.

- [ ] **Step 7: Commit**

```bash
cd /Users/artem/Desktop/TCP && git add CLAUDE.md .gitignore && git commit -m "Update docs and gitignore for qdps package restructure"
```

---

## Final verification (after all tasks)

- [ ] `ls qdps/*.py` lists only `qdps/qdps.py`.
- [ ] `python -c "from qdps import select, METHOD_NAME; from qdps.io.loader import load_subject; print(METHOD_NAME)"` prints `QDPS`.
- [ ] `make -C qdps subject S=mnist_LeNet1 N=1 K=100` runs and prints FDR/time identical to the pre-refactor baseline for that subject (pure reorg — no numeric change).
- [ ] `git log --oneline -8` shows the seven restructure commits; `git status` is clean.
