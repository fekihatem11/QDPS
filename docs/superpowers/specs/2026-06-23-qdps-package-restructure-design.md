# QDPS folder restructure & packaging — design

**Date:** 2026-06-23
**Status:** Approved (pending spec review)
**Scope:** Restructure `qdps/` from a flat pile of scripts into a proper, installable Python package. Code-only reorganization — no algorithm or methodology changes.

## Goal

`qdps/` currently has ~10 loose source/markdown files at its root plus mixed data and output dirs. The flat `from qdps import …` / `from data_loader import …` imports and per-file `os.path.dirname(__file__)` path hacks make it fragile and cluttered.

Target: only `qdps.py` (the algorithm) remains a loose source module at the root of `qdps/`; everything else is grouped into purpose folders, and the whole thing is the importable package `qdps`, installed editable into the project `.venv`.

Non-goals: changing the QDPS algorithm, FDR convention, baseline numbers, the `robustness/` Narval pipeline, the `methods/`+`scripts/` framework, or `SETS/`.

## Decisions (locked)

1. **Proper Python package** (not a sys.path reorg). Import via absolute package paths.
2. **Root `qdps/` IS the package.** `qdps.py` stays the real algorithm module (`qdps.qdps`); `qdps/__init__.py` re-exports its public API so `from qdps import select` still works.
3. **Editable install.** A `pyproject.toml` at TCP repo root; `pip install -e .` once into `.venv`. Run scripts as modules from anywhere: `python -m qdps.experiments.run_experiment`.

## Target layout

```
TCP/
  pyproject.toml              # NEW: [project] name=qdps; find qdps*; deps numpy/scipy/scikit-learn
  qdps/                       # the package `qdps`
    qdps.py                   # THE algorithm — only loose .py at root
    __init__.py               # from .qdps import select, METHOD_NAME, ADAPTIVE_DEFAULTS
    Makefile                  # recipes rewritten to `python -m qdps.<sub>.<mod>`

    io/                       # data-access layer (code)
      __init__.py
      loader.py               # was data_loader.py
      paths.py                # NEW: all path constants, anchored at package root

    experiments/              # things that RUN the method
      __init__.py
      run_experiment.py
      run_single_subject.py
      run_sets_baseline.py

    analysis/                 # things that READ results / make tables
      __init__.py
      compare_results.py
      statistical_test.py
      generate_fdr_table.py
      generate_time_table.py

    datasets/                 # INPUT data assets
      fault_clusters/
      features_for_selection/
      baseline_results/        # currently MISSING from working tree (see Known issues)

    results/                  # OUTPUT: qdps runs (gitignored)
    sets_results/             # OUTPUT: sets baseline (gitignored)

    docs/                     # BASELINES.md, TABLE_FDR.md, TABLE_TIME.md
    robustness/               # UNTOUCHED (Narval pipeline / symlink target)
```

Folder-name rationale: `io/` (data-access *code*) is kept deliberately distinct from `datasets/` (data *assets*) so the two never blur. Avoided names reserved by `.gitignore`: `lib/`, `build/`, `dist/`.

## Import mechanism

- `qdps/__init__.py` re-exports the public API: `from .qdps import select, METHOD_NAME, ADAPTIVE_DEFAULTS` (whatever the algorithm currently exposes). External callers keep `from qdps import select`.
- Submodules use absolute package imports:
  - `from qdps import select, METHOD_NAME`
  - `from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS`
  - `from qdps.io.paths import FAULT_CLUSTERS, RESULTS_DIR, …`
- No remaining flat `import data_loader` / `import qdps` as top-level modules.

## Path centralization

New `qdps/io/paths.py` is the single source of truth, anchored at the package directory:

```python
from pathlib import Path
PKG_ROOT = Path(__file__).resolve().parents[1]          # .../TCP/qdps
FAULT_CLUSTERS         = PKG_ROOT / "datasets" / "fault_clusters"
FEATURES_FOR_SELECTION = PKG_ROOT / "datasets" / "features_for_selection"
BASELINE_RESULTS       = PKG_ROOT / "datasets" / "baseline_results"
RESULTS_DIR            = PKG_ROOT / "results"
SETS_RESULTS_DIR       = PKG_ROOT / "sets_results"
```

This replaces:
- Each script's local `BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")` etc.
- The fragile `os.path.dirname(base_path), "..", "features_for_selection"` fallback inside `data_loader.py` → becomes a direct `FEATURES_FOR_SELECTION` reference.

Because paths are `__file__`-anchored (not CWD-relative), scripts work regardless of where they're launched.

## Per-file changes

| File (old) | New location | Change |
|---|---|---|
| `qdps.py` | `qdps/qdps.py` | unchanged content; now a submodule of package |
| (new) | `qdps/__init__.py` | re-export public API |
| `data_loader.py` | `qdps/io/loader.py` | imports from `qdps.io.paths`; drop `../..` hack |
| (new) | `qdps/io/paths.py` | path constants |
| `run_experiment.py` | `qdps/experiments/run_experiment.py` | package imports + `paths` |
| `run_single_subject.py` | `qdps/experiments/run_single_subject.py` | package imports + `paths` |
| `run_sets_baseline.py` | `qdps/experiments/run_sets_baseline.py` | package imports + `paths` |
| `compare_results.py` | `qdps/analysis/compare_results.py` | package imports + `paths` |
| `statistical_test.py` | `qdps/analysis/statistical_test.py` | package imports + `paths` (uses `BASELINE_RESULTS`) |
| `generate_fdr_table.py` | `qdps/analysis/generate_fdr_table.py` | package imports + `paths` |
| `generate_time_table.py` | `qdps/analysis/generate_time_table.py` | package imports + `paths` |
| `BASELINES.md`, `TABLE_FDR.md`, `TABLE_TIME.md` | `qdps/docs/` | move only |
| `fault_clusters/`, `features_for_selection/` | `qdps/datasets/` | move only |
| `__pycache__/` | — | delete |

All moves use `git mv` to preserve history.

## Makefile

Recipes change from `python3 <script>.py` to module invocation:

```
compare:  python -m qdps.analysis.compare_results
run:      python -m qdps.experiments.run_experiment 30
quick:    python -m qdps.experiments.run_experiment 5
single:   python -m qdps.experiments.run_experiment 1
subject:  python -m qdps.experiments.run_single_subject $(S) $(or $(N),30) $(if $(K),$(K))
sets:     python -m qdps.experiments.run_sets_baseline
stat:     python -m qdps.analysis.statistical_test
table:    python -m qdps.analysis.generate_fdr_table   # fixes existing bug: old recipe called nonexistent generate_table.py
clean:    rm -rf results/
```

The Makefile stays inside `qdps/`; with the editable install, `python -m qdps.…` resolves from any CWD.

## pyproject.toml (TCP root)

Minimal, e.g.:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "qdps"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy", "scipy", "scikit-learn"]

[tool.setuptools.packages.find]
include = ["qdps*"]
```

`pip install -e .` into the project `.venv`. (Deps kept minimal — the heavy TF/Keras pins in `SETS/requirements.txt` are not needed for QDPS selection.)

## Doc & config updates

- `CLAUDE.md`:
  - "Common commands" block → module-invocation forms.
  - Data-flow paths: `qdps/fault_clusters/` → `qdps/datasets/fault_clusters/`, `qdps/features_for_selection/` → `qdps/datasets/features_for_selection/`, `qdps/baseline_results/` → `qdps/datasets/baseline_results/`.
  - `qdps/BASELINES.md` → `qdps/docs/BASELINES.md`.
  - The BASELINES-duplication convention list (4 files) — update the two `qdps/` paths (`compare_results.py`, `run_single_subject.py` now under subfolders).
- `.gitignore`: add `*.egg-info/` (editable install artifact). Existing `qdps/sets_results`, `__pycache__` entries still valid.

## Known issues (flagged, not fixed here)

- `datasets/baseline_results/` is **absent** from the working tree (git shows the whole dir deleted). `analysis/statistical_test.py` reads it. The restructure preserves the *reference* (`BASELINE_RESULTS` path) but does not restore the data — that's a separate decision.
- `methods/qdps.py` is a hand-synced twin of `qdps/qdps.py`. It is gitignored and out of scope; the sync remains a manual convention.

## Verification

After the change:
1. `pip install -e .` succeeds in `.venv`.
2. `python -c "from qdps import select, METHOD_NAME; print(METHOD_NAME)"` works.
3. `python -c "from qdps.io.loader import load_subject"` works.
4. `make -C qdps compare` (or `make subject S=mnist_LeNet1 N=1 K=100`) runs end-to-end and prints FDRs identical to pre-refactor (no numeric change expected — pure reorg).
5. `git status` shows renames (R), not delete+add, for moved files.
```

## Out of scope

`robustness/`, `methods/`, `scripts/`, `SETS/`, algorithm/methodology, restoring `baseline_results/`.
