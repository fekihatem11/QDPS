"""Rename conditions in the subjects' result files, so obsolete runs are kept but not used.

Renames are applied in order, inside one lock per subject file. report.py reads only
FULL_T / FULL_G.

History:
- 2026-10-27: switch to DeepGD's notebook definition (inputs only).
      FULL_T=FULL_T_with_train_set,FULL_G=FULL_G_with_train_set
- 2026-10-28: back to the paper's definition (training set + all inputs), which is what
  produced DeepGD's Table 7; the notebook-definition runs are kept as *_inputs_only.
      FULL_T=FULL_T_inputs_only,FULL_G=FULL_G_inputs_only,FULL_T_with_train_set=FULL_T,FULL_G_with_train_set=FULL_G
- 2026-10-28: tuned retraining settings (protocol.TUNED_SETTINGS); the runs with the old
  settings of the 5 tuned subjects are kept as *_old_settings.
      QDPS=QDPS_old_settings,SETS=SETS_old_settings,FULL_T=FULL_T_old_settings,FULL_G=FULL_G_old_settings
      Fashion_mnist_LeNet4,cifar10_12Conv,cifar10_ResNet20,SVHN_LeNet5,Fruit360_ResNet50

Stdlib only and run by path (importing the qdps package needs numpy), so it runs on any
node, e.g. Fir login nodes where the venv's numpy does not import:
    python3 qdps/retrain/retrain_deepgd_version/archive_conditions.py OLD=NEW[,OLD=NEW...] [subject,...]
"""
import fcntl
import json
import sys
from pathlib import Path

# = qdps.io.paths.RESULTS_DIR / "retrain_deepgd" / "subjects"
SUBJECTS_DIR = Path(__file__).resolve().parents[2] / "results" / "retrain_deepgd" / "subjects"


def rename(pairs, subjects=None):
    for path in sorted(SUBJECTS_DIR.glob("*.json")):
        if subjects and path.stem not in subjects:
            continue
        with open(path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            store = json.load(open(path))
            done = []
            for old, new in pairs:
                if old in store["conditions"]:
                    if new in store["conditions"]:
                        sys.exit(f"{path.stem}: {new} already exists, not overwriting")
                    store["conditions"][new] = store["conditions"].pop(old)
                    done.append(f"{old}->{new}")
            if done:
                with open(path, "w") as f:
                    json.dump(store, f, indent=2)
        print(f"{path.stem}: {', '.join(done) or 'nothing'}")


if __name__ == "__main__":
    rename([p.split("=") for p in sys.argv[1].split(",")],
           sys.argv[2].split(",") if len(sys.argv) > 2 else None)
