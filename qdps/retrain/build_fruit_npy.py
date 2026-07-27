"""Rebuild the Fruit-360 npy arrays used by the SETS retraining experiment.

Replicates SETS/Source_code/cluster.py exactly: load Training/ and Test/ with
``image_dataset_from_directory(image_size=(100,100), batch_size=32, shuffle=False)``
(deterministic alphabetical ordering) and dump the UNSCALED float32 arrays:

    fruit_x_train_origin.npy  fruit_y_train.npy
    fruit_x_test_origin.npy   fruit_y_test.npy

Then VERIFY the rebuild: predict the test set with the pretrained
fruit_resnet2.h5 and compare the misclassified-index set against the recorded
fault-cluster ground truth (mis_index_test.npy). An exact (or near-exact) match
proves the directory ordering reproduces the original arrays.

Usage:
    python -m qdps.retrain.build_fruit_npy <fruits-360-root>
where <fruits-360-root> contains Training/ and Test/ class subdirectories.
"""
import sys

import numpy as np

from qdps.io.paths import RAW_DATA_DIR, FAULT_CLUSTERS
from qdps.retrain.raw_data import load_pretrained_model


def build_split(split_dir):
    import tensorflow as tf
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(split_dir), image_size=(100, 100), batch_size=32, shuffle=False
    )
    images, labels = [], []
    for x_batch, y_batch in ds:
        images.append(x_batch.numpy())
        labels.append(y_batch.numpy())
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def verify_against_fault_clusters(x_test, y_test):
    """Predict with fruit_resnet2.h5 and compare mis-set vs recorded ground truth."""
    model = load_pretrained_model("Fruit360_ResNet50")
    preds = np.argmax(model.predict(x_test, batch_size=128, verbose=0), axis=1)
    mis_ours = set(np.where(preds != y_test)[0].tolist())

    recorded = np.load(FAULT_CLUSTERS / "Fruit360_ResNet50" / "mis_index_test.npy",
                       allow_pickle=True)
    recorded = recorded[0]  # stored wrapped (see qdps.io.loader Fruit360 case)
    mis_ref = set(int(i) for i in np.asarray(recorded).ravel())

    inter = len(mis_ours & mis_ref)
    print(f"verify: |mis_ours|={len(mis_ours)} |mis_ref|={len(mis_ref)} "
          f"overlap={inter} jaccard={inter / max(1, len(mis_ours | mis_ref)):.4f}")
    return mis_ours == mis_ref


def main(argv):
    root = argv[0] if argv else None
    if root is None:
        print(__doc__)
        sys.exit(1)
    from pathlib import Path
    root = Path(root)
    out_dir = RAW_DATA_DIR / "fruit360"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = build_split(root / "Training")
    print(f"train: {x_train.shape} labels {y_train.shape}")
    x_test, y_test = build_split(root / "Test")
    print(f"test:  {x_test.shape} labels {y_test.shape}")

    np.save(out_dir / "fruit_x_train_origin.npy", x_train)
    np.save(out_dir / "fruit_y_train.npy", y_train)
    np.save(out_dir / "fruit_x_test_origin.npy", x_test)
    np.save(out_dir / "fruit_y_test.npy", y_test)
    print(f"saved arrays -> {out_dir}")

    ok = verify_against_fault_clusters(x_test, y_test)
    print("VERIFICATION:", "EXACT MATCH" if ok else "MISMATCH — inspect before retraining")


if __name__ == "__main__":
    main(sys.argv[1:])
