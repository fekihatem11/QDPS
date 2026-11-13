"""Pilot: can other retraining settings make the negative max-improvement runs positive?

The max-improvement runs (FULL_T = training set + test set, FULL_G = training set +
generated set) are evaluated on the very inputs they train on, so a drop in accuracy
means the training settings do not work for that subject. DeepGD published only MNIST
LeNet5's settings (Retraining.ipynb cell 32). Negative maxima so far: SVHN (T and G),
Fruit-360 (G). Fruit-360's last layer is linear (logits) but SETS's code trains it with
from_logits=False; --from-logits tests the corrected loss.

One run per setting, accuracy on T and G after every epoch. Writes
RESULTS_DIR/retrain_deepgd/pilot/<subject>__<condition>__<tag>.json; the subjects'
result files are not touched. QDPS and SETS are not run here.

Usage (repo root):
    python -m qdps.retrain.retrain_deepgd_version.pilot_settings <subject> <FULL_T|FULL_G> \
        <adadelta|adam> <lr> [--epochs 30] [--batch 50] [--from-logits] [--seed 0]
"""
import argparse
import json
from dataclasses import replace
from datetime import datetime

import numpy as np

from qdps.io.paths import RESULTS_DIR
from qdps.retrain.raw_data import load_pretrained_model, load_raw_subject
from qdps.retrain.retrain_deepgd_version.backends import keras_accuracies
from qdps.retrain.retrain_deepgd_version.protocol import (
    FULL_G_INCLUDES_TEST, MAX_CONDITIONS_WITH_TRAIN_SET, load_generated_combined, model_input,
)
from qdps.retrain.retrain_step import config_for

PILOT_DIR = RESULTS_DIR / "retrain_deepgd" / "pilot"


def run(subject, condition, optimizer, lr, epochs, batch, from_logits, seed):
    import tensorflow as tf
    from sklearn.utils import shuffle
    from tensorflow.keras.utils import to_categorical

    from qdps.retrain.backends.keras_backend import _make_optimizer

    cfg = replace(config_for(subject), optimizer=optimizer, lr=lr, epochs=epochs, batch_size=batch)
    raw = load_raw_subject(subject)
    assert raw.framework == "keras", "pilot covers the Keras subjects"
    g_x, g_y = load_generated_combined(subject)
    model_loader = lambda: load_pretrained_model(subject)  # noqa: E731

    orig_t, orig_g = keras_accuracies(model_loader(), raw, subject, g_x, g_y)
    print(f"{subject} {condition} {optimizer} lr={lr} epochs={epochs} batch={batch} "
          f"from_logits={from_logits}: original acc(T)={orig_t:.4f} acc(G)={orig_g:.4f}", flush=True)

    if condition == "FULL_T":
        extra_x, extra_y = raw.x_test, raw.y_test_int
    else:
        extra_x, extra_y = model_input(subject, g_x), g_y
        if FULL_G_INCLUDES_TEST:
            extra_x = np.concatenate([extra_x, raw.x_test])
            extra_y = np.concatenate([extra_y, raw.y_test_int])
    assert MAX_CONDITIONS_WITH_TRAIN_SET

    tf.keras.utils.set_random_seed(seed)
    sparse = "sparse" in cfg.loss
    y_extra = extra_y if sparse else to_categorical(extra_y, raw.n_classes)
    y_test_model = raw.y_test_int if sparse else to_categorical(raw.y_test_int, raw.n_classes)
    x, y = shuffle(np.concatenate([raw.x_train, extra_x]),
                   np.concatenate([raw.y_train_oh, y_extra]), random_state=seed)
    del extra_x

    if from_logits:
        loss = (tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) if sparse
                else tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    else:
        loss = cfg.loss
    model = model_loader()
    if from_logits:
        assert model.layers[-1].get_config().get("activation") == "linear", \
            "--from-logits is only for models whose last layer outputs logits (Fruit-360)"
    model.compile(optimizer=_make_optimizer(cfg), loss=loss)
    rng = np.random.default_rng(seed)
    v_id = rng.choice(len(raw.x_test), size=min(cfg.val_size, len(raw.x_test)), replace=False)

    curve = []

    class EpochEval(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc_t, acc_g = keras_accuracies(self.model, raw, subject, g_x, g_y)
            curve.append({"epoch": epoch + 1, "loss": float(logs["loss"]),
                          "val_loss": float(logs.get("val_loss", np.nan)), "acc_T": acc_t, "acc_G": acc_g})
            print(f"  epoch {epoch + 1}/{epochs}: loss={logs['loss']:.4f} acc(T)={acc_t:.4f} "
                  f"({100 * (acc_t - orig_t):+.2f}) acc(G)={acc_g:.4f} ({100 * (acc_g - orig_g):+.2f})", flush=True)

    model.fit(x, y, epochs=cfg.epochs, batch_size=cfg.batch_size,
              validation_data=(raw.x_test[v_id], y_test_model[v_id]), verbose=0, callbacks=[EpochEval()])

    tag = f"{optimizer}_lr{lr:g}_e{epochs}_b{batch}" + ("_logits" if from_logits else "") + f"_s{seed}"
    out = PILOT_DIR / f"{subject}__{condition}__{tag}.json"
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"subject": subject, "condition": condition, "optimizer": optimizer, "lr": lr,
               "epochs": epochs, "batch_size": batch, "from_logits": from_logits, "seed": seed,
               "loss": cfg.loss, "val_size": cfg.val_size,
               "original": {"acc_T": orig_t, "acc_G": orig_g}, "curve": curve,
               "timestamp": datetime.now().isoformat(timespec="seconds")}, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("subject")
    p.add_argument("condition", choices=["FULL_T", "FULL_G"])
    p.add_argument("optimizer", choices=["adadelta", "adam"])
    p.add_argument("lr", type=float)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--from-logits", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.subject, a.condition, a.optimizer, a.lr, a.epochs,
        a.batch or config_for(a.subject).batch_size, a.from_logits, a.seed)
