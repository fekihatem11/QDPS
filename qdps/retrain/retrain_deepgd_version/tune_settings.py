"""Step 2: choose each subject's retraining settings without looking at QDPS or SETS.

DeepGD tuned the retraining settings per subject but published only MNIST LeNet5's
(Retraining.ipynb cell 32). The settings pilot (pilot_settings.py) showed that the
current setting is too weak for SVHN (negative maxima) and that Fruit-360 needs the
logits loss. This script scores one candidate setting on a held-out split:

- the test set T is split once (seed SPLIT_SEED) into T_fit and T_val, half each;
  G_fit / G_val are the generated images whose source test image is in T_fit / T_val;
- the pretrained model is retrained on training set + T_fit + G_fit (a max-improvement
  run on half of the data) and scored on T_val and G_val, which it never saw;
- score = mean of the gains on T_val and G_val (accuracy points).

The rule that picks the setting (tune_report.py) uses only these scores. QDPS and SETS
are never run here.

Writes RESULTS_DIR/retrain_deepgd/tuning/<subject>__<tag>.json.

Usage (repo root):
    python -m qdps.retrain.retrain_deepgd_version.tune_settings <subject> <adadelta|adam> <lr> [--from-logits]
"""
import argparse
import json
from dataclasses import replace
from datetime import datetime

import numpy as np

from qdps.io.paths import RESULTS_DIR
from qdps.retrain.raw_data import load_pretrained_model, load_raw_subject
from qdps.retrain.retrain_deepgd_version.backends import _keras_predict
from qdps.retrain.retrain_deepgd_version.protocol import load_generated_combined, model_input
from qdps.retrain.retrain_step import config_for

TUNING_DIR = RESULTS_DIR / "retrain_deepgd" / "tuning"
SPLIT_SEED = 2026
SEED = 0


def tag_of(optimizer, lr, from_logits):
    return f"{optimizer}_lr{lr:g}" + ("_logits" if from_logits else "")


def split(n_test):
    """Boolean mask over T: True = T_fit."""
    fit = np.zeros(n_test, dtype=bool)
    fit[np.random.default_rng(SPLIT_SEED).permutation(n_test)[: n_test // 2]] = True
    return fit


def _acc(model, subject, x, y, preprocessed, chunk=20000):
    pred = np.concatenate([
        _keras_predict(model, x[i:i + chunk] if preprocessed else model_input(subject, x[i:i + chunk]))
        for i in range(0, len(x), chunk)])
    return float(np.mean(pred == y))


def run(subject, optimizer, lr, from_logits):
    import tensorflow as tf
    from sklearn.utils import shuffle
    from tensorflow.keras.utils import to_categorical

    from qdps.retrain.backends.keras_backend import _make_optimizer

    cfg = replace(config_for(subject), optimizer=optimizer, lr=lr)
    raw = load_raw_subject(subject)
    assert raw.framework == "keras", "tuning covers the Keras subjects"
    g_x, g_y = load_generated_combined(subject)
    n_test = len(raw.y_test_int)
    fit_t = split(n_test)
    fit_g = np.tile(fit_t, len(g_y) // n_test)          # G is op-major, test order inside each op
    assert np.array_equal(g_y[:n_test], raw.y_test_int)

    def scores(model):
        return (_acc(model, subject, raw.x_test[~fit_t], raw.y_test_int[~fit_t], True),
                _acc(model, subject, g_x[~fit_g], g_y[~fit_g], False))

    model = load_pretrained_model(subject)
    orig_t, orig_g = scores(model)
    print(f"{subject} {optimizer} lr={lr} from_logits={from_logits}: |T_fit|={fit_t.sum()} "
          f"|G_fit|={fit_g.sum()} original acc(T_val)={orig_t:.4f} acc(G_val)={orig_g:.4f}", flush=True)

    tf.keras.utils.set_random_seed(SEED)
    sparse = "sparse" in cfg.loss
    extra_x = np.concatenate([raw.x_test[fit_t], model_input(subject, g_x[fit_g])])
    extra_y = np.concatenate([raw.y_test_int[fit_t], g_y[fit_g]])
    y_extra = extra_y if sparse else to_categorical(extra_y, raw.n_classes)
    x, y = shuffle(np.concatenate([raw.x_train, extra_x]),
                   np.concatenate([raw.y_train_oh, y_extra]), random_state=SEED)
    del extra_x

    if from_logits:
        assert model.layers[-1].get_config().get("activation") == "linear", \
            "--from-logits is only for models whose last layer outputs logits (Fruit-360)"
        loss = (tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) if sparse
                else tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    else:
        loss = cfg.loss
    model = load_pretrained_model(subject)
    model.compile(optimizer=_make_optimizer(cfg), loss=loss)
    # fit-monitoring split drawn from T_val (monitoring only, as in the protocol)
    val_idx = np.flatnonzero(~fit_t)
    v_id = np.random.default_rng(SEED).choice(val_idx, size=min(cfg.val_size, len(val_idx)), replace=False)
    y_val = raw.y_test_int[v_id] if sparse else to_categorical(raw.y_test_int[v_id], raw.n_classes)
    model.fit(x, y, epochs=cfg.epochs, batch_size=cfg.batch_size,
              validation_data=(raw.x_test[v_id], y_val), verbose=2)
    del x, y

    acc_t, acc_g = scores(model)
    gain_t, gain_g = 100 * (acc_t - orig_t), 100 * (acc_g - orig_g)
    print(f"  acc(T_val)={acc_t:.4f} ({gain_t:+.2f}) acc(G_val)={acc_g:.4f} ({gain_g:+.2f}) "
          f"score={(gain_t + gain_g) / 2:+.2f}", flush=True)

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    out = TUNING_DIR / f"{subject}__{tag_of(optimizer, lr, from_logits)}.json"
    json.dump({"subject": subject, "optimizer": optimizer, "lr": lr, "from_logits": from_logits,
               "epochs": cfg.epochs, "batch_size": cfg.batch_size, "loss": cfg.loss, "seed": SEED,
               "split_seed": SPLIT_SEED, "n_T_fit": int(fit_t.sum()), "n_G_fit": int(fit_g.sum()),
               "original": {"acc_T_val": orig_t, "acc_G_val": orig_g},
               "retrained": {"acc_T_val": acc_t, "acc_G_val": acc_g},
               "gain_T_val": gain_t, "gain_G_val": gain_g, "score": (gain_t + gain_g) / 2,
               "timestamp": datetime.now().isoformat(timespec="seconds")}, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("subject")
    p.add_argument("optimizer", choices=["adadelta", "adam"])
    p.add_argument("lr", type=float)
    p.add_argument("--from-logits", action="store_true")
    a = p.parse_args()
    run(a.subject, a.optimizer, a.lr, a.from_logits)
