"""Keras retraining backend (mnist / Fashion / cifar / SVHN / fruit).

Faithful port of the retrain step in ``SETS/Source_code/retrain_four.py``:
augment the training set with the selected test inputs (and their labels),
reload a fresh pretrained model, compile with legacy Adadelta, fit, then
evaluate accuracy on the validation set V.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

FRAMEWORK = "keras"


def _predict_labels(model, x):
    probs = model.predict(x, batch_size=256, verbose=0)
    return np.argmax(probs, axis=1)


def original_accuracy(raw, model_loader, V):
    """Accuracy of the un-retrained pretrained model on V."""
    model = model_loader()
    preds = _predict_labels(model, raw.x_test[V])
    return float(accuracy_score(raw.y_test_int[V], preds))


def _make_optimizer(cfg):
    if cfg.optimizer == "adadelta":
        return tf.keras.optimizers.legacy.Adadelta(learning_rate=cfg.lr)
    if cfg.optimizer == "adam":
        return tf.keras.optimizers.legacy.Adam(learning_rate=cfg.lr)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def retrain_and_eval(raw, model_loader, selected, V, cfg, seed):
    """One retraining run: augment -> fresh model -> fit -> eval on V.

    Returns {"acc_re": float}. Deterministic given ``seed`` (shuffle + the
    random fit-monitoring split), modulo backend nondeterminism.
    """
    # Test labels in the form the loss expects: integer for sparse CE
    # (retrain_fruit.py), one-hot otherwise (retrain_four.py).
    if "sparse" in cfg.loss:
        y_test_model = raw.y_test_int
    else:
        y_test_model = to_categorical(raw.y_test_int, raw.n_classes)

    x_new = np.concatenate([raw.x_train, raw.x_test[selected]], axis=0)
    y_new = np.concatenate([raw.y_train_oh, y_test_model[selected]], axis=0)
    x, y = shuffle(x_new, y_new, random_state=seed)

    model = model_loader()
    model.compile(optimizer=_make_optimizer(cfg), loss=cfg.loss)

    rng = np.random.default_rng(seed)
    val_size = min(cfg.val_size, len(raw.x_test))
    v_id = rng.choice(len(raw.x_test), size=val_size, replace=False)

    model.fit(
        x, y,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(raw.x_test[v_id], y_test_model[v_id]),
        verbose=0,
    )

    preds = _predict_labels(model, raw.x_test[V])
    acc_re = float(accuracy_score(raw.y_test_int[V], preds))
    return {"acc_re": acc_re}
