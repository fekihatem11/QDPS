"""Retrain-and-evaluate for the DeepGD protocol: Keras (7 subjects) and PyTorch (TinyImageNet).

Both return accuracies on the original test set (T) and on the combined generated set (G).

Keras: same steps as qdps/retrain/backends/keras_backend.py (augment, shuffle, fresh
pretrained model, compile, fit with a 2500/5000-input monitoring split of the test set),
evaluated on the whole test set and on G.

PyTorch: same fine-tuning as qdps/retrain/backends/torch_backend.py (retrain_tiny.py).
That protocol evaluates after every epoch and keeps the best epoch; here the best epoch
is chosen on T, and G is reported at that same epoch.
"""
import numpy as np

from qdps.retrain.retrain_deepgd_version.protocol import model_input


# ---------------------------------------------------------------- Keras

def _keras_predict(model, x):
    return np.argmax(model.predict(x, batch_size=256, verbose=0), axis=1)


def keras_accuracies(model, raw, subject, g_x, g_y, chunk=20000):
    acc_t = float(np.mean(_keras_predict(model, raw.x_test) == raw.y_test_int))
    # G is preprocessed chunk by chunk: as one float32 array it is 14 GB for Fruit-360.
    pred_g = np.concatenate([_keras_predict(model, model_input(subject, g_x[i:i + chunk]))
                             for i in range(0, len(g_x), chunk)])
    acc_g = float(np.mean(pred_g == g_y))
    return acc_t, acc_g


def keras_original(raw, subject, model_loader, g_x, g_y):
    return keras_accuracies(model_loader(), raw, subject, g_x, g_y)


def keras_retrain(raw, subject, model_loader, extra_x, extra_y_int, cfg, seed, g_x, g_y,
                  with_train=True):
    """extra_x: preprocessed images; extra_y_int: integer labels. with_train=False trains on
    the extra inputs only (DeepGD's max-improvement runs)."""
    import tensorflow as tf
    from sklearn.utils import shuffle
    from tensorflow.keras.utils import to_categorical

    from qdps.retrain.backends.keras_backend import _make_optimizer

    tf.keras.utils.set_random_seed(seed)
    sparse = "sparse" in cfg.loss
    y_extra = extra_y_int if sparse else to_categorical(extra_y_int, raw.n_classes)
    y_test_model = raw.y_test_int if sparse else to_categorical(raw.y_test_int, raw.n_classes)

    if with_train:
        x, y = shuffle(np.concatenate([raw.x_train, extra_x]),
                       np.concatenate([raw.y_train_oh, y_extra]), random_state=seed)
    else:
        x, y = shuffle(extra_x, y_extra, random_state=seed)

    model = model_loader()
    loss = cfg.loss
    if cfg.from_logits:
        assert model.layers[-1].get_config().get("activation") == "linear", \
            "from_logits is only for models whose last layer outputs logits (Fruit-360)"
        loss = (tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) if sparse
                else tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    model.compile(optimizer=_make_optimizer(cfg), loss=loss)
    rng = np.random.default_rng(seed)
    v_id = rng.choice(len(raw.x_test), size=min(cfg.val_size, len(raw.x_test)), replace=False)
    model.fit(x, y, epochs=cfg.epochs, batch_size=cfg.batch_size,
              validation_data=(raw.x_test[v_id], y_test_model[v_id]), verbose=0)
    del x, y
    return keras_accuracies(model, raw, subject, g_x, g_y)


# ---------------------------------------------------------------- PyTorch (TinyImageNet)

def _tiny_transform():
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ArrayDataset:
    """uint8 (N, H, W, 3) images + int labels, with the TinyImageNet transform."""

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.transform = _tiny_transform()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.fromarray(self.x[i])), int(self.y[i])


def _loader(dataset, batch_size, shuffle):
    from torch.utils.data import DataLoader
    from qdps.retrain.backends.torch_backend import _NUM_WORKERS
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=_NUM_WORKERS, pin_memory=True)


def torch_original(raw, model_loader, g_x, g_y):
    from qdps.retrain.backends.torch_backend import _VAL_BSZ, _evaluate, build_model
    model, classifier, _, _ = build_model(model_loader())
    acc_t = _evaluate(model, classifier, _loader(raw.val_dataset, _VAL_BSZ, False))
    acc_g = _evaluate(model, classifier, _loader(ArrayDataset(g_x, g_y), _VAL_BSZ, False))
    return float(acc_t), float(acc_g)


def torch_retrain(raw, model_loader, extra_dataset, cfg, seed, g_x, g_y, with_train=True):
    """extra_dataset: a torch Dataset of (transformed image, label) appended to the train set
    (or used alone when with_train=False)."""
    import torch
    import torch.nn as nn
    from qdps.retrain.backends.torch_backend import _VAL_BSZ, _evaluate, build_model, set_seed

    set_seed(seed)
    train_loader = _loader(raw.train_set + extra_dataset if with_train else extra_dataset,
                           cfg.batch_size, True)
    t_loader = _loader(raw.val_dataset, _VAL_BSZ, False)
    g_loader = _loader(ArrayDataset(g_x, g_y), _VAL_BSZ, False)

    model, classifier, optimizer, optimizer_reg = build_model(model_loader())
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    best = (-1.0, None)
    for epoch in range(cfg.epochs):
        model.train()
        classifier.train()
        for inputs, targets in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            optimizer_reg.zero_grad()
            loss = criterion(classifier(model(inputs).squeeze(-1).squeeze(-1)), targets.long())
            loss.backward()
            optimizer.step()
            optimizer_reg.step()
        acc_t = _evaluate(model, classifier, t_loader)
        acc_g = _evaluate(model, classifier, g_loader)
        print(f"    epoch {epoch + 1}/{cfg.epochs}: acc(T)={acc_t:.4f} acc(G)={acc_g:.4f}", flush=True)
        if acc_t > best[0]:
            best = (acc_t, acc_g)
    return float(best[0]), float(best[1])
