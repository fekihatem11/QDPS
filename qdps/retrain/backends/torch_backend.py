"""PyTorch retraining backend for TinyImageNet_ResNet101 (Narval-only).

Faithful port of ``SETS/Source_code/retrain_tiny.py``. The original imports
DeepGD's ``src.utils.decomposeModel`` and ``src.regressor`` modules, which are
NOT in the replication package; both are reconstructed here from the checkpoint
structure of ``model_best.pth.tar``:

- ``state_dict``   : DataParallel over ``nn.Sequential(*resnet101.children()[:-1])``
                     (features incl. avgpool -> (N, 2048, 1, 1))
- ``classifier_state_dict`` : keys ``net.weight (200,2048)`` / ``net.bias (200,)``
                     -> a wrapper module holding a single Linear(2048, 200)

Protocol (as in the original): FINE-TUNE from the checkpoint (model + optimizer
state restored), SGD lr=1e-3 for both parts, batch 64, 10 epochs, CrossEntropy,
evaluate on V after EVERY epoch and report the BEST epoch's accuracy.
"""
import os
import random

import numpy as np

FRAMEWORK = "torch"

_VAL_BSZ = 1536
_NUM_WORKERS = int(os.environ.get("QDPS_TORCH_WORKERS", "8"))


def _torch():
    import torch
    return torch


def set_seed(seed):
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(checkpoint_path):
    """Rebuild the decomposed resnet101 + Linear classifier from the checkpoint.

    Returns (model, classifier, optimizer, optimizer_reg) with weights and the
    feature-extractor optimizer state restored (as in retrain_tiny.py).
    """
    torch = _torch()
    import torch.nn as nn
    import torchvision.models as models

    resnet = models.resnet101()
    model = nn.Sequential(*list(resnet.children())[:-1])   # drop fc, keep avgpool
    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    class ClassifierNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(2048, 200)

        def forward(self, x):
            return self.net(x)

    classifier = ClassifierNet()
    if torch.cuda.is_available():
        classifier = classifier.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer_reg = torch.optim.SGD(classifier.parameters(), lr=1e-3)

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    optimizer.load_state_dict(ck["optimizer"])
    classifier.load_state_dict(ck["classifier_state_dict"])
    return model, classifier, optimizer, optimizer_reg


def _evaluate(model, classifier, loader):
    torch = _torch()
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            features = model(inputs).squeeze(-1).squeeze(-1)
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


def _val_loader(raw, indices):
    from torch.utils.data import DataLoader, Subset
    return DataLoader(
        Subset(raw.val_dataset, list(indices)),
        batch_size=_VAL_BSZ, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True,
    )


def original_accuracy(raw, model_loader, V):
    model, classifier, _, _ = build_model(model_loader())
    return float(_evaluate(model, classifier, _val_loader(raw, V)))


def retrain_and_eval(raw, model_loader, selected, V, cfg, seed):
    """One fine-tuning run; returns {"acc_re": best-epoch accuracy on V}."""
    torch = _torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    set_seed(seed)
    extra = Subset(raw.val_dataset, list(selected))
    train_loader = DataLoader(
        raw.train_set + extra,
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=_NUM_WORKERS, pin_memory=True,
    )
    val_loader = _val_loader(raw, V)

    model, classifier, optimizer, optimizer_reg = build_model(model_loader())
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        classifier.train()
        for inputs, targets in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            optimizer_reg.zero_grad()
            features = model(inputs).squeeze(-1).squeeze(-1)
            outputs = classifier(features)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            optimizer_reg.step()

        acc = _evaluate(model, classifier, val_loader)
        best_acc = max(best_acc, acc)
        print(f"    epoch {epoch + 1}/{cfg.epochs}: acc(V)={acc:.4f}", flush=True)

    return {"acc_re": float(best_acc)}


# ---- TinyImageNet data (val ordering = val_annotations.txt lines) ----

def parse_classes(file):
    filenames, classes = [], []
    with open(file) as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                filenames.append(tokens[0])
                classes.append(tokens[1])
    return filenames, classes


def load_tiny_data(root):
    """Build (train_set, val_dataset) exactly as retrain_tiny.py does."""
    import torchvision.transforms as transforms
    import torchvision.datasets as tvdatasets
    from PIL import Image
    from torch.utils import data as torch_data

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_set = tvdatasets.ImageFolder(os.path.join(root, "train"), transform)

    class TImgNetDataset(torch_data.Dataset):
        def __init__(self, img_path, gt_path, class_to_idx, transform):
            self.img_path = img_path
            self.transform = transform
            self.imgs, self.classnames = parse_classes(gt_path)
            self.classidx = [class_to_idx[c] for c in self.classnames]

        def __getitem__(self, index):
            with open(os.path.join(self.img_path, self.imgs[index]), "rb") as f:
                img = Image.open(f).convert("RGB")
                img = self.transform(img)
            return img, self.classidx[index]

        def __len__(self):
            return len(self.imgs)

    val_dataset = TImgNetDataset(
        os.path.join(root, "val", "images"),
        os.path.join(root, "val", "val_annotations.txt"),
        class_to_idx=train_set.class_to_idx.copy(),
        transform=transform,
    )
    return train_set, val_dataset
