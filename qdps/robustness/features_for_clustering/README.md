# `qdps/robustness/features_for_clustering/`

**Raw** VGG16 image features used by the **fault-clustering** pipeline
(UMAP + HDBSCAN) for each retrained model instance.

> Not the same files as `qdps/features_for_selection/`. Those are
> min-max normalized to [0, 1] (per column) and are used by the QDPS /
> SETS selection kernel. The clustering pipeline needs the RAW VGG output
> instead, because the existing normalization loses information needed to
> preserve geometric structure across train + test (per-column min-max only
> makes sense if computed on the same feature pool that's later clustered).

## Layout

One subfolder per dataset:

```
features_for_clustering/
├── mnist/
│   ├── features_train_raw.npy      # (60000, 4608) float32
│   ├── features_train_meta.json
│   ├── features_test_raw.npy       # (10000, 4608) float32
│   └── features_test_meta.json
├── cifar10/
│   └── ...
└── ...
```

Features are dataset-level (not per-instance) — VGG16 has frozen ImageNet
weights, so the output is a deterministic function of input pixels only.
All retrained instances of the same subject reuse the same arrays.

## Reproduce

```
python qdps/robustness/extract_vgg_features.py --dataset mnist
```
