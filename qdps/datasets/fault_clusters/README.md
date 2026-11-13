# `qdps/fault_clusters/`

Per-subject **fault-cluster ground truth** computed against the original
SETS pretrained models (one folder per subject). The QDPS / SETS / DeepGD /
RS evaluation pipeline uses these as the answer key for the Fault Detection
Rate metric.

> Used by the main QDPS *vs* SETS comparison (not by the robustness
> pipeline). Per-instance fault clusters for the 5 retrained instances per
> subject live separately under
> `qdps/robustness/<subject>/instances/seed_<n>/`.

---

## Layout

```
qdps/fault_clusters/
├── mnist_LeNet1/
│   ├── output_probability.npy   (10000, 10)   model softmax outputs on test set
│   ├── mis_index_test.npy       (n_mis,)      indices of misclassified test inputs
│   └── cluster_results.npy      (n_mis_all,)  cluster label per misclassified input
│                                              -1 = noise, otherwise cluster id
├── mnist_LeNet5/...
├── Fashion_mnist_LeNet4/...
├── cifar10_12conv/...           (note: lowercase folder name)
├── cifar10_ResNet20/...
├── SVHN_LeNet5/...
├── Fruit360_ResNet50/...
└── TinyImageNet_ResNet101/...   (uses .pkl instead of .npy for the index files)
```

`cluster_results.npy` is shaped `(n_mis_test + n_mis_train,)` — SETS clusters
the train+test misclassifications together; the **test slice is the first
`n_mis_test` entries**, indexed by `mis_index_test.npy`.

---

## Provenance

These files are copied verbatim from the official SETS replication package
(Wang et al., TOSEM 2025), originally at `SETS/Input_data/Fault_clusters/`.

How SETS produced them:
1. Run the original pretrained model on the full train + test set.
2. Take the misclassifications (predicted ≠ true label).
3. Extract VGG16 `block5_conv3` features for those misclassified images.
4. Augment with the (truth, pred) label pair as two extra dimensions.
5. Two-stage UMAP → HDBSCAN with `min_cluster_size = 5`.
6. Grid sweep over UMAP hyperparameters, pick best by silhouette.

See `SETS/Source_code/cluster.py` for the exact code. The per-subject UMAP
hyperparameters that produced these specific clusterings were not published
— the published files are the final selected output.

---

## How FDR uses these files

`compute_fdr` follows the SETS paper's **"1noisy"** convention: all noise
(`-1`) misclassifications collapse to a single "fault", and

```
FDR = unique_faults_found / min(budget, total_faults)
```

See `qdps/data_loader.py::compute_fdr` for the implementation.

---

## Subject-specific quirks

- `cifar10_12conv` is lowercase — case-sensitive on Linux. Use
  `FOLDER_MAP` in `qdps/data_loader.py`, don't hand-construct paths.
- `Fruit360_ResNet50` saves `mis_index_test.npy` as a 2D object array; the
  loader unwraps with `mis_index = mis_index[0]`.
- `TinyImageNet_ResNet101` uses pickle (`.pkl`) instead of `.npy` and
  stores tuples; see `data_loader.py` for the unwrap logic.
