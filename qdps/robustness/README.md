# QDPS robustness pipeline

Evaluates how QDPS and SETS hold up when the model under test varies. For each
subject (dataset × architecture) we train **5 instances** from scratch with
different random seeds, then run SETS and QDPS on each instance and aggregate
robustness metrics.

> **Status (May 2026):** Training implemented for `mnist_LeNet1` and
> `mnist_LeNet5`. Other subjects (Fashion-MNIST, CIFAR-10, SVHN, Fruit-360,
> TinyImageNet) and downstream stages (predict, cluster, evaluate, aggregate)
> land in later commits.

---

## Folder layout

One self-contained folder per subject. Each folder owns its training script
and stores its trained instances locally:

```
qdps/robustness/
├── README.md
├── requirements.txt
├── mnist_LeNet1/
│   ├── train.py
│   └── instances/
│       ├── seed_0/  (model.h5, history.json, meta.json)
│       ├── seed_1/
│       └── ...
└── mnist_LeNet5/
    ├── train.py
    └── instances/...
```

`train.py` per subject is intentionally self-contained — it has its own data
loader, its own architecture builder, and its own training loop. No shared
modules to chase. This keeps each subject independently auditable and easy
to port to a new cluster.

---

## Variance source

Each of the 5 instances differs only in **one seed**. That seed controls:

1. Which 80 % subset of the MNIST training pool the instance is trained on
   (split-seed protocol).
2. Weight initialization, batch shuffling, dropout, augmentation noise.

The test set is **fixed across all 5 instances** so FDR is computed on a
common substrate.

---

## Reproducibility rules

| Lever | How it's controlled |
|---|---|
| Seeds | `seed_everything(seed)` at the top of every `train.py`. Seeds `random`, `numpy`, `tensorflow`, `PYTHONHASHSEED`, `tf.keras.utils.set_random_seed`. |
| Deterministic GPU ops | `TF_DETERMINISTIC_OPS=1` + `TF_CUDNN_DETERMINISTIC=1` exported before TF is imported. |
| Architectures | Exact reconstructions of `SETS/Input_data/Pretrained_model/model_mnist_LeNet{1,5}.h5` — same layers, same initializers, same compile-time optimizer (Adadelta lr=0.001 rho=0.95). |
| Data split | `np.random.default_rng(seed).choice(...)` — deterministic given the seed. A SHA-256 prefix of the chosen pool indices is written to `meta.json` for cross-checking. |
| Environment | `requirements.txt` pins TensorFlow 2.13.1, NumPy 1.24.4, etc. |
| Provenance | Every `model.h5` gets a sibling `meta.json` (package versions, git commit, timestamp, seed, pool hash, test accuracy). |

---

## Quick start

### Local laptop (smoke test)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r qdps/robustness/requirements.txt

# Train one instance — 5 epochs for a quick check
python qdps/robustness/mnist_LeNet1/train.py --seed 0 --epochs 5
```

### Compute Canada (Narval)

```bash
module load python/3.10 cuda/11.8 cudnn

virtualenv --no-download $SCRATCH/QDPS/.venv
source $SCRATCH/QDPS/.venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r $HOME/QDPS/qdps/robustness/requirements.txt

cd $HOME/QDPS

# One seed
python qdps/robustness/mnist_LeNet1/train.py --seed 0

# All 5 seeds (loop or SLURM array — array script added in a later commit)
for s in 0 1 2 3 4; do
    python qdps/robustness/mnist_LeNet1/train.py --seed $s
done
```

---

## Outputs

For each `train.py --seed n`:

```
mnist_LeNet1/instances/seed_n/
├── model.h5         # final trained Keras model
├── history.json     # per-epoch train + val loss / accuracy
└── meta.json        # subject, seed, env, git commit, test accuracy, pool hash
```

`meta.json` contains the `pool_indices_sha256_prefix` field — two instances
trained with the same seed should produce identical hashes. This is the
quick check that "my retrain on Narval reproduces my retrain on my laptop".
