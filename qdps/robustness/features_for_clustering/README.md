# `qdps/robustness/features_for_clustering/`

**Raw** VGG16 image features used by the **fault-clustering** pipeline
(UMAP + HDBSCAN) for each retrained model instance.

> Not the same files as `qdps/features_for_selection/`. Those are
> min-max normalized to [0, 1] (per column) and are used by the QDPS /
> SETS selection kernel. The clustering pipeline needs the RAW VGG output
> instead, because the existing normalization loses information needed to
> preserve geometric structure across train + test (per-column min-max only
> makes sense if computed on the same feature pool that's later clustered).

---


