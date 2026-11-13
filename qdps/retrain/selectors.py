"""Selection-method registry for the retraining experiment.

Both methods select GLOBAL test indices from the same pool (T restricted to valid
indices) using the package's existing selectors, driven by the VGG features and
softmax probabilities from ``load_subject``.
"""
from qdps import select as qdps_select
from qdps.baselines.sets import sets_select


def _qdps(k, pool, features, output_probability):
    return qdps_select(k, pool, features, output_probability)


def _sets(k, pool, features, output_probability):
    return sets_select(k, pool, features, output_probability)


SELECTORS = {
    "QDPS": _qdps,
    "SETS": _sets,
}
