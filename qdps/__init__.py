"""QDPS — Quality-Diversity DPP Selection (importable package)."""
from .qdps import (
    select,
    METHOD_NAME,
    ALPHA,
    maxp_score,
)

__all__ = [
    "select",
    "METHOD_NAME",
    "ALPHA",
    "maxp_score",
]
