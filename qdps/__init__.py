"""QDPS — Quality-Diversity DPP Selection (importable package)."""
from .qdps import (
    select,
    METHOD_NAME,
    ADAPTIVE_DEFAULTS,
    get_adaptive_defaults,
    maxp_score,
)

__all__ = [
    "select",
    "METHOD_NAME",
    "ADAPTIVE_DEFAULTS",
    "get_adaptive_defaults",
    "maxp_score",
]
