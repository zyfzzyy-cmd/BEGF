"""Public BEGF implementation for the ICASSP 2027 companion repository."""

from .ensemble import build_x0, labels_to_one_hot
from .operators import PartitionProjectionLaplacian
from .solver import BEGFConfig, BEGFResult, fit_begf, fit_begf_from_partitions

__all__ = [
    "BEGFConfig",
    "BEGFResult",
    "PartitionProjectionLaplacian",
    "build_x0",
    "fit_begf",
    "fit_begf_from_partitions",
    "labels_to_one_hot",
]
