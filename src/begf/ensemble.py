"""Construction of the normalized ensemble representation used by BEGF."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def labels_to_one_hot(labels: ArrayLike) -> NDArray[np.float64]:
    """Convert one label vector into a nonempty-cluster indicator matrix.

    Labels may be integer, floating-point, or string values. The column order
    is deterministic because it follows ``numpy.unique``.
    """
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
        raise ValueError("labels contain NaN or infinite values")
    _, inverse = np.unique(values, return_inverse=True)
    indicators = np.zeros((values.size, int(inverse.max()) + 1), dtype=np.float64)
    indicators[np.arange(values.size), inverse] = 1.0
    return indicators


def _as_partition_sequence(partitions: Iterable[ArrayLike] | NDArray[np.generic]) -> list[np.ndarray]:
    if isinstance(partitions, np.ndarray):
        if partitions.ndim != 2:
            raise ValueError("a partition matrix must have shape (n_partitions, n_samples)")
        return [partitions[row] for row in range(partitions.shape[0])]
    result = [np.asarray(partition) for partition in partitions]
    if not result:
        raise ValueError("at least one base partition is required")
    return result


def build_x0(
    partitions: Iterable[ArrayLike] | NDArray[np.generic],
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Build ``X0=[F^(1),...,F^(m)]/sqrt(m)`` from base label vectors.

    For partition ``r``, ``F^(r)=H^(r)D^(r)^(-1/2)``. Every nonempty cluster
    is retained, so each block has orthonormal columns and the resulting
    normalized co-membership kernel is ``X0 @ X0.T``.
    """
    raw_partitions = _as_partition_sequence(partitions)
    if not raw_partitions:
        raise ValueError("at least one base partition is required")

    blocks: list[NDArray[np.float64]] = []
    cluster_counts: list[int] = []
    n_samples: int | None = None
    for index, labels in enumerate(raw_partitions):
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError(f"partition {index} must be a nonempty label vector")
        if n_samples is None:
            n_samples = int(labels.size)
        elif labels.size != n_samples:
            raise ValueError("all base partitions must contain the same number of samples")
        indicators = labels_to_one_hot(labels)
        counts = indicators.sum(axis=0)
        block = indicators / np.sqrt(counts)[None, :]
        blocks.append(block)
        cluster_counts.append(int(block.shape[1]))

    assert n_samples is not None
    x0 = np.ascontiguousarray(np.concatenate(blocks, axis=1) / np.sqrt(len(blocks)))
    x0.setflags(write=False)
    metadata: dict[str, object] = {
        "n_samples": n_samples,
        "n_partitions": len(blocks),
        "retained_cluster_counts": cluster_counts,
        "x0_shape": [int(size) for size in x0.shape],
        "x0_frobenius_norm_squared": float(np.sum(x0 * x0)),
    }
    return x0, metadata
