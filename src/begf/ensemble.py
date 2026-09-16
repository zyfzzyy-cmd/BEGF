"""Construction of the normalized ensemble representation used by BEGF."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse


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


def _normalized_partition_block(labels: ArrayLike) -> sparse.csr_matrix:
    """Build one normalized partition block directly in CSR format."""
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
        raise ValueError("labels contain NaN or infinite values")

    _, inverse = np.unique(values, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64, copy=False)
    row_indices = np.arange(values.size, dtype=np.int32)
    data = 1.0 / np.sqrt(counts[inverse])
    return sparse.csr_matrix(
        (data, (row_indices, inverse)),
        shape=(values.size, n_clusters),
        dtype=np.float64,
    )


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
) -> tuple[sparse.csr_matrix, dict[str, object]]:
    """Build ``X0=[F^(1),...,F^(m)]/sqrt(m)`` from base label vectors.

    For partition ``r``, ``F^(r)=H^(r)D^(r)^(-1/2)``. Every nonempty cluster
    is retained, so each block has orthonormal columns and the resulting
    normalized co-membership kernel is represented by the sparse factor
    ``X0`` rather than materialized as a sample-by-sample matrix.
    """
    raw_partitions = _as_partition_sequence(partitions)
    if not raw_partitions:
        raise ValueError("at least one base partition is required")

    blocks: list[sparse.csr_matrix] = []
    cluster_counts: list[int] = []
    n_samples: int | None = None
    for index, labels in enumerate(raw_partitions):
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError(f"partition {index} must be a nonempty label vector")
        if n_samples is None:
            n_samples = int(labels.size)
        elif labels.size != n_samples:
            raise ValueError("all base partitions must contain the same number of samples")
        block = _normalized_partition_block(labels)
        blocks.append(block)
        cluster_counts.append(int(block.shape[1]))

    assert n_samples is not None
    x0 = sparse.hstack(blocks, format="csr").multiply(1.0 / np.sqrt(len(blocks))).tocsr()
    x0.sum_duplicates()
    x0.sort_indices()
    metadata: dict[str, object] = {
        "n_samples": n_samples,
        "n_partitions": len(blocks),
        "retained_cluster_counts": cluster_counts,
        "x0_shape": [int(size) for size in x0.shape],
        "x0_frobenius_norm_squared": float(x0.multiply(x0).sum()),
    }
    return x0, metadata
