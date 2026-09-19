"""Final BEGF embedding readout: row normalization followed by k-means."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans


def row_l2_normalize(embedding: ArrayLike) -> NDArray[np.float64]:
    """Return a finite two-dimensional embedding with unit nonzero rows."""
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("embedding must be a nonempty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("embedding contains NaN or infinite values")

    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe_norms = np.maximum(row_norms, np.finfo(np.float64).eps)
    return values / safe_norms


def cluster_embedding(
    embedding: ArrayLike,
    n_clusters: int,
    *,
    random_state: int | None = 0,
    n_init: int = 20,
) -> NDArray[np.int64]:
    """Normalize rows of ``embedding`` and return final k-means labels."""
    if isinstance(n_clusters, (bool, np.bool_)) or not isinstance(
        n_clusters, (int, np.integer)
    ):
        raise ValueError("n_clusters must be a positive integer")
    if int(n_clusters) < 1:
        raise ValueError("n_clusters must be a positive integer")
    if isinstance(n_init, (bool, np.bool_)) or not isinstance(n_init, (int, np.integer)):
        raise ValueError("n_init must be a positive integer")
    if int(n_init) < 1:
        raise ValueError("n_init must be a positive integer")

    normalized = row_l2_normalize(embedding)
    if int(n_clusters) > normalized.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of embedding rows")
    model = KMeans(
        n_clusters=int(n_clusters),
        random_state=random_state,
        n_init=int(n_init),
        max_iter=300,
        tol=1.0e-4,
        algorithm="lloyd",
    )
    return np.asarray(model.fit_predict(normalized), dtype=np.int64)
