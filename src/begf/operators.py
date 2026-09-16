"""Matrix-free Laplacian and three-band operators for BEGF."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


class PartitionProjectionLaplacian:
    """Matrix-free ``L = 2 * (I - X0 X0.T)``.

    Only the low-rank factor ``X0`` is stored. The constructor verifies the
    theorem premise ``0 <= X0.T @ X0 <= I`` only when the optional diagnostic
    check is requested. For ``X0`` returned by :func:`begf.build_x0`, the
    premise follows from averaging orthogonal partition projectors, so normal
    construction does not need a q-by-q eigendecomposition.
    """

    def __init__(
        self,
        x0: ArrayLike | sparse.spmatrix,
        *,
        tolerance: float = 1.0e-10,
        validate_theorem: bool = False,
    ) -> None:
        if sparse.issparse(x0):
            stored: sparse.csr_matrix | NDArray[np.float64] = sparse.csr_matrix(
                x0, dtype=np.float64, copy=True
            )
            if min(stored.shape) < 1:
                raise ValueError("X0 must be a nonempty two-dimensional array")
            if stored.nnz and not np.isfinite(stored.data).all():
                raise ValueError("X0 contains NaN or infinite values")
            stored.sum_duplicates()
            stored.sort_indices()
        else:
            matrix = np.asarray(x0, dtype=np.float64)
            if matrix.ndim != 2 or min(matrix.shape) < 1:
                raise ValueError("X0 must be a nonempty two-dimensional array")
            if not np.isfinite(matrix).all():
                raise ValueError("X0 contains NaN or infinite values")
            stored = np.ascontiguousarray(matrix).copy()
            stored.setflags(write=False)
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")

        self._x0 = stored
        self.shape = (int(stored.shape[0]), int(stored.shape[0]))
        self.dtype = np.dtype(np.float64)
        self._tolerance = tolerance
        self._theorem_diagnostics: tuple[float, float] | None = None
        if validate_theorem:
            self._theorem_diagnostics = self._compute_theorem_diagnostics()

    @property
    def x0(self) -> sparse.csr_matrix | NDArray[np.float64]:
        """The normalized ensemble factor, stored as CSR when supplied sparse."""
        return self._x0

    @property
    def n_samples(self) -> int:
        return self.shape[0]

    def _compute_theorem_diagnostics(self) -> tuple[float, float]:
        """Compute the optional q-by-q theorem diagnostic."""
        gram_raw = self._x0.T @ self._x0
        if sparse.issparse(gram_raw):
            gram_raw = gram_raw.toarray()
        gram = np.asarray(gram_raw, dtype=np.float64)
        gram = 0.5 * (gram + gram.T)
        gram_eigenvalues = np.linalg.eigvalsh(gram)
        minimum = float(gram_eigenvalues[0])
        maximum = float(gram_eigenvalues[-1])
        if minimum < -self._tolerance or maximum > 1.0 + self._tolerance:
            raise ValueError(
                "theorem premise failed: eigenvalues of X0.T @ X0 must lie in [0, 1], "
                f"got [{minimum}, {maximum}]"
            )
        return minimum, maximum

    def _prepare(self, values: ArrayLike) -> tuple[NDArray[np.float64], bool]:
        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim not in (1, 2) or raw.shape[0] != self.n_samples:
            raise ValueError("values must have one or two dimensions with n_samples rows")
        if not np.isfinite(raw).all():
            raise ValueError("values contains NaN or infinite values")
        was_vector = raw.ndim == 1
        return (raw[:, None] if was_vector else raw), was_vector

    def apply_projection(self, values: ArrayLike) -> NDArray[np.float64]:
        """Apply ``X0 X0.T`` without materializing a sample-by-sample matrix."""
        work, was_vector = self._prepare(values)
        projected = self._x0 @ (self._x0.T @ work)
        return np.asarray(projected[:, 0] if was_vector else projected)

    def apply_laplacian(self, values: ArrayLike) -> NDArray[np.float64]:
        """Apply the fixed paper Laplacian to one or more right-hand sides."""
        work, was_vector = self._prepare(values)
        result = 2.0 * (work - self._x0 @ (self._x0.T @ work))
        return np.asarray(result[:, 0] if was_vector else result)

    def apply_laplacian_powers(
        self, values: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``LZ``, ``L^2Z``, and ``L^3Z`` by successive actions."""
        z1 = self.apply_laplacian(values)
        z2 = self.apply_laplacian(z1)
        z3 = self.apply_laplacian(z2)
        return z1, z2, z3

    def apply_band_operators(
        self, values: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Apply ``R_lo``, ``R_mid``, and ``R_hi`` without storing powers."""
        z1, z2, z3 = self.apply_laplacian_powers(values)
        return z1 - z2 + 0.25 * z3, z2 - 0.5 * z3, 0.25 * z3

    def weighted_band_action(self, values: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
        """Apply ``sum_b weights[b] R_b`` to one or more right-hand sides."""
        work, was_vector = self._prepare(values)
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.shape != (3,) or not np.isfinite(weight_values).all():
            raise ValueError("weights must be a finite length-three vector")
        bands = self.apply_band_operators(work)
        result = weight_values[0] * bands[0] + weight_values[1] * bands[1] + weight_values[2] * bands[2]
        return np.asarray(result[:, 0] if was_vector else result)

    def band_energies(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return ``[<Y,R_loY>, <Y,R_midY>, <Y,R_hiY>]``."""
        work, _ = self._prepare(values)
        bands = self.apply_band_operators(work)
        raw = np.asarray([np.sum(work * band) for band in bands], dtype=np.float64)
        tolerance = 1.0e-10 * max(1.0, float(np.sum(np.abs(raw))))
        if np.any(raw < -tolerance):
            raise ValueError(f"band energy is significantly negative: {raw.tolist()}")
        raw[(raw < 0.0) & (raw >= -tolerance)] = 0.0
        return raw

    def as_linear_operator(self) -> LinearOperator:
        """Expose the same action to SciPy iterative-solver interfaces."""
        return LinearOperator(
            shape=self.shape,
            dtype=np.float64,
            matvec=self.apply_laplacian,
            matmat=self.apply_laplacian,
            rmatvec=self.apply_laplacian,
            rmatmat=self.apply_laplacian,
        )

    def validate(self, *, check_theorem: bool = False) -> dict[str, Any]:
        """Return storage diagnostics, optionally checking the q-by-q theorem premise.

        ``check_theorem=True`` is intended for small tests or explicit
        diagnostics. The default path reports that the check was not run and
        does not form ``X0.T @ X0``.
        """
        diagnostics: dict[str, Any] = {
            "laplacian_source": "partition_projection",
            "formula": "2*(I-X0*X0.T)",
            "matrix_free": True,
            "shape": [self.n_samples, self.n_samples],
            "x0_shape": [int(size) for size in self._x0.shape],
            "x0_gram_min_eigenvalue": None,
            "x0_gram_max_eigenvalue": None,
            "laplacian_spectrum_interval": [0.0, 2.0],
            "theorem_premise_checked": False,
            "theorem_premise_verified": None,
        }
        if self._theorem_diagnostics is not None or check_theorem:
            minimum, maximum = self._theorem_diagnostics or self._compute_theorem_diagnostics()
            self._theorem_diagnostics = (minimum, maximum)
            diagnostics.update(
                {
                    "x0_gram_min_eigenvalue": minimum,
                    "x0_gram_max_eigenvalue": maximum,
                    "theorem_premise_checked": True,
                    "theorem_premise_verified": True,
                }
            )
        return diagnostics
