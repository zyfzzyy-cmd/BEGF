import numpy as np
from scipy import sparse

from begf import BEGFConfig, PartitionProjectionLaplacian, build_x0, fit_begf
from begf import solver as solver_module


def _partitions() -> list[np.ndarray]:
    return [
        np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2]),
        np.array([0, 1, 1, 0, 2, 2, 1, 0, 2, 0, 1, 2]),
        np.array([1, 0, 1, 2, 2, 0, 0, 1, 2, 0, 2, 1]),
    ]


def _dense_reference(partitions: list[np.ndarray]) -> np.ndarray:
    blocks = []
    for labels in partitions:
        values, inverse = np.unique(labels, return_inverse=True)
        indicators = np.zeros((labels.size, values.size), dtype=np.float64)
        indicators[np.arange(labels.size), inverse] = 1.0
        counts = indicators.sum(axis=0)
        blocks.append(indicators / np.sqrt(counts)[None, :])
    return np.concatenate(blocks, axis=1) / np.sqrt(len(blocks))


def test_sparse_dense_x0_equivalence() -> None:
    partitions = _partitions()
    sparse_x0, metadata = build_x0(partitions)
    dense_reference = _dense_reference(partitions)

    assert sparse.isspmatrix_csr(sparse_x0)
    assert sparse_x0.nnz == len(partitions) * len(partitions[0])
    assert metadata["x0_shape"] == list(dense_reference.shape)
    np.testing.assert_allclose(sparse_x0.toarray(), dense_reference, atol=1.0e-14)


def test_sparse_dense_operator_equivalence() -> None:
    sparse_x0, _ = build_x0(_partitions())
    dense_x0 = sparse_x0.toarray()
    sparse_operator = PartitionProjectionLaplacian(sparse_x0)
    dense_laplacian = 2.0 * (np.eye(dense_x0.shape[0]) - dense_x0 @ dense_x0.T)
    rng = np.random.default_rng(123)
    signal = rng.normal(size=(dense_x0.shape[0], 3))

    np.testing.assert_allclose(
        sparse_operator.apply_laplacian(signal),
        dense_laplacian @ signal,
        atol=1.0e-12,
    )
    z1 = dense_laplacian @ signal
    z2 = dense_laplacian @ z1
    z3 = dense_laplacian @ z2
    expected_bands = (z1 - z2 + 0.25 * z3, z2 - 0.5 * z3, 0.25 * z3)
    for actual, expected in zip(sparse_operator.apply_band_operators(signal), expected_bands):
        np.testing.assert_allclose(actual, expected, atol=1.0e-11)
    np.testing.assert_allclose(
        sparse_operator.band_energies(signal),
        np.asarray([np.sum(signal * band) for band in expected_bands]),
        atol=1.0e-10,
    )


def test_default_validation_skips_gram_eigendecomposition(monkeypatch) -> None:
    sparse_x0, _ = build_x0(_partitions())

    def fail_if_called(*args, **kwargs):
        raise AssertionError("eigvalsh must be explicit diagnostic work")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail_if_called)
    diagnostics = PartitionProjectionLaplacian(sparse_x0).validate()
    assert diagnostics["theorem_premise_checked"] is False
    assert diagnostics["theorem_premise_verified"] is None


def test_sparse_solver_matches_dense_reference() -> None:
    sparse_x0, _ = build_x0(_partitions())
    dense_x0 = sparse_x0.toarray()
    config = BEGFConfig(
        mu=0.5,
        delta=0.8,
        eta=0.2,
        max_mm_iterations=20,
        cg_tolerance=1.0e-12,
    )

    sparse_result = fit_begf(sparse_x0, config)
    dense_result = fit_begf(dense_x0, config)

    np.testing.assert_allclose(sparse_result.signal, dense_result.signal, atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(sparse_result.scales, dense_result.scales, atol=1.0e-12)
    np.testing.assert_allclose(sparse_result.weights, dense_result.weights, atol=1.0e-10)
    np.testing.assert_allclose(
        sparse_result.objective_history,
        dense_result.objective_history,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_fit_keeps_sparse_x0_in_production_operator(monkeypatch) -> None:
    sparse_x0, _ = build_x0(_partitions())
    captured: list[PartitionProjectionLaplacian] = []
    original = solver_module.PartitionProjectionLaplacian

    class CapturingOperator(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(solver_module, "PartitionProjectionLaplacian", CapturingOperator)
    fit_begf(sparse_x0, BEGFConfig(max_mm_iterations=1))

    assert len(captured) == 1
    assert sparse.isspmatrix_csr(captured[0].x0)


def test_subspace_preservation() -> None:
    sparse_x0, _ = build_x0(_partitions())
    result = fit_begf(
        sparse_x0,
        BEGFConfig(mu=0.5, delta=0.8, eta=0.2, max_mm_iterations=20),
    )
    dense_x0 = sparse_x0.toarray()
    coefficients, *_ = np.linalg.lstsq(dense_x0, result.signal, rcond=None)
    reconstructed = dense_x0 @ coefficients
    np.testing.assert_allclose(result.signal, reconstructed, atol=1.0e-9, rtol=1.0e-9)
