import numpy as np
import pytest

from begf import PartitionProjectionLaplacian, build_x0


def _example_x0() -> np.ndarray:
    partitions = [
        np.array([0, 0, 1, 1, 2, 2, 0, 1]),
        np.array([1, 1, 0, 0, 2, 2, 1, 0]),
        np.array([0, 2, 1, 1, 0, 2, 0, 1]),
    ]
    x0, _ = build_x0(partitions)
    return x0


def test_build_x0_uses_normalized_partition_blocks() -> None:
    x0 = _example_x0()
    assert x0.shape == (8, 9)
    np.testing.assert_allclose(np.sum(x0 * x0), 3.0, atol=1.0e-12)

    first_partition = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    counts = np.bincount(first_partition)
    expected_first_block = np.zeros((8, 3), dtype=float)
    expected_first_block[np.arange(8), first_partition] = 1.0
    expected_first_block /= np.sqrt(counts)[None, :]
    np.testing.assert_allclose(x0[:, :3], expected_first_block / np.sqrt(3.0))


def test_matrix_free_laplacian_and_bands_match_small_reference() -> None:
    x0 = _example_x0()
    operator = PartitionProjectionLaplacian(x0)
    dense_laplacian = 2.0 * (np.eye(x0.shape[0]) - x0 @ x0.T)
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(x0.shape[0], 4))

    np.testing.assert_allclose(operator.apply_laplacian(signal), dense_laplacian @ signal, atol=1.0e-12)
    z1 = dense_laplacian @ signal
    z2 = dense_laplacian @ z1
    z3 = dense_laplacian @ z2
    expected_bands = (z1 - z2 + 0.25 * z3, z2 - 0.5 * z3, 0.25 * z3)
    for actual, expected in zip(operator.apply_band_operators(signal), expected_bands):
        np.testing.assert_allclose(actual, expected, atol=1.0e-11)
    np.testing.assert_allclose(sum(operator.apply_band_operators(signal)), dense_laplacian @ signal, atol=1.0e-11)
    np.testing.assert_allclose(
        operator.band_energies(signal).sum(),
        np.sum(signal * (dense_laplacian @ signal)),
        atol=1.0e-10,
    )


def test_spectrum_contract_and_scipy_adapter() -> None:
    x0 = _example_x0()
    operator = PartitionProjectionLaplacian(x0)
    dense_laplacian = 2.0 * (np.eye(x0.shape[0]) - x0 @ x0.T)
    eigenvalues = np.linalg.eigvalsh(dense_laplacian)
    assert eigenvalues.min() >= -1.0e-10
    assert eigenvalues.max() <= 2.0 + 1.0e-10
    diagnostics = operator.validate()
    assert diagnostics["formula"] == "2*(I-X0*X0.T)"
    assert diagnostics["matrix_free"] is True
    assert diagnostics["theorem_premise_verified"] is True

    signal = np.arange(x0.shape[0], dtype=float)
    np.testing.assert_allclose(operator.as_linear_operator() @ signal, operator.apply_laplacian(signal))


def test_theorem_premise_is_checked() -> None:
    with pytest.raises(ValueError, match="theorem premise"):
        PartitionProjectionLaplacian(1.01 * np.eye(3))
