import numpy as np
from scipy import sparse

from begf import BEGFConfig, PartitionProjectionLaplacian, build_x0, fit_begf


def _x0() -> sparse.csr_matrix:
    labels = [
        np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0]),
        np.array([0, 1, 1, 0, 2, 2, 1, 0, 2, 0]),
        np.array([1, 0, 1, 2, 2, 0, 0, 1, 2, 0]),
        np.array([0, 0, 2, 1, 2, 1, 0, 1, 2, 0]),
    ]
    return build_x0(labels)[0]


def test_eta_one_recovers_quadratic_filter() -> None:
    x0 = _x0()
    dense_x0 = x0.toarray()
    mu = 0.7
    result = fit_begf(
        x0,
        BEGFConfig(
            mu=mu,
            delta=1.0,
            eta=1.0,
            max_mm_iterations=50,
            mm_tolerance=1.0e-12,
            cg_tolerance=1.0e-12,
        ),
    )
    operator = PartitionProjectionLaplacian(x0)
    dense_laplacian = 2.0 * (np.eye(x0.shape[0]) - dense_x0 @ dense_x0.T)
    expected = np.linalg.solve(np.eye(x0.shape[0]) + mu * dense_laplacian, dense_x0)
    np.testing.assert_allclose(result.signal, expected, atol=1.0e-9)
    np.testing.assert_allclose(result.weights, np.ones(3), atol=1.0e-12)
    assert result.objective_history[-1] <= result.objective_history[0] + 1.0e-10
    assert operator.validate()["laplacian_source"] == "partition_projection"


def test_adaptive_solver_returns_bounded_weights_and_monotone_objective() -> None:
    result = fit_begf(
        _x0(),
        BEGFConfig(mu=0.5, delta=0.8, eta=0.2, max_mm_iterations=30),
    )
    assert result.signal.flags.writeable is False
    assert np.all(result.weights >= 0.2 - 1.0e-12)
    assert np.all(result.weights <= 1.0 + 1.0e-12)
    differences = np.diff(np.asarray(result.objective_history))
    assert np.all(differences <= 1.0e-8)
    assert result.iterations
