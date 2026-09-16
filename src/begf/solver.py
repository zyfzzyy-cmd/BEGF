"""Convex pseudo-Huber MM solver for the public BEGF implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from .ensemble import build_x0
from .operators import PartitionProjectionLaplacian


@dataclass(frozen=True)
class BEGFConfig:
    """Numerical parameters for the convex MM/CG updates."""

    mu: float = 1.0
    delta: float = 1.0
    eta: float = 0.1
    epsilon: float = 1.0e-12
    max_mm_iterations: int = 50
    mm_tolerance: float = 1.0e-6
    cg_tolerance: float = 1.0e-10
    cg_absolute_tolerance: float = 0.0
    cg_max_iterations: int | None = None
    monotonicity_tolerance: float = 1.0e-10
    max_cg_refinements: int = 3


@dataclass(frozen=True)
class BEGFIteration:
    """Auditable state recorded after each accepted MM update."""

    iteration: int
    objective: float
    relative_change: float
    energies: NDArray[np.float64]
    weights: NDArray[np.float64]
    cg_iterations: int
    relative_linear_residual: float


@dataclass(frozen=True)
class BEGFResult:
    """Result of fitting BEGF to a normalized ensemble representation."""

    signal: NDArray[np.float64]
    scales: NDArray[np.float64]
    weights: NDArray[np.float64]
    objective_history: tuple[float, ...]
    iterations: tuple[BEGFIteration, ...]
    converged: bool


def _validate_config(config: BEGFConfig) -> None:
    for name in ("mu", "delta", "epsilon", "mm_tolerance", "cg_tolerance"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("cg_absolute_tolerance", "monotonicity_tolerance"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.0 <= float(config.eta) <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    if not isinstance(config.max_mm_iterations, int) or config.max_mm_iterations < 1:
        raise ValueError("max_mm_iterations must be a positive integer")
    if config.cg_max_iterations is not None and (
        not isinstance(config.cg_max_iterations, int) or config.cg_max_iterations < 1
    ):
        raise ValueError("cg_max_iterations must be None or a positive integer")
    if not isinstance(config.max_cg_refinements, int) or config.max_cg_refinements < 0:
        raise ValueError("max_cg_refinements must be a nonnegative integer")


def pseudo_huber(z: ArrayLike, *, delta: float, eta: float) -> NDArray[np.float64]:
    """Evaluate the paper's convex pseudo-Huber penalty on nonnegative inputs."""
    values = np.asarray(z, dtype=np.float64)
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError("pseudo-Huber inputs must be finite and nonnegative")
    return 0.5 * eta * values**2 + (1.0 - eta) * delta**2 * (
        np.sqrt(1.0 + (values / delta) ** 2) - 1.0
    )


def band_weights(
    energies: ArrayLike, scales: ArrayLike, *, delta: float, eta: float
) -> NDArray[np.float64]:
    """Compute the bounded MM weights from normalized current band energies."""
    current = np.asarray(energies, dtype=np.float64)
    reference = np.asarray(scales, dtype=np.float64)
    if current.shape != (3,) or reference.shape != (3,):
        raise ValueError("energies and scales must be length-three vectors")
    if np.any(current < 0.0) or np.any(reference <= 0.0):
        raise ValueError("energies must be nonnegative and scales must be positive")
    return eta + (1.0 - eta) / np.sqrt(1.0 + current / (reference * delta**2))


def _objective(
    operator: PartitionProjectionLaplacian,
    signal: NDArray[np.float64],
    x0: NDArray[np.float64],
    scales: NDArray[np.float64],
    config: BEGFConfig,
) -> float:
    energies = operator.band_energies(signal)
    normalized = np.sqrt(energies / scales)
    fidelity = 0.5 * float(np.sum((signal - x0) ** 2))
    penalty = config.mu * float(np.sum(scales * pseudo_huber(normalized, delta=config.delta, eta=config.eta)))
    return fidelity + penalty


def _cg_multi_rhs(
    apply_system: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    rhs: NDArray[np.float64],
    *,
    tolerance: float,
    absolute_tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], int, float]:
    """Solve one SPD operator against multiple columns using matrix-free CG."""
    solution = np.zeros_like(rhs)
    residual = rhs - apply_system(solution)
    rhs_norm = np.linalg.norm(rhs, axis=0)
    thresholds = absolute_tolerance + tolerance * rhs_norm
    residual_squared = np.sum(residual * residual, axis=0)
    done = np.sqrt(residual_squared) <= thresholds
    direction = residual.copy()
    direction[:, done] = 0.0
    last_iteration = 0

    for iteration in range(1, max_iterations + 1):
        last_iteration = iteration
        if np.all(done):
            break
        action = apply_system(direction)
        denominator = np.sum(direction * action, axis=0)
        active = ~done
        if np.any(~np.isfinite(denominator[active])) or np.any(denominator[active] <= 0.0):
            raise RuntimeError("CG encountered a non-positive direction curvature")
        alpha = np.zeros(rhs.shape[1], dtype=np.float64)
        alpha[active] = residual_squared[active] / denominator[active]
        solution += direction * alpha[None, :]
        residual -= action * alpha[None, :]
        new_squared = np.sum(residual * residual, axis=0)
        new_norm = np.sqrt(new_squared)
        new_done = new_norm <= thresholds
        if np.all(new_done):
            residual_squared = new_squared
            done = new_done
            break
        beta = np.zeros(rhs.shape[1], dtype=np.float64)
        nonzero = active & (residual_squared > 0.0)
        beta[nonzero] = new_squared[nonzero] / residual_squared[nonzero]
        direction = residual + direction * beta[None, :]
        direction[:, new_done] = 0.0
        residual_squared = new_squared
        done = new_done

    final_norm = np.sqrt(residual_squared)
    relative_residual = float(np.max(final_norm / np.maximum(1.0, rhs_norm)))
    if not np.all(final_norm <= thresholds):
        raise RuntimeError(
            "CG did not reach its requested residual tolerance: "
            f"max_relative_residual={relative_residual}"
        )
    return solution, last_iteration, relative_residual


def fit_begf(
    x0: ArrayLike | sparse.spmatrix, config: BEGFConfig | None = None
) -> BEGFResult:
    """Fit BEGF to ``X0`` using the paper's matrix-free MM/CG updates."""
    settings = config if config is not None else BEGFConfig()
    _validate_config(settings)
    operator = PartitionProjectionLaplacian(x0)
    x0_factor = operator.x0
    if sparse.issparse(x0_factor):
        x0_sparse = x0_factor
        # Keep sparse X0 inside the operator for graph/Laplacian actions. This
        # dense n-by-q copy is only the multi-RHS linear-system RHS.
        rhs_dense = np.asarray(x0_sparse.toarray(), dtype=np.float64)
    else:
        rhs_dense = np.asarray(x0_factor, dtype=np.float64)
    scales = operator.band_energies(rhs_dense) + settings.epsilon
    current = rhs_dense.copy()
    objective = _objective(operator, current, rhs_dense, scales, settings)
    objective_history = [objective]
    records: list[BEGFIteration] = []
    converged = False
    max_cg_iterations = settings.cg_max_iterations or max(100, min(1000, 10 * operator.n_samples))

    for iteration in range(1, settings.max_mm_iterations + 1):
        energies = operator.band_energies(current)
        weights = band_weights(energies, scales, delta=settings.delta, eta=settings.eta)
        accepted: tuple[NDArray[np.float64], int, float, float] | None = None
        for refinement in range(settings.max_cg_refinements + 1):
            cg_tolerance = settings.cg_tolerance * (0.1**refinement)

            def apply_system(values: NDArray[np.float64]) -> NDArray[np.float64]:
                return values + settings.mu * operator.weighted_band_action(values, weights)

            candidate, cg_iterations, relative_residual = _cg_multi_rhs(
                apply_system,
                rhs_dense,
                tolerance=cg_tolerance,
                absolute_tolerance=settings.cg_absolute_tolerance,
                max_iterations=max_cg_iterations,
            )
            candidate_objective = _objective(operator, candidate, rhs_dense, scales, settings)
            allowed = objective + settings.monotonicity_tolerance * max(1.0, abs(objective))
            if candidate_objective <= allowed:
                accepted = (candidate, cg_iterations, relative_residual, candidate_objective)
                break
        if accepted is None:
            raise RuntimeError("MM objective increased after the requested CG refinements")

        candidate, cg_iterations, relative_residual, candidate_objective = accepted
        relative_change = float(
            np.linalg.norm(candidate - current, ord="fro")
            / max(1.0, float(np.linalg.norm(current, ord="fro")))
        )
        records.append(
            BEGFIteration(
                iteration=iteration,
                objective=candidate_objective,
                relative_change=relative_change,
                energies=energies.copy(),
                weights=weights.copy(),
                cg_iterations=cg_iterations,
                relative_linear_residual=relative_residual,
            )
        )
        current = candidate
        objective = candidate_objective
        objective_history.append(objective)
        if relative_change < settings.mm_tolerance:
            converged = True
            break

    final_weights = band_weights(operator.band_energies(current), scales, delta=settings.delta, eta=settings.eta)
    current.setflags(write=False)
    scales.setflags(write=False)
    final_weights.setflags(write=False)
    return BEGFResult(
        signal=current,
        scales=scales,
        weights=final_weights,
        objective_history=tuple(objective_history),
        iterations=tuple(records),
        converged=converged,
    )


def fit_begf_from_partitions(
    partitions: list[ArrayLike] | tuple[ArrayLike, ...] | NDArray[np.generic],
    config: BEGFConfig | None = None,
) -> tuple[BEGFResult, dict[str, object]]:
    """Build ``X0`` from labels and run BEGF, returning construction metadata."""
    x0, metadata = build_x0(partitions)
    return fit_begf(x0, config), metadata
