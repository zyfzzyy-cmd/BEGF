"""Deterministic, data-free BEGF smoke demo."""

from __future__ import annotations

import numpy as np

from begf import BEGFConfig, PartitionProjectionLaplacian, build_x0, fit_begf


def main() -> None:
    rng = np.random.default_rng(2027)
    n_samples = 180
    n_classes = 3
    truth = np.repeat(np.arange(n_classes), n_samples // n_classes)
    partitions: list[np.ndarray] = []
    for _ in range(8):
        labels = truth.copy()
        corrupted = rng.random(n_samples) < 0.08
        labels[corrupted] = rng.integers(0, n_classes, size=int(corrupted.sum()))
        partitions.append(labels)

    x0, metadata = build_x0(partitions)
    operator = PartitionProjectionLaplacian(x0)
    result = fit_begf(
        x0,
        BEGFConfig(mu=0.5, delta=1.0, eta=0.1, max_mm_iterations=50),
    )

    print("X0 shape:", metadata["x0_shape"])
    print("operator:", operator.validate())
    print("filtered signal shape:", list(result.signal.shape))
    print("MM iterations:", len(result.iterations), "converged:", result.converged)
    print("objective:", f"{result.objective_history[0]:.6f} -> {result.objective_history[-1]:.6f}")
    print("final weights:", np.array2string(result.weights, precision=6))


if __name__ == "__main__":
    main()
