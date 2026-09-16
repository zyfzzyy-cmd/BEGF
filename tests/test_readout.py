import numpy as np
import pytest

from begf import cluster_embedding, row_l2_normalize


def test_readout_normalizes_rows_and_is_deterministic() -> None:
    embedding = np.array(
        [[3.0, 4.0], [6.0, 8.0], [4.0, 0.0], [8.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    normalized = row_l2_normalize(embedding)
    np.testing.assert_allclose(np.linalg.norm(normalized[:4], axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(normalized[4], np.zeros(2), atol=1.0e-12)

    labels_a = cluster_embedding(embedding, 2, random_state=19, n_init=20)
    labels_b = cluster_embedding(embedding, 2, random_state=19, n_init=20)
    assert labels_a.shape == (5,)
    assert np.issubdtype(labels_a.dtype, np.integer)
    assert labels_a.min() >= 0
    assert labels_a.max() < 2
    np.testing.assert_array_equal(labels_a, labels_b)


def test_readout_validates_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        cluster_embedding(np.ones(4), 2)
    with pytest.raises(ValueError, match="finite"):
        cluster_embedding(np.array([[0.0, np.nan]]), 1)
    with pytest.raises(ValueError, match="positive"):
        cluster_embedding(np.ones((2, 2)), 0)
