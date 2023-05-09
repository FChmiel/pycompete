import pytest
from pycompete.features.transformers import RemoveCorrelated
import numpy as np


@pytest.mark.parametrize(
    "X, X_expected, max_correlation",
    [
        (
            np.array([[0, 0.1, 0.04], [0.4, 0.45, -1], [0.78, 0.81, 2]]),
            np.array([[0, 0.04], [0.4, -1], [0.78, 2]]),
            0.95,
        ),
        (
            np.array([[0, 0.1, 0.04], [0.4, 0.45, -1], [0.78, 0.81, 2]]),
            np.array([[0, 0.1, 0.04], [0.4, 0.45, -1], [0.78, 0.81, 2]]),
            1,
        ),
    ],
)
def test_remove_correlated(X, X_expected, max_correlation):
    rc = RemoveCorrelated(max_correlation=max_correlation)
    rc.fit(X)
    X_transformed = rc.transform(X)
    assert np.allclose(X_transformed, X_expected, rtol=1e-05)
