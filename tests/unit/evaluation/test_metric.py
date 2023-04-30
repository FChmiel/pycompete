import pytest
from pycompete.evaluation.metrics import sharpe_ratio
import numpy as np


@pytest.mark.parametrize(
    "preds, target, moons, expected",
    [
        (
            np.array([0.8, 0.5, 0.3, 0.9, 0.25, 0.34, 0.27, 0.52]),
            np.array([1, 0, 1, 1, 0, 1, 1, 1]),
            np.array([1, 1, 1, 1, 2, 2, 2, 2]),
            3.8443069389471085,
        )
    ],
)
def test_sharpe_ratio(preds, target, moons, expected):
    result = sharpe_ratio(preds, target, moons)
    assert pytest.approx(result, rel=0.01) == expected
