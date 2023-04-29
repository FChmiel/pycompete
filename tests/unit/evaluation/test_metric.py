import pytest
from pycompete.evaluation.metric import some_metric

# unit test for some_metric using pytest parametrize 
@pytest.mark.parametrize(
    "x, y, expected",
    [
        (1, 2, 3),
        (2, 3, 5),
        (3, 4, 7)]
)
def test_some_metric(x, y, expected):
    result = some_metric(x,y)
    assert result == expected