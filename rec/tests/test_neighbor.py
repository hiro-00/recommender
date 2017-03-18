import pytest
import numpy as np
from rec.cf.neighbor import NeighborhoodModel

@pytest.fixture()
def model():
    yield NeighborhoodModel()


def test_fill(model):
    dataset = np.array([[1,2,3],
                     [0,-1,1]])
    result = model.fill(dataset)
    assert result[1,1] != -1