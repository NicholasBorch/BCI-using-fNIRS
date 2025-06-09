import numpy as np
import pytest
from finger_tapping.feature_preparation import extract_X_y
from finger_tapping.preprocessing import simple_pipeline

@pytest.fixture(scope="module")
def epochs():
    return simple_pipeline(subject="01", save=False)

def test_extract_X_y_returns_not_empty(epochs):
    X, y = extract_X_y(epochs)
    assert X.size > 0
    assert y.size > 0
    

def test_extract_X_y_returns_numpy_arrays(epochs):
    X, y = extract_X_y(epochs)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_extract_X_y_shapes_align(epochs):
    X, y = extract_X_y(epochs)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]

