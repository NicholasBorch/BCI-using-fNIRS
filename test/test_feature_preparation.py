import numpy as np
import pytest
from finger_tapping.feature_preparation import extract_X_y
from finger_tapping.preprocessing import simple_pipeline

@pytest.fixture(scope="module")
def epochs():
    # use subject “01” and don’t write to disk
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
    # X must be 2-dim and y must be 1-dim, and their first dims must match
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]

def test_extract_X_y_label_values(epochs):
    _, y = extract_X_y(epochs)
    unique = set(y.tolist())
    # these are the labels hard-coded in extract_X_y
    assert unique == {"control", "left", "rigth"}
