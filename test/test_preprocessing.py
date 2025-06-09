import pytest
from mne import Epochs
from finger_tapping.preprocessing import (
    raw_intensity_pipeline,
    preprocessing_pipeline,
    simple_pipeline,
    SUBJECTS
)

@pytest.fixture(scope="module")
def example_raw():
    return raw_intensity_pipeline(SUBJECTS[0])

def test_preprocessing_pipeline_returns_epochs(example_raw):
    epochs = preprocessing_pipeline(example_raw, SUBJECTS[0], save=False)
    assert isinstance(epochs, Epochs)
    assert len(epochs) > 0

def test_simple_pipeline_returns_epochs():
    epochs = simple_pipeline(SUBJECTS[0], save=False)
    assert isinstance(epochs, Epochs)
    assert len(epochs) > 0
