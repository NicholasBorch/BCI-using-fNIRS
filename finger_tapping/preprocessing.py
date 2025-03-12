from itertools import compress
import numpy as np
from pathlib import Path
from typing import Optional

import mne
import mne_nirs
from mne_bids import BIDSPath, read_raw_bids
from mne.io import Raw
from mne import Epochs

FILTER_FREQS: dict[str, float] = {'low': 0.05, 
                                  'high':0.7,
                                  'l_trans_bandwidth': 0.02,
                                  'h_trans_bandwidth': 0.2}
NUM_SUBJECTS: int = 5
TASK: str = "tapping"
DATATYPE: str = "nirs"
ROOT: str = mne_nirs.datasets.fnirs_motor_group.data_path()

STIMULUS_DURATION: float = 5.0
TRIGGER_CODE: str = "15.0"

SHORT_CHANNELS_DIST: float = 0.01
SCI_THRESHOLD: float = 0.5

BEER_PPF: float = 0.1

REJECT_CRITERIA: dict[str, float] = {"hbo": 80e-6}

TMIN: float = -5.0
TMAX: float = 15.0

RENAME_DICT = {
        "Control": "Control",
        "Tapping/Left": "TappingLeft",
        "Tapping/Right": "TappingRight"
    }

def load_raw_intensity(subject: int) -> Raw:
    """Loads the raw intensity data for a given subject. Returns the raw intensity data."""
    bids_path = BIDSPath(subject=f'0{subject}', task=TASK, datatype=DATATYPE, root=ROOT)
    raw_intensity = read_raw_bids(bids_path, verbose=True)
    raw_intensity.load_data()
    return raw_intensity

def set_annotations(raw_intensity: Raw, stimulus_duration: Optional[float] = None, trigger_code: Optional[str] = None) -> Raw:
    """Sets the annotations for the raw intensity data. Returns the raw intensity data with annotations."""
    if stimulus_duration is None:
        stimulus_duration = STIMULUS_DURATION
    if trigger_code is None:
        trigger_code = TRIGGER_CODE
    raw_intensity.annotations.rename(RENAME_DICT)
    raw_intensity.annotations.set_durations(stimulus_duration)
    unwanted = np.nonzero(raw_intensity.annotations.description == trigger_code)
    raw_intensity.annotations.delete(unwanted)
    return raw_intensity

def remove_short_channels(raw_intensity: Raw, dist: Optional[float] = None) -> Raw:
    """
    Removes channels that are too close together.
    
    Parameters
    ----------
    raw_intensity : Raw
        The raw intensity data.
    dist : float
        The minimum distance between optodes (default=0.01) (1cm).
    
    Returns
    -------
    raw_intensity : Raw
        The raw intensity data with short channels removed.
    """
    if dist is None:
        dist = SHORT_CHANNELS_DIST
    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(raw_intensity.info, picks=picks)
    raw_intensity.pick(picks[dists > dist])
    return raw_intensity

def convert_to_od(raw_intensity: Raw) -> Raw:
    """Converts raw intensity values to optical density. Returns the raw optical density data."""
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    return raw_od

def set_bad_channels(raw_od: Raw, sci_threshold: Optional[float] = None) -> Raw:
    """
    Sets bad channels based on the scalp coupling index (SCI) of the raw optical density data.
    
    Parameters
    ----------
    raw_od : Raw
        The raw optical density data.
    sci_threshold : float
        The threshold for the scalp coupling index (default=0.5).
    
    Returns
    -------
    raw_od : Raw
        The raw optical density data with bad channels set.
    """
    if sci_threshold is None:
        sci_threshold = SCI_THRESHOLD
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < sci_threshold))
    return raw_od

def convert_to_haemoglobin(raw_od: Raw, ppf: Optional[float] = None) -> Raw:
    """Converts the optical density data to haemoglobin concentration using the modified Beer-Lambert Law. Returns the raw haemoglobin data."""
    if ppf is None:
        ppf = BEER_PPF
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=ppf)
    return raw_haemo

def bandpass_filter(raw_haemo: Raw, 
                    low: Optional[float] = None, 
                    high: Optional[float] = None,
                    l_trans_bandwidth: Optional[float] = None,
                    h_trans_bandwidth: Optional[float] = None) -> Raw:
    """Applies a bandpass filter to the haemoglobin data. Returns the filtered haemoglobin data."""
    if low is None:
        low = FILTER_FREQS['low']
    if high is None:
        high = FILTER_FREQS['high']
    if l_trans_bandwidth is None:
        l_trans_bandwidth = FILTER_FREQS["l_trans_bandwidth"]
    if h_trans_bandwidth is None:
        h_trans_bandwidth = FILTER_FREQS["h_trans_bandwidth"]
    raw_haemo.filter(low, high, h_trans_bandwidth=h_trans_bandwidth, l_trans_bandwidth=l_trans_bandwidth) #type: ignore
    return raw_haemo
        
def convert_annotations_to_events(raw_haemo: Raw) -> tuple[np.ndarray, dict]:
    """Converts the annotations to events. Returns the events and event dictionary."""
    events, event_dict = mne.events_from_annotations(raw_haemo)
    return events, event_dict

def get_epochs(raw_haemo: Raw, 
               events: np.ndarray, 
               event_dict: dict, 
               tmin: Optional[float] = None, 
               tmax: Optional[float] = None, 
               reject_criteria: Optional[dict] = None) -> mne.Epochs:
    """Epochs the haemoglobin data. Returns the epoched data."""
    if tmin is None:
        tmin = TMIN
    if tmax is None:
        tmax = TMAX
    if reject_criteria is None:
        reject_criteria = REJECT_CRITERIA

    epochs = mne.Epochs(
        raw_haemo,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        reject=reject_criteria,
        reject_by_annotation=True,
        proj=True,
        baseline=(None, 0),
        preload=True,
        detrend=None,
        verbose=True,
    )
    return epochs

def save_epochs(epochs: mne.Epochs, subject: int, out_dir_name: str = 'data') -> None:
    """Saves the epoched data."""
    out_dir = Path(out_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fname = out_dir / f"sub-0{subject}_preproc-epo.fif"
    epochs.save(str(save_fname), overwrite=True)
    
def raw_intensity_pipeline(subject: int) -> Raw:
    """Runs the full preprocessing pipeline on the raw intensity data for a given subject. Returns the epoched data."""
    raw_intensity = load_raw_intensity(subject)
    annotated_intensity = set_annotations(raw_intensity)
    short_channels_removed = remove_short_channels(annotated_intensity)
    return short_channels_removed


def simple_pipeline(subject: int, save: bool = True) -> Epochs:
    """Runs the full preprocessing pipeline on the raw intensity data for a given subject. Returns the epoched data.
    Runs pipeline based on default values specified at the top of the file.""" 
    intensity = raw_intensity_pipeline(subject = subject)

    # Converting raw intensity values to optical density
    raw_od = convert_to_od(intensity)

    raw_od = set_bad_channels(raw_od)
    
    # Converting the optical density data to haemoglobin concentration using the modified Beer-Lambert Law
    raw_haemo = convert_to_haemoglobin(raw_od)

    # Removing heart rate from signal (low-pass filter) and removing slow drifts (high-pass filter)
    raw_haemo = bandpass_filter(raw_haemo)

    events, event_dict = convert_annotations_to_events(raw_haemo)

    epochs = get_epochs(raw_haemo, events, event_dict)

    # # Saving data for further analysis
    if save:
        save_epochs(epochs, subject=1)
    return epochs


if __name__ == "__main__":
    simple_pipeline(subject=1)
