from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mne
import mne_bids.stats
import mne_nirs
from mne_bids import BIDSPath, read_raw_bids


def main():
    # Loading the Finger Tapping dataset from MNE in BIDS format
    bids_root = mne_nirs.datasets.fnirs_motor_group.data_path()

    # Selecting a subject (e.g., sub-01)
    bids_path = BIDSPath(subject="01", task="tapping", datatype="nirs", root=bids_root)

    # Loading fNIRS data in SNIRF format
    raw_intensity = read_raw_bids(bids_path, verbose=True)
    raw_intensity.load_data()

    # Include info about duration of each stimulus (5 seconds) and remove the trigger code 15,
    # which signals the start and end of the experiment and is not relevant.
    rename_dict = {
        "Control": "Control",
        "Tapping/Left": "TappingLeft",
        "Tapping/Right": "TappingRight"
    }
    raw_intensity.annotations.rename(rename_dict)
    raw_intensity.annotations.set_durations(5)
    unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
    raw_intensity.annotations.delete(unwanted)

    # Remove channels that are too close together (short channels: < 1 cm distance between optodes)
    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(raw_intensity.info, picks=picks)
    raw_intensity.pick(picks[dists > 0.01])

    # Converting raw intensity values to optical density
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    # Marking all channels with a scalp coupling index (SCI) less than 0.5 as bad
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))

    # Converting the optical density data to haemoglobin concentration using the modified Beer-Lambert Law
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

    # Removing heart rate from signal (low-pass filter) and removing slow drifts (high-pass filter)
    raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

    # Converting annotations to events, then epoching the data
    # Here we create 20s epochs (−5 to +15s around each event)
    # We also apply a simple reject criterion for large HbO artifacts
    events, event_dict = mne.events_from_annotations(raw_haemo)
    print(f"Event dict: {event_dict}")

    reject_criteria = dict(hbo=80e-6)
    tmin, tmax = -5.0, 15.0

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

    # Saving data for further analysis
    out_dir = Path("Data")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fname = out_dir / "sub-01_preproc-epo.fif"
    epochs.save(str(save_fname), overwrite=True)
    print(f"Preprocessed epochs saved to: {save_fname}")


if __name__ == "__main__":
    main()
