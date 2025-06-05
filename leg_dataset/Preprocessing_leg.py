from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import mne

bids_root = Path("../BIDS-NIRS-Tapping-0.1.0")

bids_path = BIDSPath(
    root=bids_root,
    subject="01",
    task="tapping",
    datatype="nirs",
    suffix="nirs",
    extension=".snirf"
)

raw_finger = read_raw_bids(bids_path=bids_path, verbose=True)
raw_finger.load_data()

print(raw_finger)


# Path
bids_root = Path("/Users/vietnguyen/DTU/BCI-using-fNIRS/leg_dataset/bids_raw")

# Construct Path
bids_path = BIDSPath(
    root=bids_root,
    subject="06",             # ← sub-06
    task="fingerauto",    # ← from the filename
    datatype="nirs",
    suffix="nirs",
    extension=".snirf"         # SNIRF container
)
# Read and load the raw intensity data
raw_intensity = read_raw_bids(bids_path=bids_path, verbose=True)
raw_intensity.load_data()


raw_intensity.annotations.description

import pandas as pd
df = pd.read_csv('bids_raw/sub-06/nirs/sub-06_task-fingerauto_events.tsv', sep="\t")

annotations = mne.Annotations(df['onset'], df['duration'], df['event_type'])

raw_intensity.set_annotations(annotations)

raw_intensity.set_channel_types({ch: 'fnirs_cw_amplitude' for ch in raw_intensity.ch_names})
picks = mne.pick_types(raw_intensity.info, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(raw_intensity.info, picks=picks)

# fail
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)

for s in raw_intensity.ch_names:
    s = s.replace('Rx', 'S')
    s = s.replace('Tx', 'D')
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('nm', '')
    s = s.strip()
    print(s)

    