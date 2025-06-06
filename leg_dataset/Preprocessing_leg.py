from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import mne

bids_root = Path("BIDS-NIRS-Tapping-0.1.0")

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
    subject="06",             # sub-06
    task="fingerauto",    # from the filename
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



ch_list = ['S2_D4 750', 'S2_D4 850',
 'S2_D3 750', 'S2_D3 850',
 'S2_D1 750', 'S2_D1 850',
 'S1_D3 750', 'S1_D3 850',
 'S1_D2 750', 'S1_D2 850',
 'S1_D1 750', 'S1_D1 850',
 'S4_D4 750', 'S4_D4 850',
 'S4_D3 750', 'S4_D3 850',
 'S4_D5 750', 'S4_D5 850',
 'S4_D1 750', 'S4_D1 850',
 'S3_D3 750', 'S3_D3 850',
 'S3_D2 750', 'S3_D2 850',
 'S3_D5 750', 'S3_D5 850',
 'S3_D1 750', 'S3_D1 850',
 'S6_D9 750', 'S6_D9 850',
 'S6_D6 750', 'S6_D6 850',
 'S5_D8 750', 'S5_D8 850',
 'S5_D7 750', 'S5_D7 850',
 'S5_D6 750', 'S5_D6 850',
 'S8_D9 750', 'S8_D9 850',
 'S8_D10 750', 'S8_D10 850',
 'S8_D6 750', 'S8_D6 850',
 'S7_D8 750', 'S7_D8 850',
 'S7_D7 750', 'S7_D7 850',
 'S7_D6 750', 'S7_D6 850',
 'S9_D12 750', 'S9_D12 850',
 'S9_D13 750', 'S9_D13 850',
 'S9_D11 750', 'S9_D11 850',
 'S11_D12 750', 'S11_D12 850',
 'S11_D13 750', 'S11_D13 850',
 'S11_D11 750', 'S11_D11 850',
 'S10_D14 750', 'S10_D14 850',
 'S10_D11 750', 'S10_D11 850',
 'S12_D14 750', 'S12_D14 850',
 'S12_D15 750', 'S12_D15 850',
 'S12_D11 750', 'S12_D11 850']



# for s in raw_intensity.ch_names:
#     s = s.replace('Rx', 'S')
#     s = s.replace('Tx', 'D')
#     s = s.replace('[', '')
#     s = s.replace(']', '')
#     s = s.replace('nm', '')
#     s = s.replace('-','_')
#     s = s.strip()
#     ch_list.append(s)
    

temp_channels = dict(zip(raw_intensity.ch_names, ch_list))
raw_intensity.rename_channels(temp_channels)


# fail
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)