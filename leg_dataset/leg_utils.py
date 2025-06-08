from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import mne
import numpy as np
import re
import pandas as pd
from finger_tapping.preprocessing import preprocessing_pipeline


def convert_tsv_bak(bids_root):
    bids_root = Path(bids_root)
    for sub_folder in bids_root.glob("sub-*"):
        if sub_folder.is_dir():
            sub_id = sub_folder.name
            scans_tsv = sub_folder / f"{sub_id}_scans.tsv"
            
            if scans_tsv.exists():
                scans_tsv_bak = scans_tsv.with_suffix(".tsv.bak")
                scans_tsv.rename(scans_tsv_bak)
                print(f"Renamed {scans_tsv} to {scans_tsv_bak}")
            else:
                print(f"No scans.tsv found in {sub_folder}")
    print("Done.")


def load_snirf(bids_root_path, subject, task):
    """ Loads snirf file from folder, subject, and task
    bids_root_path: 'BIDS-NIRS-Tapping-0.1.0' or 'bids_raw'
    subject: [01,02,...,05] or [06,10,...,95]
    task: 'tapping' or ['fingerauto', 'fingerautodual', 'footauto', 'footautodual'....]
    """
    bids_root = Path(bids_root_path)
    
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject,
        task=task,
        datatype="nirs",
        suffix="nirs",
        extension=".snirf"
    )
    
    raw_intensity = read_raw_bids(bids_path, verbose=True)
    raw_intensity.load_data()
    
    return raw_intensity

def set_annotatations(raw_intensity:mne.io.Raw, subject:str, task:str)->mne.io.Raw:
    """ Documentation is for stupid people"""
    df = pd.read_csv(f'bids_raw/sub-{subject}/nirs/sub-06_task-{task}_events.tsv', sep="\t")
    annotations = mne.Annotations(df['onset'], df['duration'], df['event_type'])
    raw_intensity.set_annotations(annotations)
    
    return raw_intensity


def set_channels(raw_intensity:mne.io.Raw)->mne.io.Raw:
    """ Don't judge my code"""
    # Set type
    raw_intensity.set_channel_types({ch: 'fnirs_cw_amplitude' for ch in raw_intensity.ch_names})

    # Rename channel names
    ch_list = []
    for s in raw_intensity.ch_names:
        s = s.replace('Rx', 'S')
        s = s.replace('Tx', 'D')
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = s.replace('nm', '')
        s = s.replace('-','_')
        s = re.sub(r'[a-z]', '', s)
        s = s.strip()
        s = s.split()
        s[1] = '850' if int(s[1]) > 800 else '760'
        s = ' '.join(s)
        ch_list.append(s)
        
    channels_mapping = dict(zip(raw_intensity.ch_names, ch_list))
    raw_intensity.rename_channels(channels_mapping)
    return raw_intensity


def run_tapping_pipeline(subject: str, task: str, bids_root_path: str, save:bool):
    
    intensity = load_snirf(bids_root_path=bids_root_path, subject=subject, task=task)
    intensity = set_annotatations(intensity, subject, task)
    intensity = set_channels(intensity)
    
    epochs = preprocessing_pipeline(intensity, subject, save)
    return epochs



if __name__ == '__main__':
    # # Only run once: convert_tsv_bak()
    # convert_tsv_bak(bids_root="BIDS-NIRS-Tapping-0.1.0")
    
    # Example 1: Load raw SNIRF and print basic info
    raw_intensity = load_snirf(bids_root_path='bids_raw', subject='06', task='fingerauto')
    raw_intensity = set_annotatations(raw_intensity, subject='06', task='fingerauto')
    raw_intensity = set_channels(raw_intensity)

    print("Annotations:", raw_intensity.annotations.description)
    print("Channel names:", raw_intensity.ch_names)

    # Example 2: Run full pipeline
    epochs = run_tapping_pipeline(subject='06', task='fingerauto', bids_root_path='bids_raw', save=False)
    print(f"Number of epochs: {len(epochs)}")

