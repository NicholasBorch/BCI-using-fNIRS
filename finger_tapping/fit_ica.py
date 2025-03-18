import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne

# Parameters to set for ICA
N_COMPONENTS = 10       ### Number of ICA components to compute. Cannot exceed number of channels.
RANDOM_STATE = 42
MAX_ITER = 800

def fit_ica(use_hbo_only: bool = True, plot: bool = False, data_path: str = ""):
    """
    fit_ica performs Independent Component Analysis on preprocessed fNIRS data.
    
    Parameters:
    -----------
    use_hbo_only : bool
        If True, only HbO channels are selected for analysis.
    plot : bool
        If True, ICA components are visualized.
    data_path : str
        Path to preprocessed epochs data.
    """
    # Loading preprocessed epochs data.
    epochs = mne.read_epochs(str(data_path), preload=True)
    print(f"Loaded epochs with shape: {epochs.get_data().shape}")
    
    # Option to only use HbO channels.
    if use_hbo_only:
        hbo_channels = [ch for ch in epochs.ch_names if "hbo" in ch.lower()]
        epochs = epochs.pick_channels(hbo_channels)
        print(f"Using only HbO channels: {hbo_channels}")
    else:
        print("Using all channels (HbO and HbR).")
    
    
    raw = mne.concatenate_epochs([epochs])
    print(f"Concatenation done with shape: {raw.get_data().shape}")
    
    ica = mne.preprocessing.ICA(n_components=N_COMPONENTS, method='fastica',
                                  random_state=RANDOM_STATE, max_iter=MAX_ITER)
    ica.fit(raw)
    print("ICA fitting complete.")
    
    # Save
    ica_save_path = Path("Data") / "ica_fitted.fif"
    ica.save(str(ica_save_path))
    print(f"ICA fit saved to: {ica_save_path}")
    
    # Visualizing ICA components.
    if plot == True:
        ica.plot_components(inst=raw, show=True)
        ica.plot_sources(raw, show_scrollbars=True, block=True)

if __name__ == "__main__":
    USE_HBO_ONLY = True
    data_path = Path("Data") / "sub-01_preproc-epo.fif"
    fit_ica(use_hbo_only=USE_HBO_ONLY, plot=True, data_path=data_path)
