import numpy as np
from sklearn.decomposition import PCA
import mne
from sklearn.decomposition import FastICA
from mne import EpochsArray

def reconstruct_epochs_with_pca(epochs: mne.Epochs, n_components: int = 1) -> mne.Epochs:
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    X = data.transpose(0, 2, 1).reshape(-1, n_channels)            # (n_epochs*n_times, n_channels)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)                                   # (n_epochs*n_times, n_components)
    X_recon = pca.inverse_transform(scores)                         # (n_epochs*n_times, n_channels)

    data_recon = X_recon.reshape(n_epochs, n_times, n_channels).transpose(0, 2, 1)

    info     = epochs.info.copy()
    events   = epochs.events
    event_id = epochs.event_id
    tmin     = epochs.tmin

    return EpochsArray(data_recon, info, events, tmin, event_id)

def reconstruct_epochs_with_ica(epochs: mne.Epochs, n_components: int = 1) -> mne.Epochs:
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    X = data.transpose(0, 2, 1).reshape(-1, n_channels)  # (n_epochs * n_times, n_channels)
    ica = FastICA(n_components=n_components, max_iter=1000, tol=1e-4, random_state=42)
    sources = ica.fit_transform(X)                      # (n_epochs * n_times, n_components)
    X_recon = ica.inverse_transform(sources)            # (n_epochs * n_times, n_channels)

    data_recon = X_recon.reshape(n_epochs, n_times, n_channels).transpose(0, 2, 1)

    info     = epochs.info.copy()
    events   = epochs.events
    event_id = epochs.event_id
    tmin     = epochs.tmin

    return EpochsArray(data_recon, info, events, tmin, event_id)

def plot_everything(new_epochs):
    evoked_dict = {
        "Tapping/HbO": new_epochs["Tapping"].average(picks="hbo"),
        "Tapping/HbR": new_epochs["Tapping"].average(picks="hbr"),
        "Control/HbO": new_epochs["Control"].average(picks="hbo"),
        "Control/HbR": new_epochs["Control"].average(picks="hbr"),
    }

    # Rename channels until the encoding of frequency in ch_name is fixed
    for condition in evoked_dict:
        evoked_dict[condition].rename_channels(lambda x: x[:-4])

    color_dict = dict(HbO="#AA3377", HbR="b")
    styles_dict = dict(Control=dict(linestyle="dashed"))

    mne.viz.plot_compare_evokeds(
        evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict
    )

    times = np.arange(-3.5, 13.2, 3.0)
    topomap_args = dict(extrapolate="local")
    new_epochs["Tapping"].average(picks="hbo").plot_joint(
        times=times, topomap_args=topomap_args
    )

    times = np.arange(4.0, 11.0, 1.0)
    
    print("Tapping Left Hbo, Hbr")
    new_epochs["Tapping/Left"].average(picks="hbo").plot_topomap(times=times, **topomap_args)
    new_epochs["Tapping/Left"].average(picks="hbr").plot_topomap(times=times, **topomap_args)

    print("Tapping Right Hbo, Hbr")
    new_epochs["Tapping/Right"].average(picks="hbo").plot_topomap(times=times, **topomap_args)
    new_epochs["Tapping/Right"].average(picks="hbr").plot_topomap(times=times, **topomap_args)
    
    print("Control Hbo, Hbr")
    new_epochs["Control"].average(picks="hbo").plot_topomap(times=times, **topomap_args)
    new_epochs["Control"].average(picks="hbr").plot_topomap(times=times, **topomap_args)


# usage:
from finger_tapping.preprocessing import simple_pipeline
epochs = simple_pipeline(subject="01")
pca_epochs = reconstruct_epochs_with_pca(epochs, 1)
ica_epochs = reconstruct_epochs_with_ica(epochs, 1)

  
# plot_everything(epochs)
# # print("############## WHAT THE SIGMA ##############")
# # plot_everything(pca_epochs)
# print("############## WHAT THE SIGMA ##############")
# plot_everything(ica_epochs)


def plot_epoch_topomap(epochs, epoch_index, times):
    evoked = epochs[epoch_index].copy().average(picks="hbo")
    evoked.plot_topomap(times=times, extrapolate="local")

def plot_all(label="Control", epochs=None):
    for x in range(len(epochs[label])):
        if x==0 : print(f"######### {label} ##############")
        # print(epochs[label].events[:,2][x])
        plot_epoch_topomap(epochs[label], x, np.arange(4, 11, 1.0))
        
# plot_all('Control', pca_epochs)
# plot_all('Tapping/Left', pca_epochs)
# plot_all('Tapping/Right', pca_epochs)
