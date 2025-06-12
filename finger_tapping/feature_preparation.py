import numpy as np
from mne import Epochs, pick_types
import pandas as pd

def _select_channels(epochs: Epochs, include_hbr: bool = True) -> Epochs:
    """
    Returns a copy of "epochs" that contains only the wanted fNIRS channel types.
    If "include_hbr" is False we keep HbO channels only.
    """
    if include_hbr:
        return epochs
    hbo_picks = pick_types(epochs.info, fnirs='hbo')
    return epochs.copy().pick(hbo_picks)

def split_activities(subject: Epochs) -> tuple[Epochs, Epochs, Epochs]:
    """ Takes the subject activities and splits it into categories"""
    control = subject['Control']
    left = subject['Tapping/Left']
    right = subject['Tapping/Right']
    return control, left, right

def get_labels_numeric(subject: Epochs) -> np.ndarray:
    """Gets numeric labels from activities for 1 subject"""
    return subject.events[:, -1]

def get_minimum_bound(control: Epochs, left: Epochs, right: Epochs) -> int:
    """Find the activity with min amount of epochs """
    return np.min([x.get_data().shape[0] for x in [control, left, right]])


### Old reshape_activity function, kept for reference
# def reshape_activity(epoch: Epochs, min_bound: int) -> np.ndarray:
#     """Limit epochs to min bound, reshape data to channels x merged_epochs """
#     epoch_data = epoch.get_data()[:min_bound,:,:]
#     n_epoch, n_channels, n_epoch_size = epoch_data.shape
#     epoch_data_reshaped = epoch_data.reshape(n_channels, n_epoch * n_epoch_size)
    
#     return epoch_data_reshaped.T

def reshape_activity(epoch: Epochs, min_bound: int, activity: str) -> np.ndarray:
    """Limit epochs to min bound, reshape data to channels x merged_epochs """
    epoch_data = epoch.get_data()[:min_bound,:,:]
    n_epoch, n_channels, n_epoch_size = epoch_data.shape
    epoch_data_reshaped = epoch_data.swapaxes(1,2).reshape(n_epoch * n_epoch_size, n_channels)

    epoch_idx = np.repeat(np.arange(min_bound), n_epoch_size)
    time_idx = np.tile(np.arange(n_epoch_size), min_bound)
    act = np.full(epoch_idx.shape, activity)
    
    return epoch_data_reshaped, act, epoch_idx, time_idx

def create_labels(labels: list[str], lenght: int) -> np.ndarray:
    """ Creates labels for data"""
    init_list = []
    for label in labels:
        init_list.append(np.full(lenght, label))
    return np.concatenate(init_list)

def extract_X_y(subject: Epochs, include_hbr: bool = True, return_index: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
    """
    Combines all functions in one function and creates X and y ~ (features and labels)
    """

    subject = _select_channels(subject, include_hbr=include_hbr)
    control, left, right = split_activities(subject)
    # min_bound = get_minimum_bound(control, left, right)
    min_bound = min(x.get_data().shape[0] for x in (control, left, right))

    
    control_reshaped, act_control, ep_control, t_control = reshape_activity(control, min_bound, activity="control")
    left_reshaped, act_left, ep_left, t_left = reshape_activity(left, min_bound, activity="left")
    right_reshaped, act_right, ep_right, t_right = reshape_activity(right, min_bound, activity="right")
    
    X = np.vstack([control_reshaped, left_reshaped, right_reshaped])
    y = np.concatenate([act_control, act_left, act_right])
    # X = np.concatenate([control_reshaped, left_reshaped, right_reshaped], axis=0)

    # # Create labels and define y
    # lenght = control_reshaped.shape[0]
    # labels = ['control', 'left','rigth']
    # y = create_labels(labels, lenght)
    # return X, y

    if not return_index:
        return X, y

    index = pd.MultiIndex.from_arrays(
        [np.concatenate([act_control, act_left, act_right]), np.concatenate([ep_control, ep_left, ep_right]), np.concatenate([t_control, t_left, t_right])], names=["activity", "epoch", "t_in_epoch"])
    return X, y, index
    
def sliding_window(arr, window_size, step=1):
    """ Sliding window function. Down sampling by using the mean of each window. Takes windows size and step size into consideration"""
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_size)
    windows = windows[::step]
    return windows.mean(axis=1)

if __name__ == '__main__':
    from finger_tapping.preprocessing import simple_pipeline
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import FastICA
    import pandas as pd

    # Load and create features
    subject = simple_pipeline(subject="01")
    X, y = extract_X_y(subject, False, False)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Unique labels:", np.unique(y))

    # Standardizing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # pca testing
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df['label'] = y
    print(f"PCA output shape: {X_pca.shape}")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print(pca_df.head())

    # ica testing
    ica = FastICA(n_components=5, max_iter=1000, tol=0.0001, random_state=42)
    X_ica = ica.fit_transform(X_scaled)
    ica_df = pd.DataFrame(X_ica, columns=[f"IC{i+1}" for i in range(X_ica.shape[1])])
    ica_df['label'] = y
    print(f"ICA output shape: {X_ica.shape}")
    print(ica_df.head())