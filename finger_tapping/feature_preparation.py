import numpy as np
from mne import Epochs

# def split_activities(subject: Epochs) -> tuple[Epochs, Epochs, Epochs]:
#     """ Takes the subject activities and splits it into categories"""
#     control = subject['Control']
#     left = subject['Tapping/Left']
#     right = subject['Tapping/Right']
#     return control, left, right

def split_activities(subject: Epochs) -> tuple[Epochs, Epochs, Epochs]:
    """ Takes the subject activities and splits it into categories"""
    string_label = list(set(subject.annotations.description))
    
    activity_data = []
    for label in string_label:
        activity_data.append(subject[label].get_data())
        
    return activity_data

def get_labels_numeric(subject: Epochs) -> np.ndarray:
    """Gets numeric labels from activities for 1 subject"""
    return subject.events[:, -1]

# def get_minimum_bound(control: Epochs, left: Epochs, right: Epochs) -> int:
#     """Find the activity with min amount of epochs """
#     return np.min([x.get_data().shape[0] for x in [control, left, right]])

def get_minimum_bound(epoch_list:list) -> int:
    """Find the activity with min amount of epochs """
    n_epochs = [epoch_list[x].shape[0] for x in range(len(epoch_list))] 
    minimum_epochs = np.min(n_epochs)
    return minimum_epochs

def reshape_activity(epoch: list, min_bound: int) -> np.ndarray:
    """Limit epochs to min bound, reshape data to channels x merged_epochs """
    epoch_data = epoch[:min_bound,:,:]
    n_epoch, n_channels, n_epoch_size = epoch_data.shape
    epoch_data_reshaped = epoch_data.reshape(n_channels, n_epoch * n_epoch_size)
    
    return epoch_data_reshaped.T

def create_labels(labels: list[str], lenght: int) -> np.ndarray:
    """ Creates labels for data"""
    init_list = []
    for label in labels:
        init_list.append(np.full(lenght, label))
    return np.concatenate(init_list)

# def extract_X_y(subject: Epochs) -> tuple[np.ndarray, np.ndarray]:
#     """Combines all functions in one function and creates X and y ~ (features and labels)"""
    
#     control, left, right = split_activities(subject)
#     min_bound = get_minimum_bound(control, left, right)
    
#     control_reshaped = reshape_activity(control, min_bound)
#     left_reshaped = reshape_activity(left, min_bound)
#     right_reshaped = reshape_activity(right, min_bound)
    
#     # Define X input
#     X = np.concatenate([control_reshaped, left_reshaped, right_reshaped], axis=0)

#     # Create labels and define y
#     lenght = control_reshaped.shape[0]
#     labels = ['control', 'left','rigth']
#     y = create_labels(labels, lenght)
#     return X, y

def extract_X_y(subject: Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Combines all functions in one function and creates X and y ~ (features and labels)"""
    
    splitted_data = split_activities(subject)
    min_bound = get_minimum_bound(splitted_data)
    
    reshaped_data = []
    lenght = [] # Used later for labels
    for split in splitted_data:
        reshape = reshape_activity(split, min_bound)
        lenght.append(reshape.shape[0]) 
        reshaped_data.extend(reshape)

    # Define X input
    X = reshaped_data
    
    # Create labels and define y
    labels = list(set(subject.annotations.description))
    y = []
    for idx, label in enumerate(labels):
        y.extend(np.full(lenght[idx], label))

    return np.asarray(X), np.asarray(y)

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
    X, y = extract_X_y(subject)
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