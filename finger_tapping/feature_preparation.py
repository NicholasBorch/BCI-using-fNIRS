import numpy as np

def split_activities(subject):
    """ Takes the subject activities and splits it into categories"""
    control = subject['Control']
    left = subject['Tapping/Left']
    right = subject['Tapping/Right']
    return control, left, right

def get_labels_numeric(subject):
    """Gets numeric labels from activities for 1 subject"""
    return subject.events[:, -1]

def get_minimum_bound(control, left, right):
    """Find the activity with min amount of epochs """
    return np.min([x.get_data().shape[0] for x in [control, left, right]])

def reshape_activity(epoch, min_bound):
    """Limit epochs to min bound, reshape data to channels x merged_epochs """
    epoch_data = epoch.get_data()[:min_bound,:,:]
    n_epoch, n_channels, n_epoch_size = epoch_data.shape
    epoch_data_reshaped = epoch_data.reshape(n_channels, n_epoch * n_epoch_size)
    
    return epoch_data_reshaped.T

def create_labels(labels, lenght):
    """ Creates labels for data"""
    init_list = []
    for label in labels:
        init_list.append(np.full(lenght, label))
    return np.concatenate(init_list)

def extract_X_y(subject):
    """Combines all functions in one function and creates X and y ~ (features and labels)"""
    
    control, left, right = split_activities(subject)
    min_bound = get_minimum_bound(control, left, right)
    
    control_reshaped = reshape_activity(control, min_bound)
    left_reshaped = reshape_activity(left, min_bound)
    right_reshaped = reshape_activity(right, min_bound)
    
    # Define X input
    X = np.concatenate([control_reshaped, left_reshaped, right_reshaped], axis=0)

    # Create labels and define y
    lenght = control_reshaped.shape[0]
    labels = ['control', 'left','rigth']
    y = create_labels(labels, lenght)
    return X, y
    
if __name__ == '__main__':
    from preprocessing import simple_pipeline
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