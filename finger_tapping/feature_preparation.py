from preprocessing import simple_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

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
    
    
def plotting_pca(pc_x, pc_y):
    """Plotting, pca index starts from 0 e.g plot pca1 and pca2 then set pc_x=0, pc_y=1"""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x=pc_x, y=pc_y, hue='label', alpha=0.7, palette="Set1")
    plt.xlabel(f'Principal Component {pc_x + 1}')
    plt.ylabel(f'Principal Component {pc_y + 1}')
    plt.title('PCA on Finger Tapping Data')
    plt.legend(title="Class")
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    # Load data
    subject = simple_pipeline(subject="01")
    
    # Define X,y
    X, y = extract_X_y(subject)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    # Combine in dataframe
    pca_df = pd.DataFrame(X_pca)
    pca_df['label'] = y
    
    # plot
    plotting_pca(pc_x=0, pc_y=1)
    
