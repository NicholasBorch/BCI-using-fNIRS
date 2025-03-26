from preprocessing import simple_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load data
subject = simple_pipeline(subject="01")
subject_data = subject.get_data()
y = subject.events[:, -1]

# Divide each activity
control = subject['Control']
left = subject['Tapping/Left']
right = subject['Tapping/Right']

# Find the activity with min amount of epochs
min_bound = np.min([x.get_data().shape[0] for x in [control, left, right]])

# Reshape left
def reshape_activity(epoch, min_bound):
    """Limit epochs to min bound, reshape data to channels x merged_epochs """
    epoch_data = epoch.get_data()[:min_bound,:,:]
    n_epoch, n_channels, n_epoch_size = epoch_data.shape
    epoch_data_reshaped = epoch_data.reshape(n_channels, n_epoch * n_epoch_size)
    
    return epoch_data_reshaped.T

control_reshaped = reshape_activity(control, min_bound)
left_reshaped = reshape_activity(left, min_bound)
right_reshaped = reshape_activity(right, min_bound)

# Define X input
X = np.concatenate([control_reshaped, left_reshaped, right_reshaped], axis=0)

# Create labels
lenght =control_reshaped.shape[0]
y = np.concatenate([np.full(lenght,1),np.full(lenght,2),np.full(lenght,3)])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Combine in dataframe
pca_df = pd.DataFrame(X_pca)
pca_df['label'] = y

# Renaming label 1,2,3 name to right, control and, left
def set_label_name(x):
    if x == 1: return 'control'
    elif x == 2: return 'left'
    elif x == 3: return 'right'
    else: return np.nan
    
pca_df['label_name'] = pca_df['label'].apply(lambda x: set_label_name(x))

# Define which pc to plot 
pc_x = 1
pc_y = 2

# Plot PCA results
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=pca_df, x=pc_x, y=pc_y, hue='label_name', alpha=0.7, palette="Set1")
# plt.xlabel(f'Principal Component {pc_x + 1}')
# plt.ylabel(f'Principal Component {pc_y + 1}')
# plt.title('PCA on Finger Tapping Data')
# plt.legend(title="Class")
# plt.grid()
# plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(data=pca_df, x=0, hue='label_name', kde=True, palette="Set1", element="step")
plt.xlabel('Principal Component 1')
plt.title('Distribution of PC1 for Finger Tapping Data')
plt.legend(title="Class")
plt.grid(True)
plt.show()