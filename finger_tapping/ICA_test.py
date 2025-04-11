from preprocessing import simple_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
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


# Reshape function
def reshape_activity(epoch, min_bound):
    """Limit epochs to min bound, reshape data to channels x merged_epochs"""
    epoch_data = epoch.get_data()[:min_bound, :, :]
    n_epoch, n_channels, n_epoch_size = epoch_data.shape
    epoch_data_reshaped = epoch_data.reshape(n_channels, n_epoch * n_epoch_size)
    return epoch_data_reshaped.T

# Reshape all activities
control_reshaped = reshape_activity(control, min_bound)
left_reshaped = reshape_activity(left, min_bound)
right_reshaped = reshape_activity(right, min_bound)

# Define input matrix
X = np.concatenate([control_reshaped, left_reshaped, right_reshaped], axis=0)

# Create labels
length = control_reshaped.shape[0]
y = np.concatenate([np.full(length, 1), np.full(length, 2), np.full(length, 3)])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply ICA
ica = FastICA(n_components=5, max_iter=1000, tol=0.0001, random_state=42)
X_ica = ica.fit_transform(X_scaled)

# Prepare DataFrame
ica_df = pd.DataFrame(X_ica)
ica_df['label'] = y

# Label mapping
def set_label_name(x):
    if x == 1: return 'control'
    elif x == 2: return 'left'
    elif x == 3: return 'right'
    else: return np.nan

ica_df['label_name'] = ica_df['label'].apply(set_label_name)

# Select components to plot
ic_x = 0  # First component
ic_y = 1  # Second component

# Plot ICA results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=ica_df, x=ic_x, y=ic_y, hue='label_name', alpha=0.7, palette="Set1")
plt.xlabel(f'Independent Component {ic_x + 1}')
plt.ylabel(f'Independent Component {ic_y + 1}')
plt.title('ICA on Finger Tapping Data')
plt.legend(title="Class")
plt.grid()
plt.tight_layout()
plt.show()

# Plot ICA component weights
print(subject.info['chs'][0])
plt.figure(figsize=(10, 4))
plt.bar(range(ica.mixing_.shape[0]), ica.mixing_[:, 0])
plt.title('Sensor Contributions to IC1 (Motor Laterality?)')
plt.xlabel('Channels')
plt.ylabel('Weight')
plt.show()
