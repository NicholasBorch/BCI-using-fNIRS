import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import mne

from preprocessing import simple_pipeline 
from preprocessing import raw_intensity_pipeline


# --- Step 1: Load preprocessed data ---
subject = "01"
epochs = simple_pipeline(subject=subject, save=False)
raw = raw_intensity_pipeline(subject=subject)

# Print all annotations with durations
print("Cue annotation durations:")
for desc, onset, duration in zip(raw.annotations.description, raw.annotations.onset, raw.annotations.duration):
    print(f"{desc}: starts at {onset:.1f}s, duration = {duration:.1f}s")

# Count total and per cue type
cue_descriptions = raw.annotations.description
unique_cues, counts = np.unique(cue_descriptions, return_counts=True)
print(f"\nTotal number of cues: {len(cue_descriptions)}")
for cue, count in zip(unique_cues, counts):
    print(f"  {cue}: {count}")

# --- Step 2: Extract data and labels ---
data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
n_epochs, n_channels, n_times = data.shape
print(f"Data shape: {data.shape} (epochs, channels, timepoints)")

# Reshape for ICA: concatenate all epochs in sequence along time axis
data_2d = data.transpose(1, 0, 2).reshape(n_channels, -1).T  # (n_epochs*n_times, n_channels)

# --- Step 3: ICA on scaled data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_2d)

ica = FastICA(n_components=5, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)

# --- Step 4: GMM Clustering into 2 clusters ---
gmm = GaussianMixture(n_components=2, random_state=42)
clusters = gmm.fit_predict(X_ica[:, :2])  # Use only IC1 and IC2 for clustering

# --- Step 5: Time vector for concatenated epochs ---
sampling_rate = epochs.info['sfreq']  # e.g. 7.81 Hz
epoch_duration_sec = n_times / sampling_rate  # should be 20s (5s before + 15s after cue)
total_time = n_epochs * epoch_duration_sec
time = np.linspace(0, total_time, n_epochs * n_times)  # continuous time axis for all epochs concatenated

# --- Step 6: Plot IC1 and IC2 over concatenated epochs ---
fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

axes[0].scatter(time, X_ica[:, 0], c=clusters, cmap='bwr', s=1)
axes[0].set_ylabel('IC1')
axes[0].set_title('IC1 with GMM Clusters')

axes[1].scatter(time, X_ica[:, 1], c=clusters, cmap='bwr', s=1)
axes[1].set_ylabel('IC2')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('IC2 with GMM Clusters')

# --- Step 7: Add vertical lines for cue onsets in each epoch, colored and annotated ---
cue_onset_sec = 5  # cue onset 5 seconds into each epoch (since epoch starts 5s before cue)

# Assign a color to each unique cue
import matplotlib.cm as cm
cmap = cm.get_cmap('tab10', len(unique_cues))
cue_color_dict = {cue: cmap(i) for i, cue in enumerate(unique_cues)}

for i, cue_label in enumerate(cue_descriptions):
    cue_time = i * epoch_duration_sec + cue_onset_sec
    color = cue_color_dict[cue_label]
    for ax in axes:
        ax.axvline(x=cue_time, color=color, linestyle='--', alpha=0.8)
        ax.text(cue_time, ax.get_ylim()[1]*0.9, cue_label, rotation=90,
                verticalalignment='top', fontsize=8, color=color)

plt.tight_layout()
plt.xlim(0, total_time)
plt.show()
