import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from feature_preparation import extract_X_y
from preprocessing import simple_pipeline

# --- Load and preprocess data ---
subject = simple_pipeline(subject="01")
X, y = extract_X_y(subject)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Apply ICA ---
ica = FastICA(n_components=5, max_iter=1000, tol=0.0001, random_state=42)
X_ica = ica.fit_transform(X_scaled)

# --- Create DataFrame with IC1 and IC2 ---
ica_df = pd.DataFrame(X_ica[:, :2], columns=['IC1', 'IC2'])
ica_df['time'] = np.arange(len(X_ica)) / 7.81  # 7.81 Hz sampling rate

# --- Fit binary GMM ---
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(ica_df[['IC1', 'IC2']])
ica_df['cluster'] = gmm_labels

# --- Plot IC1 over time ---
plt.figure(figsize=(12, 4))
sns.scatterplot(data=ica_df, x='time', y='IC1', hue='cluster', palette='Set2', s=10)
plt.axvline(x=ica_df.time[len(ica_df)//3], color='red', linestyle='--', label='Cue 1')
plt.axvline(x=2*ica_df.time[len(ica_df)//3], color='red', linestyle='--', label='Cue 2')
plt.title("IC1 over Time with GMM Clusters")
plt.xlabel("Time (s)")
plt.ylabel("IC1 Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot IC2 over time ---
plt.figure(figsize=(12, 4))
sns.scatterplot(data=ica_df, x='time', y='IC2', hue='cluster', palette='Set2', s=10)
plt.axvline(x=ica_df.time[len(ica_df)//3], color='red', linestyle='--', label='Cue 1')
plt.axvline(x=2*ica_df.time[len(ica_df)//3], color='red', linestyle='--', label='Cue 2')
plt.title("IC2 over Time with GMM Clusters")
plt.xlabel("Time (s)")
plt.ylabel("IC2 Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
