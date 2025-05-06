import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from preprocessing import simple_pipeline
from feature_preparation import extract_X_y

# Load data
subject = simple_pipeline(subject="01")

# Define X,y
X, y = extract_X_y(subject)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply ICA
ica = FastICA(n_components=5, max_iter=1000, tol=0.0001, random_state=42)
X_ica = ica.fit_transform(X_scaled)

# Prepare DataFrame
ica_df = pd.DataFrame(X_ica)
ica_df['label'] = y


X_ic2 = ica_df[[0, 1]].values

kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
labels_km = kmeans.fit_predict(X_ic2)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_ic2[:, 0], y=X_ic2[:, 1], hue=labels_km, palette="Set2", s=20)
plt.title("K-Means (k = 2) on IC space")
plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.grid(True)
plt.show()

gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
labels_gmm = gmm.fit_predict(X_ic2)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_ic2[:, 0], y=X_ic2[:, 1], hue=labels_gmm, palette="Set2", s=20)
plt.title("Gaussian Mixture (2 components) on IC space")
plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.grid(True)
plt.show()
