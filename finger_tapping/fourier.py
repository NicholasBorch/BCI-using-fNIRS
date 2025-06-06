from preprocessing import simple_pipeline
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d

epochs = simple_pipeline(subject='01')
events = epochs.events
map_ = {v: k for k, v in epochs.event_id.items()}

data = epochs.get_data()  # (n_epochs, n_channels, n_times)
n_epochs, n_channels, n_times = data.shape
sfreq = epochs.info['sfreq']


## Concatenate all epochs along time for ICA
data_concat = data.transpose(0, 2, 1).reshape(n_epochs * n_times, n_channels)

## Sliding average to smooth the data
window_size = 3

data_concat = uniform_filter1d(data_concat, size=window_size, axis=0)

## Centering data
data_concat = data_concat - np.mean(data_concat, axis=0)  # Centering data
data_concat = data_concat / np.std(data_concat, axis=0)  # Standardizing data

## ICA
ica = FastICA(n_components=n_channels, random_state=42, whiten='unit-variance')
S_ = ica.fit_transform(data_concat)  # (n_epochs*n_times, n_components)
ica_signal = S_[:, 0]  # Choose component

## Split by event onsets
sample_onsets = events[:, 0]
labels = events[:, 2]
segments = []
segment_labels = []
delay_secs = 5  # Delay before the cue in seconds
for i in range(len(sample_onsets)):
    start = sample_onsets[i]
    end = sample_onsets[i+1] if i < len(sample_onsets) - 1 else len(ica_signal)
    if len(ica_signal[start+delay_secs:end]) == 0:
        continue
    segments.append(ica_signal[start + delay_secs:end])
    segment_labels.append(labels[i])

## FFT per segment
fft_results = []
freqs = []
sfreq = epochs.info['sfreq']
for seg in segments:
    if len(seg) == 0:
        continue
    N = len(seg)
    yf = rfft(seg)
    xf = rfftfreq(N, 1/sfreq)
    fft_results.append(np.abs(yf) / N) # type: ignore
    freqs.append(xf)
    
unique_labels = np.unique(segment_labels)

# Plot mean FFT for each state

plt.figure(figsize=(10, 6))
for cond in np.unique(segment_labels):
    idx = [i for i, lab in enumerate(segment_labels) if lab == cond]
    if not idx:
        continue
    # Find minimum length for this group
    minlen = min([len(fft_results[i]) for i in idx])
    group_ffts = np.array([fft_results[i][:minlen] for i in idx])
    group_freqs = freqs[idx[0]][:minlen]  # Use the freq axis from one segment (they are all similar)
    mean_amp = group_ffts.mean(axis=0)
    plt.plot(group_freqs, mean_amp, label=f"Cond {map_[cond]}")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Mean ICA Segment FFT by Condition")
plt.legend()
plt.tight_layout()
plt.show()

## Plot random segments
plots_per_condition = 4  

unique_labels = np.unique(segment_labels)
fig, axes = plt.subplots(len(unique_labels), plots_per_condition, figsize=(4*plots_per_condition, 4*len(unique_labels)), sharex=True, sharey=True)

for row, cond in enumerate(unique_labels):
    idx = [i for i, lab in enumerate(segment_labels) if lab == cond]
    if not idx:
        continue
    # If fewer segments than needed, just plot all; else, sample
    sampled = np.random.choice(idx, size=min(plots_per_condition, len(idx)), replace=False)
    for col in range(plots_per_condition):
        ax = axes[row, col] if len(unique_labels) > 1 else axes[col]
        if col < len(sampled):
            i = sampled[col]
            fs_ = freqs[i]
            as_ = fft_results[i]
            ax.plot(fs_, as_)
            ax.set_title(f"{map_[cond]}, Seg {i}")
        else:
            ax.axis('off')
        if row == len(unique_labels) - 1:
            ax.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax.set_ylabel("Amplitude")
fig.suptitle("Random FFT Segments per Condition")
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.show()

for fs_, as_ in zip(freqs, fft_results):
    pass

# def fourier_transform(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
#     """Apply Fourier Transform to the data"""
#     _, df = run_ica(X, y, n_components=5, random_state=42)
    
#     # Apply Fourier Transform to each column in the DataFrame
#     transformed_data = np.array([fftfreq(df[col]) for col in df.columns[:-1]]).T  # Exclude the label column
#     transformed_df = pd.DataFrame(transformed_data, columns=[f"FT_{col}" for col in df.columns[:-1]])
#     transformed_df['label'] = df['label'].values  # Add the label column back
#     return transformed_df


# def cluster_fourier_components(transformed_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
#     """Cluster the Fourier transformed components using GMM or KMeans"""
    
#     # Fit GMM
#     gmm = GaussianMixture(n_components=n_clusters, random_state=42)
#     transformed_df['cluster'] = gmm.fit_predict(transformed_df.drop(columns='label'))
    
#     return transformed_df

# if __name__ == '__main__':
#     # Load data
#     subject = simple_pipeline(subject="01")
    
#     # Extract features and labels
#     X, y = extract_X_y(subject)
    
#     # Apply Fourier Transform
#     transformed_df = fourier_transform(X, y)
#     print(transformed_df.head())
    
#     # Cluster the Fourier transformed components
#     clustered_df = cluster_fourier_components(transformed_df, n_clusters=3)
#     print(clustered_df.head())