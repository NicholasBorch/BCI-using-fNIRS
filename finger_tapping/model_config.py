from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from mne import Epochs
import numpy as np
from typing import Literal

from preprocessing import simple_pipeline

def prepare_data_classification(epoch: Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Loads data from epoch, reshapes, and scales"""
    X = epoch.get_data()
    y = epoch.events[:, -1]
    
    # Flatten X channels
    X_flat = X.reshape(X.shape[0], -1)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    return X_scaled, y

def train_lda(X_train: np.ndarray, y_train: np.ndarray) -> LinearDiscriminantAnalysis:
    """Train LDA model"""
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda 

def train_svm(X_train: np.ndarray, y_train: np.ndarray,
            kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "linear"
            ):
    """Train SVM model"""
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    return svm


def train_ann(X_train, y_train) -> MLPClassifier:
    """Train ann"""
    mlp = MLPClassifier(hidden_layer_sizes=(50, 20), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp

def train_baseline(X, y) -> DummyClassifier:
    """Train a baseline model"""
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X, y)
    return baseline

if __name__ == "__main__":
    epoch = simple_pipeline(subject="01")
    X, y= prepare_data_classification(epoch)
    lda = train_lda(X, y)
    svm = train_svm(X, y)
    ann = train_ann(X, y)
    
    
