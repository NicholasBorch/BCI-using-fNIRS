import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from model_config import prepare_data_classification, train_lda, train_svm, train_ann, train_baseline
from preprocessing import simple_pipeline

def evaluate_model(model, X, y, cv=5):
    """Evaluates a model using k-fold cross-validation"""
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    return scores

if __name__ == "__main__":
    epoch = simple_pipeline(subject=1)
    X, y = prepare_data_classification(epoch)
    
    # Train models
    lda = train_lda(X, y)
    svm = train_svm(X, y)
    ann = train_ann(X, y)
    baseline = train_baseline(X, y)

    # Evaluate models
    lda_scores = evaluate_model(lda, X, y)
    svm_scores = evaluate_model(svm, X, y)
    ann_scores = evaluate_model(ann, X, y)
    baseline_scores = evaluate_model(baseline, X, y)
    
    # Print results
    print(f"LDA Accuracy: {np.mean(lda_scores):.4f}")
    print(f"SVM Accuracy: {np.mean(svm_scores):.4f}")
    print(f"ANN Accuracy: {np.mean(ann_scores):.4f}")
    print(f"Baseline Accuracy: {np.mean(baseline_scores):.4f}")

