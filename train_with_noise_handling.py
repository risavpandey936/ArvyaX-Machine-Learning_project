import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# Important integration for label noise - helps the model generalize by avoiding fitting to wrong annotations.
# 'cleanlab' is an industry-standard framework for dealing with noisy labels.
try:
    from cleanlab.filter import find_label_issues
except ImportError:
    print("Warning: please install cleanlab (`pip install cleanlab`)")

def handle_label_noise(X, labels):
    """
    Finds and filters out label noise using Confident Learning frameworks (cleanlab). 
    Cleanlab identifies samples that are likely mislabeled by comparing 
    model-predicted probabilities against the (noisy) ground truth labels.
    """
    print(f"Original dataset shape: {X.shape}")
    
    # We need out-of-sample probabilities via cross validation to objectively detect noisy labels
    clf = LogisticRegression(max_iter=1000)
    pred_probs = cross_val_predict(clf, X, labels, cv=5, method="predict_proba")
    
    # Detect label issues (noisy labels in the training set)
    label_issues_idx = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    
    # Keep only the samples that DO NOT have label issues
    clean_idx = np.setdiff1d(np.arange(len(labels)), label_issues_idx)
    
    # Optional filtering
    X_clean = X[clean_idx]
    
    # Convert series to numpy if necessary, then index
    if isinstance(labels, pd.Series):
        labels = labels.values
        
    y_clean = labels[clean_idx]
    
    print(f"Detected and removed {len(label_issues_idx)} incorrectly labelled points from the training set.")
    print(f"Cleaned dataset shape: {X_clean.shape}")
    
    return X_clean, y_clean

if __name__ == "__main__":
    # Example usage:
    # 
    # df = pd.read_csv("dataset.csv")
    # X = get_tfidf_and_metadata_features(df)
    # y = df['emotional_state'].values
    #
    # X_clean, y_clean = handle_label_noise(X, y)
    # 
    # clf = LogisticRegression(max_iter=1000)
    # clf.fit(X_clean, y_clean)
    # pickle.dump(clf, open("model.pkl", "wb"))
    pass
