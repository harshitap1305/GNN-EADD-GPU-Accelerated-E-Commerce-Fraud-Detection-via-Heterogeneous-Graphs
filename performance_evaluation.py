import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score

def load_json_safe(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    elif os.path.exists(f'data/{filename}'):
        with open(f'data/{filename}', 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Could not find {filename}")

def metrics_at_k(y_true, y_scores, k=100):
    """
    Computes Top-K ranking metrics for anomaly retrieval.
    
    Args:
        y_true (ndarray): Ground truth binary labels.
        y_scores (ndarray): Predicted continuous anomaly scores.
        k (int): Truncation threshold for top-ranked anomalies. 
                 Default of 100 represents a standard budget constraint for manual review.
    
    Returns:
        tuple: (Precision@K, Recall@K)
    """
    if len(y_true) == 0: 
        return 0.0, 0.0
    
    # Bound K to the maximum available evaluated instances
    k = min(k, len(y_true))
    
    # Isolate the indices of the top-K highest predicted anomaly scores
    order = np.argsort(y_scores)[::-1]
    top_k_labels = y_true[order[:k]]
    
    # Precision@K: Proportion of retrieved top-K instances that are true anomalies
    prec_at_k = top_k_labels.sum() / k
    
    # Recall@K: Proportion of total true anomalies successfully retrieved in the top-K
    rec_at_k = top_k_labels.sum() / y_true.sum() if y_true.sum() > 0 else 0.0
    
    return float(prec_at_k), float(rec_at_k)

def evaluate_anomalies():
    print("\nInitializing Evaluation Protocol...")
    counts = load_json_safe('node_counts.json')
    N_u = counts['users']
    N_p = counts['products']
    
    print("Ingesting Ground Truth and Stage 2 Predictions...")
    try:
        labels_data = np.load('labels.npy') # Shape: [L, 2] -> (Node_ID, Binary_Label)
    except FileNotFoundError:
        labels_data = np.load('data/labels.npy')
        
    indices = labels_data[:, 0]
    y_true = labels_data[:, 1]
    
    # Retrieve the bounded anomaly probabilities generated via the GAT forward pass
    all_scores = np.load('anomaly_scores_stage2.npy')
    
    # Isolate scores corresponding strictly to the labeled evaluation subset
    y_scores = all_scores[indices]
    y_pred = np.zeros_like(y_scores, dtype=int)
    
    # Generate boolean masks to segregate heterogeneous node types
    user_mask = indices < N_u
    prod_mask = (indices >= N_u) & (indices < N_u + N_p)
    
    print("Applying Type-Specific Decision Boundaries...")
    
    # --- TYPE-SPECIFIC THRESHOLDING: USERS ---
    if user_mask.sum() > 0:
        y_scores_u = y_scores[user_mask]
        # The 82nd percentile is empirically selected based on historical anomaly prevalence in the user domain to optimize the F1-Score tradeoff.
        user_threshold = np.percentile(y_scores_u, 82)
        y_pred[user_mask] = (y_scores_u >= user_threshold).astype(int)
        print(f" -> User Threshold applied   : {user_threshold:.6f} (82nd Percentile)")
        
    # --- TYPE-SPECIFIC THRESHOLDING: PRODUCTS ---
    if prod_mask.sum() > 0:
        y_scores_p = y_scores[prod_mask]
        # Standard 0.5 decision boundary utilized for products, reflecting the neutral midpoint of the sigmoid activation function.
        prod_threshold = 0.5 
        y_pred[prod_mask] = (y_scores_p >= prod_threshold).astype(int)
        print(f" -> Product Threshold applied: {prod_threshold:.6f} (Standard Midpoint)")

    # ---------------------------------------------------------
    # 1. OVERALL METRICS
    # ---------------------------------------------------------
    prec_100, rec_100 = metrics_at_k(y_true, y_scores, k=100)
    
    print("\n==================================================")
    print(" OVERALL DETECTION PERFORMANCE")
    print("==================================================")
    print(f" Total Evaluated Nodes : {len(indices)}")
    print(f" Actual Anomalies (1)  : {int(y_true.sum())}")
    print(f" Normal Samples (0)    : {int(len(y_true) - y_true.sum())}")
    print("--------------------------------------------------")
    print(f" AUC-ROC               : {roc_auc_score(y_true, y_scores):.4f}")
    print(f" AUC-PR                : {average_precision_score(y_true, y_scores):.4f}")
    print(f" Precision@100         : {prec_100:.4f}")
    print(f" Recall@100            : {rec_100:.4f}")
    print(f" Global Precision      : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f" Global Recall         : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f" Global F1-Score       : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n Confusion Matrix:")
    print(f"   True Negatives (0->0) : {cm[0][0]}")
    print(f"   False Positives(0->1) : {cm[0][1]}")
    print(f"   False Negatives(1->0) : {cm[1][0]}")
    print(f"   True Positives (1->1) : {cm[1][1]}")
    print("==================================================")
    
    # ---------------------------------------------------------
    # 2. USER-SPECIFIC METRICS
    # ---------------------------------------------------------
    if user_mask.sum() > 0:
        y_true_u, y_pred_u = y_true[user_mask], y_pred[user_mask]
        prec_100_u, rec_100_u = metrics_at_k(y_true_u, y_scores_u, k=100)
        
        print("\n==================================================")
        print(" USER ANOMALY PERFORMANCE (Reviewers)")
        print("==================================================")
        print(f" Evaluated Users       : {user_mask.sum()} (Anomalies: {int(y_true_u.sum())})")
        print("--------------------------------------------------")
        if len(np.unique(y_true_u)) > 1:
            print(f" AUC-ROC               : {roc_auc_score(y_true_u, y_scores_u):.4f}")
            print(f" AUC-PR                : {average_precision_score(y_true_u, y_scores_u):.4f}")
        print(f" Precision@100         : {prec_100_u:.4f}")
        print(f" Recall@100            : {rec_100_u:.4f}")
        print(f" Subgroup Precision    : {precision_score(y_true_u, y_pred_u, zero_division=0):.4f}")
        print(f" Subgroup Recall       : {recall_score(y_true_u, y_pred_u, zero_division=0):.4f}")
        print(f" Subgroup F1-Score     : {f1_score(y_true_u, y_pred_u, zero_division=0):.4f}")
        print("==================================================")

    # ---------------------------------------------------------
    # 3. PRODUCT-SPECIFIC METRICS
    # ---------------------------------------------------------
    if prod_mask.sum() > 0:
        y_true_p, y_pred_p = y_true[prod_mask], y_pred[prod_mask]
        prec_100_p, rec_100_p = metrics_at_k(y_true_p, y_scores_p, k=100)
        
        print("\n==================================================")
        print(" PRODUCT ANOMALY PERFORMANCE (Items)")
        print("==================================================")
        print(f" Evaluated Products    : {prod_mask.sum()} (Anomalies: {int(y_true_p.sum())})")
        print("--------------------------------------------------")
        if len(np.unique(y_true_p)) > 1:
            print(f" AUC-ROC               : {roc_auc_score(y_true_p, y_scores_p):.4f}")
            print(f" AUC-PR                : {average_precision_score(y_true_p, y_scores_p):.4f}")
        print(f" Precision@100         : {prec_100_p:.4f}")
        print(f" Recall@100            : {rec_100_p:.4f}")
        print(f" Subgroup Precision    : {precision_score(y_true_p, y_pred_p, zero_division=0):.4f}")
        print(f" Subgroup Recall       : {recall_score(y_true_p, y_pred_p, zero_division=0):.4f}")
        print(f" Subgroup F1-Score     : {f1_score(y_true_p, y_pred_p, zero_division=0):.4f}")
        print("==================================================\n")

if __name__ == "__main__":
    evaluate_anomalies()
