import numpy as np
import json
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import pandas as pd
from tabulate import tabulate

# ==========================================
# CONFIGURATION & SEEDING
# ==========================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_evaluation_data():
    """Loads required mapping, labels, and baseline scores."""
    print("Loading node counts and ground truth labels...")
    
    try:
        with open('node_counts.json', 'r') as f:
            counts = json.load(f)
    except FileNotFoundError:
        with open('data/node_counts.json', 'r') as f:
            counts = json.load(f)
            
    N_u = counts['users']
    
    # Structure: [node_index, label]
    try:
        labels_data = np.load('labels.npy')
    except FileNotFoundError:
        labels_data = np.load('data/labels.npy')
    
    print("Loading baseline scores...")
    try:
        sage_scores = np.load('sage_anomalies.npy') 
        dominant_scores = np.load('dominant_anomalies.npy')
    except FileNotFoundError as e:
        print(f"Error: Could not find baseline score files.\nDetails: {e}")
        exit(1)
        
    return N_u, labels_data, sage_scores, dominant_scores

def calculate_metrics_at_k(y_true, y_scores, k=100):
    """Calculates Precision@K and Recall@K."""
    if len(y_true) == 0: return 0.0, 0.0
    
    # Sort indices by score in descending order
    desc_score_indices = np.argsort(y_scores)[::-1]
    top_k_indices = desc_score_indices[:k]
    
    # True labels of the top K predictions
    top_k_true = y_true[top_k_indices]
    
    # Precision@K = (TP in Top K) / K
    precision_at_k = np.sum(top_k_true) / k
    
    # Recall@K = (TP in Top K) / (Total Positive in dataset)
    total_positives = np.sum(y_true)
    if total_positives == 0:
        recall_at_k = 0.0
    else:
        recall_at_k = np.sum(top_k_true) / total_positives
        
    return precision_at_k, recall_at_k

def evaluate_entity_group(name, indices, labels_map, all_scores, threshold_pct=82):
    """Evaluates a specific entity group (e.g., Users or Products)."""
    
    # 1. Filter indices that actually have ground truth labels
    labeled_indices = [idx for idx in indices if idx in labels_map]
    
    if not labeled_indices:
        return [name, 0, 0, 0, 0, 0, 0, 0, 0]

    y_true = np.array([labels_map[idx] for idx in labeled_indices])
    y_scores = all_scores[labeled_indices]
    
    # 2. Calculate Probabilistic Metrics (AUC)
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
    except ValueError:
        # Handles cases with only one class present in selection
        auc_roc, auc_pr = 0.0, 0.0

    # 3. Calculate Threshold-based Metrics (Precision, Recall, F1)
    # Using the same 82nd percentile heuristic as GNN-EADD
    threshold = np.percentile(y_scores, threshold_pct)
    y_pred = (y_scores >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # 4. Calculate Ranking Metrics (@K)
    prec_at_100, rec_at_100 = calculate_metrics_at_k(y_true, y_scores, k=100)
    
    return [
        name,
        len(labeled_indices),
        auc_roc,
        auc_pr,
        precision,
        recall,
        f1,
        prec_at_100,
        rec_at_100
    ]

def generate_performance_table(baseline_name, N_u, labels_data, scores):
    """Generates and prints the standard performance table."""
    
    # Create quick lookup map for labels: {node_index: label}
    labels_map = {int(row[0]): int(row[1]) for row in labels_data}
    labeled_node_indices = list(labels_map.keys())
    
    # Separate User and Product indices from the labeled set
    user_indices = [idx for idx in labeled_node_indices if idx < N_u]
    product_indices = [idx for idx in labeled_node_indices if idx >= N_u]
    
    results = []
    
    # Evaluate Users
    results.append(evaluate_entity_group("User", user_indices, labels_map, scores))
    
    # Evaluate Products
    results.append(evaluate_entity_group("Product", product_indices, labels_map, scores))
    
    # Evaluate Global (All labeled nodes)
    results.append(evaluate_entity_group("Global", labeled_node_indices, labels_map, scores))
    
    # Formatting for Output
    headers = [
        f"Entity ({baseline_name})", "Count", "AUC-ROC", "AUC-PR", 
        "Prec", "Rec", "F1", "Prec@100", "Rec@100"
    ]
    
    table = tabulate(results, headers=headers, tablefmt="grid", floatfmt=".4f")
    print(f"\n{baseline_name} Baseline Performance:")
    print(table)
    return results

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("BASELINE PERFORMANCE EVALUATION (Standardized)")
    print("="*60)
    
    # 1. Load Data
    N_u, labels_data, sage_scores, dominant_scores = load_evaluation_data()
    
    # 2. Evaluate GraphSAGE
    generate_performance_table("GraphSAGE", N_u, labels_data, sage_scores)
    
    # 3. Evaluate DOMINANT
    generate_performance_table("DOMINANT", N_u, labels_data, dominant_scores)
    
    print("\nEvaluation complete. Compare these tables with your GNN-EADD results.")
