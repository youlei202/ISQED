# experiments/exp3_shapley.py

import sys
import os
import numpy as np
import pandas as pd
from itertools import combinations
from math import factorial

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.synthetic import LinearStructuralModel
from isqed.geometry import DISCOSolver

# ... (Utility calculation functions remain the same) ...
def get_utility(subset_indices, models, X_val, y_val):
    if len(subset_indices) == 0: return 0.0
    subset_preds = []
    for idx in subset_indices:
        subset_preds.append(models[idx]._forward(X_val))
    A = np.array(subset_preds).T
    try:
        beta = np.linalg.inv(A.T @ A + 1e-6 * np.eye(len(subset_indices))) @ A.T @ y_val
        y_pred = A @ beta
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        return max(0, 1 - (ss_res / ss_tot))
    except:
        return 0.0

def calculate_shapley_values(models, X_val, y_val):
    n = len(models)
    indices = list(range(n))
    shapley_values = {i: 0.0 for i in indices}
    for i in indices:
        others = [x for x in indices if x != i]
        for k in range(len(others) + 1):
            for subset in combinations(others, k):
                S = list(subset)
                S_with_i = S + [i]
                val_S = get_utility(S, models, X_val, y_val)
                val_S_i = get_utility(S_with_i, models, X_val, y_val)
                weight = (factorial(len(S)) * factorial(n - len(S) - 1)) / factorial(n)
                shapley_values[i] += weight * (val_S_i - val_S)
    return shapley_values

def run_experiment():
    print("--- Running Exp 3: The 'Misleading Attribution' Scenario ---")
    
    # Setup Ground Truth
    n_samples = 500
    dim = 5
    X = np.random.randn(n_samples, dim)
    
    # >>> CRITICAL CHANGE <<<
    # True relation: Y is DOMINATED by features 0 and 1 (95% variance).
    # Feature 4 has a very weak effect (5% variance).
    true_beta = np.array([10.0, 10.0, 0.0, 0.0, 1.0]) 
    
    y = X @ true_beta + np.random.randn(n_samples) * 0.1
    
    # --- Step A: Construct the Ecosystem ---
    
    # Model A: The "Big Model" (Captures the dominant features 0, 1)
    beta_0 = np.array([9.9, 9.8, 0.1, 0.0, 0.0]) 
    model_1 = LinearStructuralModel(dim, beta=beta_0)
    
    # Model B: The "Redundant Clone" (Almost identical to A)
    beta_1 = beta_0 + np.random.randn(dim) * 0.05 
    model_2 = LinearStructuralModel(dim, beta=beta_1)
    
    # Model C: The "Niche Specialist" (Only captures the weak feature 4)
    # Important: It behaves VERY differently from A/B, but contributes little to R^2.
    beta_2 = np.array([0.0, 0.0, 0.0, 0.0, 5.0]) # Note: High coefficient magnitude, but low task weight? 
    # Actually, to make PIER high, C just needs to be orthogonal to A/B.
    # To make Shapley low, C needs to contribute little to y_true.
    # Let's set C to focus on feature 4 strongly. 
    # Since y_true weight for feat 4 is only 1.0 (vs 10.0), C's utility contribution is capped.
    model_3 = LinearStructuralModel(dim, beta=beta_2)
    
    models = [model_1, model_2, model_3]
    model_names = ["Model 1\n(Dominant)", "Model 2\n(Redundant)", "Model 3\n(Niche Unique)"]
    
    # --- Step B: Calculate Shapley Values ---
    print("Calculating Shapley (Utility Attribution)...")
    X_val = np.random.randn(200, dim)
    y_val = X_val @ true_beta
    shap_dict = calculate_shapley_values(models, X_val, y_val)
    
    # --- Step C: Calculate PIER (Behavioral Auditing) ---
    print("Calculating PIER (Behavioral Uniqueness)...")
    pier_values = {}
    
    # In-Silico Probing (Uniqueness is defined on BEHAVIOR, not utility)
    # We probe with standard gaussian inputs to measure response space distance
    X_probe = np.random.randn(500, dim) 
    Y_responses = np.array([m._forward(X_probe) for m in models]).T 
    
    for i in range(len(models)):
        target_vec = Y_responses[:, i]
        peer_idx = [x for x in range(len(models)) if x != i]
        peer_matrix = Y_responses[:, peer_idx]
        dist, _ = DISCOSolver.solve_weights_and_distance(target_vec, peer_matrix)
        
        # Normalize: nPIER
        target_norm = np.linalg.norm(target_vec)
        pier_score = dist / target_norm if target_norm > 0 else 0
        pier_values[i] = pier_score

    # --- Step D: Save & Print ---
    results = []
    
    # Normalize Shapley for plot comparison
    total_shap = sum(shap_dict.values())
    
    for i in range(len(models)):
        results.append({
            "Model": model_names[i],
            "Metric": "Shapley Value (Attribution)",
            "Score": shap_dict[i] / total_shap # Normalized share of utility
        })
        results.append({
            "Model": model_names[i],
            "Metric": "PIER (Auditing)",
            "Score": pier_values[i]
        })
        
    df = pd.DataFrame(results)
    df.to_csv("results/tables/exp3_shapley.csv", index=False)
    
    print("\n--- Results Summary ---")
    print(df)

if __name__ == "__main__":
    run_experiment()