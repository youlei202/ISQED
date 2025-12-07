# experiments/exp1_scaling_law.py

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# Ensure the 'isqed' package can be imported from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.synthetic import LinearStructuralModel
from isqed.geometry import DISCOSolver

def generate_ecosystem(dim, n_peers, margin_gamma, is_unique=True):
    """
    Generates a synthetic ecosystem ground truth for simulation.
    """
    # 1. Generate random peers
    peers = [LinearStructuralModel(dim) for _ in range(n_peers)]
    peer_betas_matrix = np.array([p.beta for p in peers]).T 
    
    # 2. Generate Target
    if not is_unique:
        # H0: Target inside hull
        weights = np.random.dirichlet(np.ones(n_peers))
        beta_t = peer_betas_matrix @ weights
    else:
        # H1: Target outside hull
        center = np.mean(peer_betas_matrix, axis=1)
        perturbation = np.random.randn(dim)
        perturbation /= np.linalg.norm(perturbation)
        beta_t = center + perturbation * (margin_gamma + 1.0) 
        
    target = LinearStructuralModel(dim, beta=beta_t)
    return target, peers, peer_betas_matrix, beta_t

def run_experiment(dim=20, n_peers=10, noise_std=1.0, margin=0.5, trials=100):
    """
    Executes the Scaling Law experiment with ENERGY NORMALIZATION.
    """
    print(f"--- Running Exp 1: Scaling Law (Fixed SNR) ---")
    print(f"Dim: {dim}, Noise: {noise_std}, Margin: {margin}, Trials: {trials}")
    
    query_steps = range(20, 151, 10) # Start from dim (20)
    results = []

    for _ in tqdm(range(trials), desc="Simulating"):
        ground_truth_unique = np.random.choice([True, False])
        target, peers, _, _ = generate_ecosystem(dim, n_peers, margin, is_unique=ground_truth_unique)
        
        target.noise_std = noise_std
        for p in peers: p.noise_std = noise_std

        for n_queries in query_steps:
            if n_queries < dim: continue 
            
            # =========================================================
            # Strategy 1: Passive Sampling (Random Gaussian)
            # =========================================================
            # Expected row norm is sqrt(dim)
            X_pas = np.random.randn(n_queries, dim)
            
            # 1. Observe
            y_t_pas = target._forward(X_pas)
            Y_p_pas = np.array([p._forward(X_pas) for p in peers]).T 
            
            # 2. Estimate (Ridge)
            H_pas = np.linalg.inv(X_pas.T @ X_pas + 1e-6*np.eye(dim)) @ X_pas.T
            beta_t_hat_pas = H_pas @ y_t_pas
            beta_p_hat_pas = H_pas @ Y_p_pas
            
            # 3. Geometric Distance
            dist_pas, _ = DISCOSolver.solve_weights_and_distance(
                target_vec=beta_t_hat_pas, peer_matrix=beta_p_hat_pas
            )
            
            # =========================================================
            # Strategy 2: Active Auditing (Optimal Orthogonal)
            # =========================================================
            # CRITICAL FIX: Scale Identity by sqrt(dim) to match Gaussian energy!
            # If we don't do this, Active probes are sqrt(d) times weaker than Passive probes.
            
            repeats = n_queries // dim
            remainder = n_queries % dim
            
            # Scaled Orthogonal Basis
            X_base = np.eye(dim) * np.sqrt(dim) 
            
            if remainder > 0:
                X_act = np.vstack([X_base] * repeats + [X_base[:remainder]])
            else:
                X_act = np.vstack([X_base] * repeats)
            
            # 1. Observe
            y_t_act = target._forward(X_act)
            Y_p_act = np.array([p._forward(X_act) for p in peers]).T
            
            # 2. Estimate (Active OLS)
            H_act = np.linalg.inv(X_act.T @ X_act + 1e-9*np.eye(dim)) @ X_act.T
            beta_t_hat_act = H_act @ y_t_act
            beta_p_hat_act = H_act @ Y_p_act
            
            # 3. Geometric Distance
            dist_act, _ = DISCOSolver.solve_weights_and_distance(
                target_vec=beta_t_hat_act, peer_matrix=beta_p_hat_act
            )
            
            # =========================================================
            # Decision
            # =========================================================
            threshold = margin / 2.0
            pred_pas = dist_pas > threshold
            pred_act = dist_act > threshold

            results.append({
                "queries": n_queries,
                "method": "Passive",
                "correct": pred_pas == ground_truth_unique,
                "error": int(pred_pas != ground_truth_unique) 
            })
            results.append({
                "queries": n_queries,
                "method": "Active",
                "correct": pred_act == ground_truth_unique,
                "error": int(pred_act != ground_truth_unique)
            })

    # Save
    df = pd.DataFrame(results)
    output_path = "results/tables/exp1_scaling_law.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.6)
    args = parser.parse_args()
    
    run_experiment(dim=args.dim, trials=args.trials, noise_std=args.noise, margin=args.margin)