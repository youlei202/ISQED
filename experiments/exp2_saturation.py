# experiments/exp2_saturation.py

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# --- Path Setup ---
# Add parent directory to path to import 'isqed'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.geometry import DISCOSolver

def generate_random_vectors(dim, count):
    """
    Generates 'count' random unit vectors in 'dim' dimensions.
    Simulates a 'random model' in the ecosystem.
    
    Sampling from isotropic Gaussian and normalizing yields 
    Uniform distribution on the unit sphere S^(d-1).
    """
    vecs = np.random.randn(dim, count)
    norms = np.linalg.norm(vecs, axis=0)
    return vecs / norms

def run_experiment(dim=10, max_peers=50, trials=50):
    """
    Runs the Saturation Phase Transition experiment.
    
    Args:
        dim (int): The intrinsic dimension of the task space.
        max_peers (int): The maximum ecosystem size to simulate.
        trials (int): Number of Monte Carlo repeats per step.
    """
    print(f"--- Running Exp 2: Ecosystem Saturation (Phase Transition) ---")
    print(f"Feature Dimension (d): {dim}")
    print(f"Max Peers (N): {max_peers}")
    print(f"Trials per N: {trials}")
    
    results = []
    
    # We vary the Number of Peers (N)
    # Range: From 2 up to max_peers. 
    # We want dense sampling near 'd' to capture the transition.
    peer_counts = range(2, max_peers + 1)
    
    for n_peers in tqdm(peer_counts, desc="Simulating Ecosystem Growth"):
        
        # Monte Carlo Trials for statistical stability
        for _ in range(trials):
            
            # 1. Generate Ecosystem Geometry
            # Peers: Matrix of shape (d, n_peers)
            peer_matrix = generate_random_vectors(dim, n_peers)
            
            # Target: Vector of shape (d,)
            # The target is just another random model from the same distribution
            target_vec = generate_random_vectors(dim, 1).flatten()
            
            # 2. Compute PIER (Distance to Convex Hull)
            # We use the generic DISCOSolver engine.
            # Here 'target_vec' is beta_t, 'peer_matrix' is [beta_1, ..., beta_N]
            dist, _ = DISCOSolver.solve_weights_and_distance(target_vec, peer_matrix)
            
            # 3. Log Results
            # 'is_redundant': Boolean flag if PIER is effectively zero (numerical tolerance)
            # We use a small epsilon because solvers have precision limits.
            is_redundant = dist < 1e-4
            
            results.append({
                "dimension": dim,
                "n_peers": n_peers,
                "pier_score": dist,
                "is_redundant": int(is_redundant),
                "ratio_N_d": n_peers / dim  # Normalized capacity ratio
            })

    # --- Save Results ---
    df = pd.DataFrame(results)
    output_dir = "results/tables"
    output_file = os.path.join(output_dir, "exp2_saturation.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Experiment completed. Results saved to {output_file}")
    
    # --- Quick Summary Statistics ---
    summary = df.groupby("n_peers")["pier_score"].mean()
    print("\nSample Data (Mean PIER vs Peers):")
    print(summary.head(5))
    print("...")
    print(summary.tail(5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ISQED Saturation Experiment")
    parser.add_argument("--dim", type=int, default=10, help="Intrinsic dimension of the task space")
    parser.add_argument("--max_peers", type=int, default=40, help="Maximum number of peers to simulate")
    parser.add_argument("--trials", type=int, default=100, help="Monte Carlo trials per step")
    args = parser.parse_args()
    
    run_experiment(dim=args.dim, max_peers=args.max_peers, trials=args.trials)