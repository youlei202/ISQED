import cvxpy as cp
import numpy as np

class DISCOSolver:
    """
    General Convex Optimization Solver.
    Solves the problem: Find w in Simplex that minimizes || target - peers @ w ||_2
    """
    
    @staticmethod
    def solve_weights_and_distance(target_vec: np.ndarray, peer_matrix: np.ndarray):
        """
        Args:
            target_vec: shape (D,). Can be Y vector or Beta vector.
            peer_matrix: shape (D, N_peers). Each column is a peer's vector.

        Returns:
            distance: float, projection residual (PIER or Margin)
            weights: np.ndarray, shape (N_peers,)
        """
        D, N_peers = peer_matrix.shape
        
        # Define variable
        w = cp.Variable(N_peers)
        
        # Objective: Minimize || target_vec - peer_matrix @ w ||_2
        # Core formula from your paper: min || beta_t - sum(w_j * beta_j) ||
        objective = cp.Minimize(cp.norm(target_vec - peer_matrix @ w, 2))
        
        # Constraints: Simplex
        constraints = [w >= 0, cp.sum(w) == 1]
        
        # Solve the problem
        prob = cp.Problem(objective, constraints)
        
        # Use SCS or ECOS solver for robustness
        try:
            prob.solve(solver=cp.ECOS)
        except:
            prob.solve(solver=cp.SCS)
            
        return prob.value, w.value