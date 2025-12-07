import cvxpy as cp
import numpy as np

class DISCOSolver:
    """Solving the simplex projection: min ||y - Xw||^2 s.t. w in Simplex"""
    
    @staticmethod
    def solve_weights(target_y: np.ndarray, peer_Y: np.ndarray, lambda_reg=1e-4):
        """
        target_y: (m,) 
        peer_Y: (m, N_peers)
        """
        n_peers = peer_Y.shape[1]
        w = cp.Variable(n_peers)
        
        # Objective：MSE + L2 Regularization
        objective = cp.Minimize(
            cp.sum_squares(target_y - peer_Y @ w) + lambda_reg * cp.sum_squares(w)
        )
        
        # Constraints: Simplex (non-negative, sum to 1)
        constraints = [w >= 0, cp.sum(w) == 1]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return w.value

    @staticmethod
    def compute_pier(target_y, peer_Y, w):
        """Computing PIER residuals"""
        y_hat = peer_Y @ w
        residuals = target_y - y_hat
        return residuals