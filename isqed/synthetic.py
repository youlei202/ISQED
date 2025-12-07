from isqed.core import ModelUnit, Intervention
import numpy as np

class LinearStructuralModel(ModelUnit):
    """For the Linear Structural Model in Section 4"""
    def __init__(self, beta: np.ndarray):
        super().__init__(name="linear_synthetic")
        self.beta = beta  # (d,)
    
    def _forward(self, phi_x):
        # Y = phi(x)^T beta
        return np.dot(phi_x, self.beta)

class NoiseIntervention(Intervention):
    """For example, scaling certain dimensions of x by theta"""
    def apply(self, x, theta):
        # For example, scale certain dimensions of x by theta
        return x * theta