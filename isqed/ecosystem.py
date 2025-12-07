
from isqed.core import ModelUnit

class Ecosystem:
    def __init__(self, target: ModelUnit, peers: list[ModelUnit]):
        self.target = target
        self.peers = peers
    
    def batched_query(self, X, Thetas, intervention):
        """
        Query all models in parallel
        Return: 
            y_target: (batch_size,)
            Y_peers: (batch_size, n_peers)
        """
        # --- IGNORE ---
        pass