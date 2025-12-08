# isqed/real_world.py
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from isqed.core import ModelUnit, Intervention

class HuggingFaceWrapper(ModelUnit):
    """
    Wraps a Hugging Face model for ISQED auditing.
    Task: Sentiment Analysis (Binary Classification).
    Output: Probability of 'Positive' class.
    """
    def __init__(self, model_name, device='cpu'):
        super().__init__(name=model_name)
        self.device = device
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval() # Set to inference mode

    def _forward(self, text_input):
        # text_input is a string (already perturbed)
        inputs = self.tokenizer(
            text_input, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply Softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # We assume binary classification (SST-2). 
        # Return the probability of label 1 (Positive) as the scalar response Y.
        # Shape: (1,) scalar
        return probs[0, 1].item()

    
class MaskingIntervention(Intervention):
    def __init__(self, mask_token: str = "[MASK]"):
        self.mask_token = mask_token

    def apply(self, text: str, theta: float, seed: int = None) -> str:
        """
        Apply masking with a fixed seed to ensure that target and peers
        see exactly the same corrupted input for a given (text, theta).

        Args:
            text: original input sentence
            theta: masking ratio in [0, 1]
            seed: integer seed to make the masking pattern deterministic

        Returns:
            perturbed_text: sentence with a fraction of tokens replaced by [MASK]
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        words = text.split()
        n = len(words)
        n_mask = int(n * theta)

        if n_mask > 0 and n > 0:
            mask_idx = rng.choice(n, n_mask, replace=False)
            for i in mask_idx:
                words[i] = self.mask_token
        return " ".join(words)
