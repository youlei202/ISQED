from isqed.core import ModelUnit, Intervention

class HuggingFaceWrapper(ModelUnit):
    def __init__(self, model_name, device='cpu'):
        # Load model from HF
        self.model = ... 
        self.tokenizer = ...
    
    def _forward(self, text):
        # Run inference once
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.logits.detach().numpy()

class MaskingIntervention(Intervention):
    """For example, randomly mask theta proportion of words in text"""
    def apply(self, text, theta):
        # Implement masking logic
        masked_text = None
        return masked_text