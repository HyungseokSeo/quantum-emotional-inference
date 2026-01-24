"""
ERCS: Emotional Response Consistency Score.

Measures temporal consistency of emotional predictions.
"""
import torch
from typing import List


class ERCS:
    """
    Emotional Response Consistency Score.
    
    Evaluates how consistently the model predicts emotions
    across temporally related inputs.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def compute(self, predictions: List[torch.Tensor]) -> float:
        """
        Compute ERCS from sequence of predictions.
        
        Higher = more consistent
        """
        if len(predictions) < 2:
            return 1.0
        
        consistencies = []
        for i in range(len(predictions) - 1):
            # Measure similarity between consecutive predictions
            sim = torch.nn.functional.cosine_similarity(
                predictions[i].unsqueeze(0),
                predictions[i+1].unsqueeze(0)
            )
            consistencies.append(sim.item())
        
        return sum(consistencies) / len(consistencies)
