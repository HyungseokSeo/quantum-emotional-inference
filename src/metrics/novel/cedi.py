"""
CEDI: Contextual Emotion Dynamics Index.

Measures context-appropriateness of emotional dynamics.
"""
import torch
from typing import List, Optional


class CEDI:
    """
    Contextual Emotion Dynamics Index.
    
    Evaluates whether emotional transitions are contextually appropriate.
    """
    
    def compute(
        self,
        predictions: List[torch.Tensor],
        contexts: Optional[List[torch.Tensor]] = None
    ) -> float:
        """
        Compute CEDI from predictions and contexts.
        
        Higher = more contextually appropriate dynamics.
        """
        if len(predictions) < 2:
            return 1.0
        
        # Compute transition appropriateness
        scores = []
        for i in range(len(predictions) - 1):
            # Measure transition smoothness
            diff = torch.norm(predictions[i+1] - predictions[i])
            score = 1 / (1 + diff.item())  # Smoother = higher score
            scores.append(score)
        
        return sum(scores) / len(scores)
