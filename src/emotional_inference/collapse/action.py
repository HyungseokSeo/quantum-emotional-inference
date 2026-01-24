"""Collapse via behavioral commitment."""
import torch
from typing import Optional
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix


@dataclass
class ActionCollapse:
    """
    Emotional collapse through action/behavior.
    
    Taking emotion-driven action commits to that emotion,
    collapsing superposition.
    """
    
    commitment_strength: float = 1.0
    
    def act(
        self,
        rho: DensityMatrix,
        action_emotion: int
    ) -> DensityMatrix:
        """
        Commit to action associated with specific emotion.
        
        Strong commitment → near-complete collapse to action_emotion.
        """
        n = rho.n_dims
        
        # Create biased state favoring action_emotion
        target = torch.zeros(n, n, dtype=rho.matrix.dtype)
        target[action_emotion, action_emotion] = 1.0
        
        # Interpolate based on commitment strength
        new_matrix = ((1 - self.commitment_strength) * rho.matrix + 
                     self.commitment_strength * target)
        
        # Renormalize
        trace = torch.trace(new_matrix).real
        new_matrix = new_matrix / trace
        
        return DensityMatrix(new_matrix, rho.config, validate=False)
