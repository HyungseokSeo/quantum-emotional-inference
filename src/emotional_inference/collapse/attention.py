"""Collapse via introspective attention."""
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix


@dataclass  
class AttentionCollapse:
    """
    Emotional collapse through introspective attention.
    
    When we attend to our emotional state, the act of observation
    can cause (partial) collapse.
    """
    
    attention_strength: float = 1.0  # 0 = no effect, 1 = full collapse
    
    def attend(
        self, 
        rho: DensityMatrix,
        focus_emotion: Optional[int] = None
    ) -> DensityMatrix:
        """
        Apply introspective attention to emotional state.
        
        Args:
            rho: Current emotional state
            focus_emotion: If specified, attend specifically to this emotion
            
        Returns:
            State after attention (partially collapsed)
        """
        if focus_emotion is not None:
            # Focused attention: project onto specific emotion
            proj = torch.zeros_like(rho.matrix)
            proj[focus_emotion, focus_emotion] = 1.0
            
            # Partial projection
            new_matrix = ((1 - self.attention_strength) * rho.matrix + 
                         self.attention_strength * proj @ rho.matrix @ proj)
        else:
            # Diffuse attention: general decoherence
            diagonal = torch.diag(torch.diag(rho.matrix))
            new_matrix = ((1 - self.attention_strength) * rho.matrix + 
                         self.attention_strength * diagonal)
        
        # Renormalize
        trace = torch.trace(new_matrix).real
        new_matrix = new_matrix / trace
        
        return DensityMatrix(new_matrix, rho.config, validate=False)
    
    def repeated_attention(
        self, 
        rho: DensityMatrix,
        n_iterations: int = 5
    ) -> DensityMatrix:
        """Apply repeated attention (Zeno-like effect)."""
        for _ in range(n_iterations):
            rho = self.attend(rho)
        return rho
