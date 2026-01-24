"""Temporal binding across hierarchy levels."""
import torch
from typing import List, Optional
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix


@dataclass
class TemporalBinding:
    """
    Binds information across different temporal scales.
    
    Integrates fast (reactive) and slow (reflective) processing
    into coherent emotional experience.
    """
    
    integration_weights: List[float] = None
    
    def __post_init__(self):
        if self.integration_weights is None:
            self.integration_weights = [0.5, 0.3, 0.2]  # Reactive, adaptive, reflective
    
    def integrate(
        self,
        reactive_state: DensityMatrix,
        adaptive_state: DensityMatrix,
        reflective_state: DensityMatrix
    ) -> DensityMatrix:
        """
        Integrate states from all levels.
        
        Weighted combination respecting density matrix constraints.
        """
        weights = torch.tensor(self.integration_weights)
        weights = weights / weights.sum()
        
        # Weighted sum of density matrices
        integrated = (weights[0] * reactive_state.matrix +
                     weights[1] * adaptive_state.matrix +
                     weights[2] * reflective_state.matrix)
        
        # Ensure validity
        integrated = (integrated + integrated.conj().T) / 2  # Hermiticity
        integrated = integrated / torch.trace(integrated).real  # Unit trace
        
        return DensityMatrix(integrated, reactive_state.config, validate=False)
