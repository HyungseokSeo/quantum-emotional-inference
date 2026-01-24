"""Collapse via facial/bodily expression."""
import torch
from typing import Tuple, Optional
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix
from ...quantum.operators.measurement import ProjectiveMeasurement


@dataclass
class ExpressionCollapse:
    """
    Emotional collapse through expression.
    
    When we express an emotion (facial expression, body language),
    the superposition collapses to a definite state.
    """
    
    n_emotions: int
    emotion_labels: list
    
    def __post_init__(self):
        self.measurement = ProjectiveMeasurement.computational_basis(
            self.n_emotions,
            labels=self.emotion_labels
        )
    
    def collapse(
        self, 
        rho: DensityMatrix,
        seed: Optional[int] = None
    ) -> Tuple[str, DensityMatrix]:
        """
        Express emotion, collapsing superposition.
        
        Returns:
            (expressed_emotion, collapsed_state)
        """
        outcome, collapsed = self.measurement.measure_density_matrix(rho, seed)
        return self.emotion_labels[outcome], collapsed
    
    def expression_probabilities(self, rho: DensityMatrix) -> dict:
        """Get probability of expressing each emotion."""
        probs = self.measurement.probabilities(rho.matrix)
        return {label: probs[i].item() 
                for i, label in enumerate(self.emotion_labels)}
    
    def partial_expression(
        self, 
        rho: DensityMatrix,
        strength: float = 0.5
    ) -> DensityMatrix:
        """
        Partial expression: incomplete collapse.
        
        Models subtle expressions that partially reveal state.
        """
        # Interpolate between full state and diagonal (measured) state
        diagonal = torch.diag(torch.diag(rho.matrix))
        new_matrix = (1 - strength) * rho.matrix + strength * diagonal
        
        return DensityMatrix(new_matrix, rho.config, validate=False)
