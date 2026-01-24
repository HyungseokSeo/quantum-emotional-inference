"""Collapse via verbal report."""
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix
from ...quantum.operators.measurement import ProjectiveMeasurement


@dataclass
class VerbalReportCollapse:
    """
    Emotional collapse through verbal report.
    
    Saying "I feel happy" collapses state toward that report.
    """
    
    n_emotions: int
    emotion_labels: list
    report_accuracy: float = 0.9  # How well report reflects true state
    
    def report(
        self,
        rho: DensityMatrix,
        reported_emotion: str,
        confidence: float = 1.0
    ) -> Tuple[DensityMatrix, float]:
        """
        Make verbal report of emotional state.
        
        Args:
            rho: Current state
            reported_emotion: What emotion is reported
            confidence: How confident the report is
            
        Returns:
            (collapsed_state, report_probability)
        """
        idx = self.emotion_labels.index(reported_emotion)
        
        # Probability that this report reflects true state
        prob = rho.probabilities()[idx].item()
        
        # Collapse toward reported emotion
        target = torch.zeros(self.n_emotions, self.n_emotions, dtype=rho.matrix.dtype)
        target[idx, idx] = 1.0
        
        collapse_strength = confidence * self.report_accuracy
        new_matrix = (1 - collapse_strength) * rho.matrix + collapse_strength * target
        
        # Renormalize
        new_matrix = new_matrix / torch.trace(new_matrix).real
        
        return DensityMatrix(new_matrix, rho.config, validate=False), prob
