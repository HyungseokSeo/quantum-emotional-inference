"""
Context effects in emotional judgment.

The context in which emotion is measured affects the outcome.
This is captured by incompatible measurements/bases.
"""
import torch
from typing import List, Optional
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix
from ...quantum.operators.measurement import ProjectiveMeasurement


@dataclass
class ContextEffects:
    """
    Models context-dependent emotional measurement.
    
    Same underlying state can yield different outcomes
    depending on measurement context (basis).
    """
    
    default_measurement: ProjectiveMeasurement
    
    def create_context_measurement(
        self,
        rotation_angle: float
    ) -> ProjectiveMeasurement:
        """
        Create measurement in rotated basis.
        
        Different contexts = different measurement bases.
        """
        n = self.default_measurement.n_outcomes
        
        # Simple rotation for 2D case
        if n == 2:
            c, s = torch.cos(torch.tensor(rotation_angle)), torch.sin(torch.tensor(rotation_angle))
            rotation = torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)
            
            new_projectors = []
            for proj in self.default_measurement.projectors:
                new_proj = rotation @ proj @ rotation.conj().T
                new_projectors.append(new_proj)
            
            return ProjectiveMeasurement(new_projectors, self.default_measurement.labels)
        
        return self.default_measurement
    
    def context_effect(
        self,
        rho: DensityMatrix,
        context_angle: float
    ) -> dict:
        """
        Measure state in different context.
        
        Returns comparison between default and rotated measurements.
        """
        default_probs = self.default_measurement.probabilities(rho.matrix)
        
        context_meas = self.create_context_measurement(context_angle)
        context_probs = context_meas.probabilities(rho.matrix)
        
        return {
            "default_probs": default_probs.tolist(),
            "context_probs": context_probs.tolist(),
            "difference": (context_probs - default_probs).tolist(),
            "total_variation": torch.sum(torch.abs(context_probs - default_probs)).item() / 2
        }
