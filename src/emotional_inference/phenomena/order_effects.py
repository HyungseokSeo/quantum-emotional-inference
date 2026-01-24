"""
Order effects in emotional judgment.

Measuring emotion A then B gives different results than B then A.
This is a signature of non-commuting observables.
"""
import torch
from typing import Tuple, List
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix
from ...quantum.operators.measurement import ProjectiveMeasurement


@dataclass
class OrderEffects:
    """
    Demonstrates order effects in emotional measurement.
    
    Classical probability: P(A then B) = P(B then A)
    Quantum probability: P(A then B) ≠ P(B then A) in general
    """
    
    measurement: ProjectiveMeasurement
    
    def measure_sequence(
        self,
        rho: DensityMatrix,
        sequence: List[int],
        seed: int = None
    ) -> Tuple[List[int], DensityMatrix]:
        """
        Apply sequence of measurements.
        
        Returns:
            (outcomes, final_state)
        """
        outcomes = []
        current = rho.matrix.clone()
        
        for i, measure_idx in enumerate(sequence):
            # Use different seed for each measurement
            if seed is not None:
                torch.manual_seed(seed + i)
            
            probs = self.measurement.probabilities(current)
            outcome = torch.multinomial(probs, 1).item()
            outcomes.append(outcome)
            current = self.measurement.post_measurement_state(outcome, current)
        
        return outcomes, DensityMatrix(current, rho.config, validate=False)
    
    def order_effect_magnitude(
        self,
        rho: DensityMatrix,
        outcome_A: int,
        outcome_B: int,
        n_samples: int = 1000
    ) -> float:
        """
        Quantify order effect: |P(A,B) - P(B,A)|.
        
        Large value indicates strong quantum-like behavior.
        """
        # P(A then B)
        p_AB = self._joint_probability(rho, [outcome_A, outcome_B], n_samples)
        # P(B then A)
        p_BA = self._joint_probability(rho, [outcome_B, outcome_A], n_samples)
        
        return abs(p_AB - p_BA)
    
    def _joint_probability(
        self,
        rho: DensityMatrix,
        sequence: List[int],
        n_samples: int
    ) -> float:
        """Estimate joint probability via sampling."""
        count = 0
        for i in range(n_samples):
            outcomes, _ = self.measure_sequence(rho, [0, 0], seed=i)
            if outcomes == sequence:
                count += 1
        return count / n_samples
