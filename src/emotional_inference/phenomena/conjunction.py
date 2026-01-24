"""
Conjunction fallacy in emotional judgment.

People sometimes judge P(A and B) > P(A), violating classical probability.
Quantum probability can account for this via interference.
"""
import torch
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix


@dataclass
class ConjunctionFallacy:
    """
    Models conjunction fallacy in emotional judgment.
    
    Example: "She seems both happy and nervous" might be judged
    more likely than "She seems happy" in certain contexts.
    """
    
    def classical_conjunction(self, p_A: float, p_B: float) -> float:
        """Classical: P(A and B) ≤ min(P(A), P(B))."""
        return min(p_A, p_B)
    
    def quantum_conjunction(
        self,
        rho: DensityMatrix,
        emotion_A: int,
        emotion_B: int
    ) -> float:
        """
        Quantum conjunction probability.
        
        For superposition states, can exceed classical bound.
        """
        probs = rho.probabilities()
        p_A = probs[emotion_A].item()
        p_B = probs[emotion_B].item()
        
        # Get coherence between A and B
        coherence = abs(rho.matrix[emotion_A, emotion_B].item())
        
        # Quantum modification (can exceed classical bound)
        return p_A * p_B + 2 * coherence
    
    def fallacy_magnitude(
        self,
        rho: DensityMatrix,
        emotion_A: int,
        emotion_B: int
    ) -> float:
        """
        Measure how much quantum exceeds classical prediction.
        """
        probs = rho.probabilities()
        classical = self.classical_conjunction(
            probs[emotion_A].item(),
            probs[emotion_B].item()
        )
        quantum = self.quantum_conjunction(rho, emotion_A, emotion_B)
        return quantum - classical
