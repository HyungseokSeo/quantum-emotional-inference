"""
Classical uncertainty: probabilistic mixture without coherence.

Contrast with ambivalence - uncertainty is "either/or", not "both".
"""
import torch
from typing import List
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix
from ...quantum.states.emotional_basis import EmotionalBasis


@dataclass
class ClassicalUncertainty:
    """
    Classical probabilistic uncertainty about emotional state.
    
    No coherence - just doesn't know which emotion.
    """
    emotions: List[str]
    probabilities: List[float]
    
    def to_density_matrix(self, basis: EmotionalBasis) -> DensityMatrix:
        """Create diagonal density matrix (no coherences)."""
        return basis.uncertain(self.emotions, self.probabilities)
    
    @property
    def entropy(self) -> float:
        """Shannon entropy of the distribution."""
        probs = torch.tensor(self.probabilities)
        probs = probs / probs.sum()
        return -torch.sum(probs * torch.log(probs + 1e-10)).item()


def distinguish_ambivalence_uncertainty(
    rho: DensityMatrix,
    threshold: float = 0.1
) -> str:
    """
    Classify state as ambivalent or uncertain.
    
    Key distinction:
    - Ambivalent states have non-zero off-diagonal coherences
    - Uncertain states are diagonal (or nearly so)
    """
    coherence = rho.l1_coherence()
    
    if coherence > threshold:
        return "ambivalent"
    return "uncertain"
