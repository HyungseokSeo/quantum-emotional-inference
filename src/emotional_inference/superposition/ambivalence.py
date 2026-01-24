"""
Ambivalence: genuine emotional superposition.

Ambivalence differs from uncertainty:
- Ambivalence: coherent superposition (off-diagonal terms present)
- Uncertainty: classical mixture (diagonal only)

Key insight: Same probabilities P(happy)=P(sad)=0.5 can arise from:
1. Ambivalence: |ψ⟩ = (|happy⟩ + |sad⟩)/√2 → feels both
2. Uncertainty: ρ = 0.5|happy⟩⟨happy| + 0.5|sad⟩⟨sad| → unsure which
"""
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig
from ...quantum.states.emotional_basis import EmotionalBasis


@dataclass
class Ambivalence:
    """
    Represents genuine emotional ambivalence.
    
    Attributes:
        emotions: Emotions in superposition
        weights: Relative weights (amplitudes squared)
        phases: Relative phases (affect interference)
    """
    emotions: List[str]
    weights: List[float]
    phases: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.phases is None:
            self.phases = [0.0] * len(self.emotions)
    
    def to_density_matrix(self, basis: EmotionalBasis) -> DensityMatrix:
        """Create density matrix with coherent superposition."""
        return basis.ambivalent(self.emotions, self.weights)
    
    @property
    def degree(self) -> float:
        """Ambivalence degree: how mixed the weights are."""
        weights = torch.tensor(self.weights)
        weights = weights / weights.sum()
        # Entropy-based measure
        entropy = -torch.sum(weights * torch.log(weights + 1e-10))
        max_entropy = torch.log(torch.tensor(len(self.emotions), dtype=torch.float32))
        return (entropy / max_entropy).item()


class AmbivalenceDetector:
    """Detect and quantify ambivalence in emotional states."""
    
    def __init__(self, coherence_threshold: float = 0.1):
        self.threshold = coherence_threshold
    
    def is_ambivalent(self, rho: DensityMatrix) -> bool:
        """Check if state has genuine ambivalence (coherence)."""
        return rho.l1_coherence() > self.threshold
    
    def ambivalence_score(self, rho: DensityMatrix) -> float:
        """
        Quantify ambivalence: ratio of coherence to maximum possible.
        
        Score in [0, 1]:
        - 0: pure classical mixture (no ambivalence)
        - 1: maximally coherent pure state
        """
        n = rho.n_dims
        max_coherence = n - 1  # Maximum off-diagonal sum for pure state
        return min(rho.l1_coherence() / max_coherence, 1.0)
    
    def dominant_ambivalence(
        self, 
        rho: DensityMatrix,
        labels: List[str]
    ) -> Tuple[List[str], List[float]]:
        """Find dominant emotions contributing to ambivalence."""
        probs = rho.probabilities()
        
        # Find emotions with significant probability
        threshold = 0.1
        significant = [(labels[i], probs[i].item()) 
                      for i in range(len(labels)) 
                      if probs[i] > threshold]
        
        emotions = [e for e, _ in significant]
        weights = [w for _, w in significant]
        
        return emotions, weights
