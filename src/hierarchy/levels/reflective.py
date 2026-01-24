"""
Reflective level: ~10s, classical limit.

Conscious access and reportable emotion. Superposition
fully collapsed to definite (classical) state.
"""
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig
from ...quantum.dynamics.lindblad import LindbladDynamics
from ...quantum.operators.measurement import ProjectiveMeasurement


@dataclass
class ReflectiveLevel:
    """
    Reflective (conscious, slow) emotional processing.
    
    Characteristics:
    - Timescale: ~10s
    - Strong decoherence → classical limit
    - Full collapse to reportable state
    - Conscious access to emotion
    """
    
    n_emotions: int
    emotion_labels: list
    timescale: float = 10.0  # 10s
    decoherence_rate: float = 5.0  # Strong decoherence
    
    def __post_init__(self):
        self.dynamics = LindbladDynamics.dephasing(
            self.n_emotions,
            rate=self.decoherence_rate
        )
        self.measurement = ProjectiveMeasurement.computational_basis(
            self.n_emotions,
            labels=self.emotion_labels
        )
        self.config = DensityMatrixConfig(n_dims=self.n_emotions)
    
    def process(
        self,
        rho_input: DensityMatrix,
        duration: Optional[float] = None
    ) -> DensityMatrix:
        """
        Process through reflective level.
        
        Results in near-classical (diagonal) state.
        """
        if duration is None:
            duration = self.timescale
        
        t_span = torch.linspace(0, duration, 50)
        trajectory = self.dynamics.evolve(rho_input, t_span)
        
        return trajectory[-1]
    
    def report(
        self,
        rho: DensityMatrix,
        seed: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Generate conscious report of emotional state.
        
        Returns:
            (reported_emotion, confidence)
        """
        # Process to classical limit
        rho_classical = self.process(rho)
        
        # Get probabilities
        probs = rho_classical.probabilities()
        
        # Most likely emotion
        idx = torch.argmax(probs).item()
        emotion = self.emotion_labels[idx]
        confidence = probs[idx].item()
        
        return emotion, confidence
    
    def introspect(self, rho: DensityMatrix) -> dict:
        """
        Full introspective analysis of emotional state.
        """
        rho_classical = self.process(rho)
        probs = rho_classical.probabilities()
        
        return {
            "probabilities": {label: probs[i].item() 
                            for i, label in enumerate(self.emotion_labels)},
            "dominant": self.emotion_labels[torch.argmax(probs).item()],
            "confidence": torch.max(probs).item(),
            "residual_coherence": rho_classical.l1_coherence(),
            "entropy": rho_classical.von_neumann_entropy(),
        }
