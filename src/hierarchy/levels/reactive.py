"""
Reactive level: ~100ms, high coherence.

Pre-reflective emotional processing. States maintain
superposition (ambivalence) before conscious access.
"""
import torch
from typing import Optional
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig
from ...quantum.dynamics.lindblad import LindbladDynamics


@dataclass
class ReactiveLevel:
    """
    Reactive (fast, pre-conscious) emotional processing.
    
    Characteristics:
    - Timescale: ~100ms
    - High coherence preservation
    - Minimal environmental coupling
    - Full emotional superposition maintained
    """
    
    n_emotions: int
    timescale: float = 0.1  # 100ms
    decoherence_rate: float = 0.1  # Weak decoherence
    
    def __post_init__(self):
        # Weak Lindblad dynamics (preserves coherence)
        self.dynamics = LindbladDynamics.dephasing(
            self.n_emotions,
            rate=self.decoherence_rate
        )
        self.config = DensityMatrixConfig(n_dims=self.n_emotions)
    
    def process(
        self,
        rho_input: DensityMatrix,
        duration: Optional[float] = None
    ) -> DensityMatrix:
        """
        Process input through reactive level.
        
        Maintains high coherence due to weak decoherence.
        """
        if duration is None:
            duration = self.timescale
        
        t_span = torch.linspace(0, duration, 10)
        trajectory = self.dynamics.evolve(rho_input, t_span)
        
        return trajectory[-1]
    
    def coherence_after_processing(self, rho_input: DensityMatrix) -> float:
        """Check coherence retention after reactive processing."""
        rho_output = self.process(rho_input)
        return rho_output.l1_coherence() / (rho_input.l1_coherence() + 1e-10)
