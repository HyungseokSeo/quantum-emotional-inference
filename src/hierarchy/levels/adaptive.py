"""
Adaptive level: ~1s, partial decoherence.

Context integration and initial collapse. Superposition
begins to resolve based on environmental coupling.
"""
import torch
from typing import Optional
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig
from ...quantum.dynamics.lindblad import LindbladDynamics


@dataclass
class AdaptiveLevel:
    """
    Adaptive (intermediate) emotional processing.
    
    Characteristics:
    - Timescale: ~1s
    - Moderate decoherence
    - Context integration
    - Partial collapse of superposition
    """
    
    n_emotions: int
    timescale: float = 1.0  # 1s
    decoherence_rate: float = 1.0  # Moderate decoherence
    
    def __post_init__(self):
        self.dynamics = LindbladDynamics.dephasing(
            self.n_emotions,
            rate=self.decoherence_rate
        )
        self.config = DensityMatrixConfig(n_dims=self.n_emotions)
    
    def process(
        self,
        rho_input: DensityMatrix,
        context: Optional[torch.Tensor] = None,
        duration: Optional[float] = None
    ) -> DensityMatrix:
        """
        Process through adaptive level with context integration.
        """
        if duration is None:
            duration = self.timescale
        
        # Apply context bias if provided
        if context is not None:
            # Context modifies the state before decoherence
            rho_input = self._apply_context(rho_input, context)
        
        t_span = torch.linspace(0, duration, 20)
        trajectory = self.dynamics.evolve(rho_input, t_span)
        
        return trajectory[-1]
    
    def _apply_context(
        self,
        rho: DensityMatrix,
        context: torch.Tensor
    ) -> DensityMatrix:
        """Modulate state based on context."""
        # Context as bias toward certain emotions
        context_bias = torch.diag(torch.softmax(context, dim=0))
        
        # Apply bias (preserves trace)
        new_matrix = context_bias @ rho.matrix @ context_bias
        new_matrix = new_matrix / torch.trace(new_matrix).real
        
        return DensityMatrix(new_matrix, rho.config, validate=False)
