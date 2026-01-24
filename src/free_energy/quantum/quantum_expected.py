"""Quantum expected free energy for planning."""
import torch
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix


@dataclass
class QuantumExpectedFreeEnergy:
    """
    Quantum expected free energy for action selection.
    
    Extends classical expected free energy to density matrices.
    """
    
    temperature: float = 1.0
    
    def compute(
        self,
        rho_future: DensityMatrix,
        observation_operator: torch.Tensor,
        preferred_state: DensityMatrix
    ) -> float:
        """
        Compute quantum expected free energy.
        
        G_Q = Tr(ρH_obs) - T·S(ρ) + D_Q(ρ || ρ_pref)
        """
        from ...quantum.information.entropy import von_neumann_entropy, relative_entropy
        
        # Expected observation entropy
        obs_term = torch.trace(rho_future.matrix @ observation_operator).real.item()
        
        # Von Neumann entropy
        entropy = rho_future.von_neumann_entropy()
        
        # Quantum relative entropy (divergence from preferred)
        divergence = relative_entropy(rho_future.matrix, preferred_state.matrix)
        
        return obs_term - self.temperature * entropy + divergence
