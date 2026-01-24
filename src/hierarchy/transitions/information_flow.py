"""Information flow between hierarchy levels."""
import torch
from typing import List
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix
from ...quantum.information.entropy import von_neumann_entropy
from ...quantum.information.coherence import l1_coherence


@dataclass
class InformationFlow:
    """
    Tracks information flow through the hierarchy.
    
    Monitors:
    - Entropy changes
    - Coherence decay
    - Information preservation
    """
    
    def compute_flow(
        self,
        trajectory: List[DensityMatrix]
    ) -> dict:
        """Analyze information flow through trajectory."""
        entropies = [rho.von_neumann_entropy() for rho in trajectory]
        coherences = [rho.l1_coherence() for rho in trajectory]
        purities = [rho.purity() for rho in trajectory]
        
        return {
            "entropies": entropies,
            "coherences": coherences,
            "purities": purities,
            "entropy_increase": entropies[-1] - entropies[0],
            "coherence_decay": coherences[0] - coherences[-1],
            "purity_loss": purities[0] - purities[-1],
        }
    
    def mutual_information_preservation(
        self,
        rho_initial: DensityMatrix,
        rho_final: DensityMatrix
    ) -> float:
        """Measure how much information is preserved."""
        # Simplified: compare purities
        return rho_final.purity() / rho_initial.purity()
