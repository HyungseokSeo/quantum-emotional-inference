"""
Quantum variational free energy.

Key theoretical contribution: Replace Shannon entropy with von Neumann entropy.

F_Q = E_ρ[H] - T·S(ρ)

where S(ρ) = -Tr(ρ ln ρ) is the von Neumann entropy.

This allows:
- Coherent superpositions to have lower free energy than mixtures
- Natural emergence of ambivalence vs uncertainty distinction
"""
import torch
from typing import Optional
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix
from ...quantum.information.entropy import von_neumann_entropy


@dataclass
class QuantumVariationalFreeEnergy:
    """
    Quantum variational free energy using density matrices.
    
    F_Q(ρ) = Tr(ρH) - T·S(ρ)
           = Energy - Temperature × von Neumann Entropy
    
    Key properties:
    - Pure states: S(ρ) = 0, free energy = energy alone
    - Mixed states: S(ρ) > 0, entropy reduces free energy
    - Coherent superpositions can have different F than mixtures
      with same diagonal probabilities
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def compute(
        self,
        rho: DensityMatrix,
        hamiltonian: torch.Tensor
    ) -> float:
        """
        Compute quantum free energy.
        
        F = Tr(ρH) - T·S(ρ)
        
        Args:
            rho: Quantum state (density matrix)
            hamiltonian: Energy operator
            
        Returns:
            Free energy value
        """
        # Energy: ⟨H⟩ = Tr(ρH)
        energy = torch.trace(rho.matrix @ hamiltonian).real.item()
        
        # Von Neumann entropy: S(ρ) = -Tr(ρ ln ρ)
        entropy = rho.von_neumann_entropy()
        
        return energy - self.temperature * entropy
    
    def compute_from_observations(
        self,
        rho: DensityMatrix,
        log_likelihood_operator: torch.Tensor,
        log_prior_operator: torch.Tensor
    ) -> float:
        """
        Compute F using observation model.
        
        F = -Tr(ρ log p(o|s)) - Tr(ρ log p(s)) - T·S(ρ)
        """
        # Expected negative log likelihood
        energy_likelihood = -torch.trace(rho.matrix @ log_likelihood_operator).real.item()
        
        # Expected negative log prior  
        energy_prior = -torch.trace(rho.matrix @ log_prior_operator).real.item()
        
        # Von Neumann entropy
        entropy = rho.von_neumann_entropy()
        
        return energy_likelihood + energy_prior - self.temperature * entropy
    
    def free_energy_gradient(
        self,
        rho: DensityMatrix,
        hamiltonian: torch.Tensor
    ) -> torch.Tensor:
        """
        Gradient of free energy with respect to density matrix.
        
        ∂F/∂ρ = H + T(ln ρ + I)
        """
        # Matrix logarithm
        eigenvalues, eigenvectors = torch.linalg.eigh(rho.matrix)
        eigenvalues = torch.clamp(eigenvalues.real, min=1e-10)
        log_rho = eigenvectors @ torch.diag(torch.log(eigenvalues).to(rho.matrix.dtype)) @ eigenvectors.conj().T
        
        return hamiltonian + self.temperature * (log_rho + torch.eye(rho.n_dims, dtype=rho.matrix.dtype))
    
    def coherence_contribution(self, rho: DensityMatrix) -> float:
        """
        Compute free energy contribution from coherences.
        
        Compares F(ρ) with F(ρ_diagonal).
        """
        # Full state free energy (with coherences)
        # Use identity as Hamiltonian for pure entropy comparison
        H = torch.zeros(rho.n_dims, rho.n_dims, dtype=rho.matrix.dtype)
        F_full = self.compute(rho, H)
        
        # Diagonal state free energy (no coherences)
        diagonal = torch.diag(torch.diag(rho.matrix))
        diagonal_rho = DensityMatrix(diagonal, rho.config, validate=False)
        F_diagonal = self.compute(diagonal_rho, H)
        
        return F_diagonal - F_full  # Positive if coherence reduces F


def free_energy_comparison(
    ambivalent_rho: DensityMatrix,
    uncertain_rho: DensityMatrix,
    hamiltonian: torch.Tensor,
    temperature: float = 1.0
) -> dict:
    """
    Compare free energy of ambivalent vs uncertain states.
    
    Key theoretical test: Do coherent superpositions have
    different free energy than classical mixtures?
    """
    fe_calc = QuantumVariationalFreeEnergy(temperature)
    
    F_ambivalent = fe_calc.compute(ambivalent_rho, hamiltonian)
    F_uncertain = fe_calc.compute(uncertain_rho, hamiltonian)
    
    return {
        "F_ambivalent": F_ambivalent,
        "F_uncertain": F_uncertain,
        "difference": F_uncertain - F_ambivalent,
        "coherence_favored": F_ambivalent < F_uncertain,
        "ambivalent_entropy": ambivalent_rho.von_neumann_entropy(),
        "uncertain_entropy": uncertain_rho.von_neumann_entropy(),
        "ambivalent_coherence": ambivalent_rho.l1_coherence(),
        "uncertain_coherence": uncertain_rho.l1_coherence(),
    }
