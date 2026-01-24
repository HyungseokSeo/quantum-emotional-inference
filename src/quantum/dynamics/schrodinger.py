"""Schrödinger dynamics for coherent evolution."""

import torch
from typing import List
from dataclasses import dataclass

from ..states.density_matrix import DensityMatrix, DensityMatrixConfig


@dataclass
class SchrodingerDynamics:
    """Coherent (unitary) evolution under Hamiltonian."""
    
    hamiltonian: torch.Tensor
    
    def evolve_ket(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        """Evolve pure state: |ψ(t)⟩ = e^{-iHt}|ψ(0)⟩."""
        U = torch.linalg.matrix_exp(-1j * self.hamiltonian * t)
        return U @ psi
    
    def evolve_density_matrix(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        """Evolve density matrix: ρ(t) = U ρ(0) U†."""
        U = torch.linalg.matrix_exp(-1j * self.hamiltonian * t)
        return U @ rho @ U.conj().T
