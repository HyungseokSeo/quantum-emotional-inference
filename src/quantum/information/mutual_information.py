"""Quantum mutual information."""
import torch
from typing import Tuple
from .entropy import von_neumann_entropy
from .entanglement import entanglement_entropy

def quantum_mutual_information(
    rho_AB: torch.Tensor, 
    dims: Tuple[int, int]
) -> float:
    """I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)."""
    S_A = entanglement_entropy(rho_AB, dims, subsystem=1)
    S_B = entanglement_entropy(rho_AB, dims, subsystem=0)
    S_AB = von_neumann_entropy(rho_AB)
    return S_A + S_B - S_AB
