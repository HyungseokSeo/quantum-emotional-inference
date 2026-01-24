"""Entanglement measures for composite emotional states."""
import torch
from typing import Tuple

def concurrence(rho: torch.Tensor) -> float:
    """Concurrence for 2-qubit states."""
    if rho.shape != (4, 4):
        raise ValueError("Concurrence only defined for 2-qubit states")
    
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=rho.dtype)
    Y = torch.kron(sigma_y, sigma_y)
    
    rho_tilde = Y @ rho.conj() @ Y
    R = rho @ rho_tilde
    eigenvalues = torch.sqrt(torch.clamp(torch.linalg.eigvalsh(R).real, min=0))
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]
    
    return max(0, (eigenvalues[0] - eigenvalues[1:].sum()).item())

def entanglement_entropy(rho: torch.Tensor, dims: Tuple[int, int], subsystem: int = 0) -> float:
    """Von Neumann entropy of reduced state."""
    from ..states.density_matrix import DensityMatrix, DensityMatrixConfig
    from .entropy import von_neumann_entropy
    
    d1, d2 = dims
    rho_reshaped = rho.reshape(d1, d2, d1, d2)
    
    if subsystem == 0:
        reduced = torch.einsum('ijkj->ik', rho_reshaped)
    else:
        reduced = torch.einsum('ijil->jl', rho_reshaped)
    
    return von_neumann_entropy(reduced)
