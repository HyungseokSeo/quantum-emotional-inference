"""Entropy measures for quantum states."""
import torch
from typing import Optional

def von_neumann_entropy(rho: torch.Tensor, tol: float = 1e-10) -> float:
    """S(ρ) = -Tr(ρ ln ρ)."""
    eigenvalues = torch.linalg.eigvalsh(rho).real
    eigenvalues = eigenvalues[eigenvalues > tol]
    return -torch.sum(eigenvalues * torch.log(eigenvalues)).item()

def linear_entropy(rho: torch.Tensor) -> float:
    """S_L(ρ) = 1 - Tr(ρ²)."""
    return 1 - torch.trace(rho @ rho).real.item()

def relative_entropy(rho: torch.Tensor, sigma: torch.Tensor, tol: float = 1e-10) -> float:
    """S(ρ||σ) = Tr(ρ(ln ρ - ln σ))."""
    # Eigendecomposition
    eig_rho, _ = torch.linalg.eigh(rho)
    eig_sigma, vec_sigma = torch.linalg.eigh(sigma)
    
    eig_rho = torch.clamp(eig_rho.real, min=tol)
    eig_sigma = torch.clamp(eig_sigma.real, min=tol)
    
    S_rho = -torch.sum(eig_rho * torch.log(eig_rho))
    
    # Tr(ρ ln σ) requires more care
    log_sigma = vec_sigma @ torch.diag(torch.log(eig_sigma).to(sigma.dtype)) @ vec_sigma.conj().T
    cross_term = torch.trace(rho @ log_sigma).real
    
    return (-S_rho - cross_term).item()
