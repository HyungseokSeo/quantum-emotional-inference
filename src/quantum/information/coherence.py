"""Coherence measures for quantum states."""
import torch

def l1_coherence(rho: torch.Tensor) -> float:
    """C_l1(ρ) = Σᵢ≠ⱼ |ρᵢⱼ|."""
    off_diag = rho - torch.diag(torch.diag(rho))
    return torch.sum(torch.abs(off_diag)).item()

def relative_entropy_of_coherence(rho: torch.Tensor, tol: float = 1e-10) -> float:
    """C_RE(ρ) = S(ρ_diag) - S(ρ)."""
    from .entropy import von_neumann_entropy
    
    diag = torch.diag(torch.diag(rho))
    S_diag = von_neumann_entropy(diag, tol)
    S_rho = von_neumann_entropy(rho, tol)
    return S_diag - S_rho
