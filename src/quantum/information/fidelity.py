"""Fidelity and distance measures."""
import torch

def fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> float:
    """F(ρ,σ) = (Tr√(√ρ σ √ρ))²."""
    eig, vec = torch.linalg.eigh(rho)
    eig = torch.clamp(eig.real, min=0)
    sqrt_rho = vec @ torch.diag(torch.sqrt(eig).to(rho.dtype)) @ vec.conj().T
    
    inner = sqrt_rho @ sigma @ sqrt_rho
    inner_eig = torch.linalg.eigvalsh(inner)
    inner_eig = torch.clamp(inner_eig.real, min=0)
    
    return (torch.sum(torch.sqrt(inner_eig)) ** 2).item()

def trace_distance(rho: torch.Tensor, sigma: torch.Tensor) -> float:
    """D(ρ,σ) = ½||ρ-σ||₁."""
    diff = rho - sigma
    eigenvalues = torch.linalg.eigvalsh(diff)
    return 0.5 * torch.sum(torch.abs(eigenvalues)).item()
