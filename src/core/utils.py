"""
Utility functions for the framework.
"""

import torch
import numpy as np
import random
from typing import Optional, Union, Tuple
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to complex dtype if not already."""
    if not tensor.is_complex():
        return tensor.to(torch.complex64)
    return tensor


def to_real(tensor: torch.Tensor) -> torch.Tensor:
    """Extract real part of complex tensor."""
    if tensor.is_complex():
        return tensor.real
    return tensor


def ensure_tensor(
    x: Union[torch.Tensor, np.ndarray, list],
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    """Convert input to torch tensor."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, list):
        x = torch.tensor(x)
    
    if dtype is not None:
        x = x.to(dtype)
    if device is not None:
        x = x.to(device)
    
    return x


def normalize_vector(v: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize vector to unit length."""
    norm = torch.norm(v)
    if norm < eps:
        return v
    return v / norm


def is_hermitian(matrix: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if matrix is Hermitian (A = A†)."""
    return torch.allclose(matrix, matrix.conj().T, atol=tol)


def is_positive_semidefinite(matrix: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if matrix is positive semi-definite (all eigenvalues ≥ 0)."""
    eigenvalues = torch.linalg.eigvalsh(matrix)
    return bool(torch.all(eigenvalues >= -tol))


def is_unit_trace(matrix: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if matrix has unit trace."""
    trace = torch.trace(matrix).real
    return bool(torch.isclose(trace, torch.tensor(1.0), atol=tol))


def matrix_log(matrix: torch.Tensor) -> torch.Tensor:
    """Compute matrix logarithm via eigendecomposition."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    # Clamp small eigenvalues to avoid log(0)
    eigenvalues = torch.clamp(eigenvalues.real, min=1e-10)
    log_eigenvalues = torch.log(eigenvalues)
    return eigenvectors @ torch.diag(log_eigenvalues.to(matrix.dtype)) @ eigenvectors.conj().T


def matrix_exp(matrix: torch.Tensor) -> torch.Tensor:
    """Compute matrix exponential."""
    return torch.linalg.matrix_exp(matrix)


def partial_trace(
    rho: torch.Tensor,
    dims: Tuple[int, int],
    trace_out: int
) -> torch.Tensor:
    """
    Compute partial trace of bipartite density matrix.
    
    Args:
        rho: Density matrix of shape (d1*d2, d1*d2)
        dims: Dimensions of subsystems (d1, d2)
        trace_out: Which subsystem to trace out (0 or 1)
    
    Returns:
        Reduced density matrix
    """
    d1, d2 = dims
    rho_reshaped = rho.reshape(d1, d2, d1, d2)
    
    if trace_out == 0:
        # Trace out first subsystem
        return torch.einsum('ijkj->ik', rho_reshaped)
    else:
        # Trace out second subsystem
        return torch.einsum('ijil->jl', rho_reshaped)


def fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> float:
    """
    Compute fidelity between two density matrices.
    
    F(ρ, σ) = (Tr√(√ρ σ √ρ))²
    """
    # Compute √ρ
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    eigenvalues = torch.clamp(eigenvalues.real, min=0)
    sqrt_rho = eigenvectors @ torch.diag(torch.sqrt(eigenvalues).to(rho.dtype)) @ eigenvectors.conj().T
    
    # Compute √ρ σ √ρ
    inner = sqrt_rho @ sigma @ sqrt_rho
    
    # Compute trace of square root
    inner_eigenvalues = torch.linalg.eigvalsh(inner)
    inner_eigenvalues = torch.clamp(inner_eigenvalues.real, min=0)
    trace_sqrt = torch.sum(torch.sqrt(inner_eigenvalues))
    
    return (trace_sqrt ** 2).item()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Union[str, Path],
    **kwargs
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
