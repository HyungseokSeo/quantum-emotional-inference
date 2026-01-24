"""Complete physicality validation."""
import torch
from .positivity import is_positive_semidefinite
from .trace import is_unit_trace

def is_hermitian(matrix: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if matrix is Hermitian."""
    return torch.allclose(matrix, matrix.conj().T, atol=tol)

def is_valid_density_matrix(rho: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check all physical constraints: Hermitian, positive, unit trace."""
    return (is_hermitian(rho, tol) and 
            is_positive_semidefinite(rho, tol) and 
            is_unit_trace(rho, tol))

def is_cptp(kraus_operators: list, tol: float = 1e-6) -> bool:
    """Check if Kraus operators define a CPTP map: Σᵢ Kᵢ†Kᵢ = I."""
    dim = kraus_operators[0].shape[0]
    total = sum(K.conj().T @ K for K in kraus_operators)
    identity = torch.eye(dim, dtype=total.dtype)
    return torch.allclose(total, identity, atol=tol)
