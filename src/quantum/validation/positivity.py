"""Positivity checks and projections."""
import torch

def is_positive_semidefinite(matrix: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if matrix has all non-negative eigenvalues."""
    eigenvalues = torch.linalg.eigvalsh(matrix)
    return bool(torch.all(eigenvalues >= -tol))

def project_to_positive(matrix: torch.Tensor) -> torch.Tensor:
    """Project matrix to positive semidefinite cone."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = torch.clamp(eigenvalues.real, min=0)
    return eigenvectors @ torch.diag(eigenvalues.to(matrix.dtype)) @ eigenvectors.conj().T
