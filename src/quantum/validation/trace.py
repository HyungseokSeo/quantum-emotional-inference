"""Trace normalization checks."""
import torch

def is_unit_trace(matrix: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if Tr(matrix) = 1."""
    trace = torch.trace(matrix).real
    return bool(torch.isclose(trace, torch.tensor(1.0), atol=tol))

def normalize_trace(matrix: torch.Tensor) -> torch.Tensor:
    """Normalize matrix to have unit trace."""
    trace = torch.trace(matrix).real
    if trace < 1e-10:
        raise ValueError("Cannot normalize matrix with zero trace")
    return matrix / trace
