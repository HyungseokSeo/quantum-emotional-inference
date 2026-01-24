"""Coherence-related metrics."""
import torch
from typing import List

from ...quantum.states.density_matrix import DensityMatrix


def coherence_preservation(
    initial: DensityMatrix,
    final: DensityMatrix
) -> float:
    """Measure coherence preservation through processing."""
    c_initial = initial.l1_coherence()
    c_final = final.l1_coherence()
    
    if c_initial < 1e-10:
        return 1.0 if c_final < 1e-10 else 0.0
    
    return c_final / c_initial


def decoherence_rate(
    trajectory: List[DensityMatrix],
    times: torch.Tensor
) -> float:
    """Estimate decoherence rate from trajectory."""
    coherences = [rho.l1_coherence() for rho in trajectory]
    
    if len(coherences) < 2:
        return 0.0
    
    # Fit exponential decay
    log_coherences = [torch.log(torch.tensor(c + 1e-10)) for c in coherences]
    
    # Simple linear regression on log scale
    t = times.numpy()
    y = torch.tensor(log_coherences).numpy()
    
    # Rate = -slope
    slope = (y[-1] - y[0]) / (t[-1] - t[0])
    return -slope
