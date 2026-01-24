"""Decoherence models for emotional state collapse."""

import torch
from typing import Optional
from dataclasses import dataclass

from ..states.density_matrix import DensityMatrix, DensityMatrixConfig


@dataclass
class Decoherence:
    """Base class for decoherence channels."""
    rate: float
    
    def apply(self, rho: torch.Tensor, dt: float) -> torch.Tensor:
        raise NotImplementedError


class DephasingChannel(Decoherence):
    """Pure dephasing: destroys off-diagonal coherences."""
    
    def apply(self, rho: torch.Tensor, dt: float) -> torch.Tensor:
        decay = torch.exp(-self.rate * dt)
        mask = torch.ones_like(rho)
        mask.fill_diagonal_(1.0)
        off_diag_mask = 1 - torch.eye(rho.shape[0], dtype=rho.dtype, device=rho.device)
        return rho * (torch.eye(rho.shape[0], dtype=rho.dtype, device=rho.device) + 
                      decay * off_diag_mask)


class AmplitudeDamping(Decoherence):
    """Amplitude damping: decay to ground state."""
    
    def apply(self, rho: torch.Tensor, dt: float) -> torch.Tensor:
        gamma = 1 - torch.exp(-self.rate * dt)
        n = rho.shape[0]
        
        # Kraus operators for amplitude damping
        K0 = torch.eye(n, dtype=rho.dtype, device=rho.device)
        K0[1:, 1:] *= torch.sqrt(1 - gamma)
        
        K1 = torch.zeros_like(rho)
        K1[0, 1:] = torch.sqrt(gamma)
        
        return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
