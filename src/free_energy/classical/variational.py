"""Variational free energy with Shannon entropy."""
import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class VariationalFreeEnergy:
    """
    Classical variational free energy.
    
    F = E_q[log q(s) - log p(o,s)]
      = -H[q] + E_q[-log p(o|s)] + E_q[-log p(s)]
      = -Entropy + Energy
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def compute(
        self,
        q: torch.Tensor,  # Approximate posterior (probabilities)
        log_likelihood: torch.Tensor,  # log p(o|s)
        log_prior: torch.Tensor  # log p(s)
    ) -> float:
        """
        Compute variational free energy.
        
        F = Σᵢ q(sᵢ)[log q(sᵢ) - log p(o|sᵢ) - log p(sᵢ)]
        """
        q = q + 1e-10  # Numerical stability
        
        # Entropy: -Σ q log q
        entropy = -torch.sum(q * torch.log(q))
        
        # Expected energy: Σ q [-log p(o|s) - log p(s)]
        energy = -torch.sum(q * (log_likelihood + log_prior))
        
        return (energy - self.temperature * entropy).item()
    
    def shannon_entropy(self, q: torch.Tensor) -> float:
        """H[q] = -Σ q log q."""
        q = q + 1e-10
        return -torch.sum(q * torch.log(q)).item()
