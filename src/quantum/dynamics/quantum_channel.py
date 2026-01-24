"""Quantum channels (CPTP maps)."""

import torch
from typing import List
from dataclasses import dataclass


@dataclass  
class QuantumChannel:
    """Base class for quantum channels."""
    
    def apply(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class KrausChannel(QuantumChannel):
    """Channel defined by Kraus operators: ρ → Σᵢ Kᵢ ρ Kᵢ†."""
    
    kraus_operators: List[torch.Tensor]
    
    def apply(self, rho: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(rho)
        for K in self.kraus_operators:
            result += K @ rho @ K.conj().T
        return result
