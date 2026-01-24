"""Collapse decoder: density matrix to discrete emotion."""
import torch
import torch.nn as nn


class CollapseDecoder(nn.Module):
    """
    Decodes density matrix to discrete emotion via measurement.
    
    Simulates the collapse process:
    ρ → p(emotion) → sampled emotion
    """
    
    def __init__(self, n_emotions: int, temperature: float = 1.0):
        super().__init__()
        self.n_emotions = n_emotions
        self.temperature = temperature
    
    def forward(
        self,
        rho: torch.Tensor,
        sample: bool = False
    ) -> torch.Tensor:
        """
        Decode density matrix.
        
        Args:
            rho: Density matrices [B, n, n]
            sample: If True, sample from distribution
            
        Returns:
            Probabilities [B, n] or sampled indices [B]
        """
        # Extract diagonal probabilities
        probs = torch.diagonal(rho, dim1=-2, dim2=-1).real
        
        # Temperature scaling
        probs = torch.softmax(probs / self.temperature, dim=-1)
        
        if sample:
            return torch.multinomial(probs, 1).squeeze(-1)
        
        return probs
