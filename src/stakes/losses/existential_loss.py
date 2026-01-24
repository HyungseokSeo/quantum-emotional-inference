"""Existential loss: survival depends on good inference."""
import torch
import torch.nn as nn


class ExistentialLoss(nn.Module):
    """
    Loss function that ties model "existence" to performance.
    
    High loss = existential threat
    """
    
    def __init__(
        self,
        base_loss: nn.Module = None,
        existence_threshold: float = 2.0,
        penalty_scale: float = 10.0
    ):
        super().__init__()
        self.base_loss = base_loss or nn.CrossEntropyLoss()
        self.threshold = existence_threshold
        self.penalty_scale = penalty_scale
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with existential stakes.
        
        Loss increases dramatically when approaching threshold.
        """
        base = self.base_loss(predictions, targets)
        
        # Existential penalty: exponential growth near threshold
        if base > self.threshold * 0.8:
            penalty = self.penalty_scale * torch.exp(base - self.threshold * 0.8)
            return base + penalty
        
        return base
