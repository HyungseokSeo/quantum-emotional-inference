"""Stake-weighted loss functions."""
import torch
import torch.nn as nn


class StakeWeightedLoss(nn.Module):
    """
    Weight loss by stakes (importance of correct prediction).
    """
    
    def __init__(self, base_loss: nn.Module = None, stake_power: float = 2.0):
        super().__init__()
        self.base_loss = base_loss or nn.CrossEntropyLoss(reduction='none')
        self.stake_power = stake_power
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        stakes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute stake-weighted loss.
        
        Higher stakes → higher penalty for errors.
        """
        base = self.base_loss(predictions, targets)
        weights = stakes ** self.stake_power
        return (base * weights).mean()
