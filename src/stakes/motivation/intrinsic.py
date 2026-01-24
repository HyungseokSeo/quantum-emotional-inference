"""Intrinsic motivation from free energy."""
import torch
from dataclasses import dataclass


@dataclass
class IntrinsicMotivation:
    """
    Intrinsic motivation as drive to minimize free energy.
    
    Creates genuine stake in emotional inference.
    """
    
    exploration_bonus: float = 0.1
    
    def compute_motivation(
        self,
        free_energy: float,
        uncertainty: float
    ) -> float:
        """
        Compute motivational drive.
        
        Balances free energy minimization with exploration.
        """
        exploitation = -free_energy  # Want to minimize F
        exploration = self.exploration_bonus * uncertainty
        return exploitation + exploration
