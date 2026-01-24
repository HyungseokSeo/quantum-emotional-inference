"""Generative model for emotion inference."""
import torch
from typing import Tuple
from dataclasses import dataclass


@dataclass
class GenerativeModel:
    """
    Generative model p(o, s) = p(o|s)p(s).
    
    For emotion: s = emotional state, o = observation (face, etc.)
    """
    
    likelihood: torch.Tensor  # p(o|s) matrix
    prior: torch.Tensor  # p(s) vector
    
    def joint_probability(self) -> torch.Tensor:
        """p(o, s) = p(o|s)p(s)."""
        return self.likelihood * self.prior.unsqueeze(0)
    
    def marginal_observation(self) -> torch.Tensor:
        """p(o) = Σₛ p(o|s)p(s)."""
        return torch.sum(self.joint_probability(), dim=1)
    
    def posterior(self, observation: int) -> torch.Tensor:
        """p(s|o) ∝ p(o|s)p(s) via Bayes."""
        numerator = self.likelihood[observation] * self.prior
        return numerator / numerator.sum()
