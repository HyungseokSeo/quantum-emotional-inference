"""Expected free energy for planning/action selection."""
import torch
from dataclasses import dataclass


@dataclass  
class ExpectedFreeEnergy:
    """
    Expected free energy for future-oriented inference.
    
    G = E_q[log q(s|π) - log p(o,s|π)]
      ≈ Ambiguity + Risk
    """
    
    def compute(
        self,
        q_s_given_pi: torch.Tensor,  # Expected states under policy
        log_likelihood: torch.Tensor,
        preferred_outcomes: torch.Tensor
    ) -> float:
        """Compute expected free energy."""
        q = q_s_given_pi + 1e-10
        
        # Ambiguity: expected entropy of likelihood
        ambiguity = -torch.sum(q * log_likelihood)
        
        # Risk: KL from preferred outcomes
        risk = torch.sum(q * (torch.log(q) - torch.log(preferred_outcomes + 1e-10)))
        
        return (ambiguity + risk).item()
