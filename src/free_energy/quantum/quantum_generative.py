"""Quantum generative model using density matrices."""
import torch
from dataclasses import dataclass

from ...quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig


@dataclass
class QuantumGenerativeModel:
    """
    Quantum generative model for emotional inference.
    
    Uses density matrices throughout:
    - Prior: ρ_prior (prior emotional state)
    - Likelihood: quantum channel (CPTP map)
    - Posterior: ρ_posterior via quantum Bayes
    """
    
    prior_state: DensityMatrix
    n_observations: int
    
    def quantum_bayes_update(
        self,
        observation: int,
        measurement_operators: list
    ) -> DensityMatrix:
        """
        Quantum Bayesian update.
        
        ρ_post = M_o ρ_prior M_o† / Tr(M_o ρ_prior M_o†)
        """
        M = measurement_operators[observation]
        
        numerator = M @ self.prior_state.matrix @ M.conj().T
        trace = torch.trace(numerator).real
        
        if trace < 1e-10:
            raise ValueError(f"Observation {observation} has zero probability")
        
        posterior = numerator / trace
        
        return DensityMatrix(posterior, self.prior_state.config, validate=False)
    
    def observation_probability(
        self,
        observation: int,
        measurement_operators: list
    ) -> float:
        """p(o) = Tr(M_o ρ M_o†)."""
        M = measurement_operators[observation]
        return torch.trace(M @ self.prior_state.matrix @ M.conj().T).real.item()
