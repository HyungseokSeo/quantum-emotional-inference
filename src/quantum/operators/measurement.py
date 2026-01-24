"""
Quantum measurement operators.

Measurements in quantum mechanics are fundamentally different from classical:
- They disturb the system (collapse)
- Outcomes are probabilistic
- Non-commuting measurements are incompatible

For emotional modeling, measurement represents:
- Expressing an emotion (collapsing superposition)
- Introspective attention (self-observation)
- External observation (facial expression recognition)
"""

import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..states.density_matrix import DensityMatrix, DensityMatrixConfig


@dataclass
class ProjectiveMeasurement:
    """
    Projective (von Neumann) measurement.
    
    A projective measurement is defined by a set of orthogonal projectors
    {Πᵢ} that sum to identity: Σᵢ Πᵢ = I.
    
    When measuring state ρ:
    - Probability of outcome i: p(i) = Tr(Πᵢ ρ)
    - Post-measurement state: ρ' = Πᵢ ρ Πᵢ / p(i)
    
    For emotional states, projective measurement onto emotion basis
    represents the "collapse" from superposition to definite emotion.
    
    Example:
        >>> # Measurement in computational basis
        >>> meas = ProjectiveMeasurement.computational_basis(7)
        >>> probs = meas.probabilities(rho)
        >>> outcome, collapsed = meas.measure(rho)
    """
    
    projectors: List[torch.Tensor]
    labels: Optional[List[str]] = None
    
    def __post_init__(self):
        # Validate: projectors should sum to identity
        total = sum(self.projectors)
        dim = self.projectors[0].shape[0]
        identity = torch.eye(dim, dtype=self.projectors[0].dtype)
        
        if not torch.allclose(total, identity, atol=1e-5):
            raise ValueError("Projectors must sum to identity")
        
        if self.labels is None:
            self.labels = [f"outcome_{i}" for i in range(len(self.projectors))]
    
    @classmethod
    def computational_basis(
        cls,
        dim: int,
        labels: Optional[List[str]] = None,
        dtype: torch.dtype = torch.complex64
    ) -> "ProjectiveMeasurement":
        """
        Create measurement in computational basis.
        
        Πᵢ = |i⟩⟨i|
        
        Args:
            dim: Hilbert space dimension
            labels: Labels for outcomes
            dtype: Complex dtype
            
        Returns:
            Projective measurement
        """
        projectors = []
        for i in range(dim):
            proj = torch.zeros(dim, dim, dtype=dtype)
            proj[i, i] = 1.0
            projectors.append(proj)
        
        return cls(projectors=projectors, labels=labels)
    
    @classmethod
    def from_observable(cls, observable) -> "ProjectiveMeasurement":
        """
        Create measurement from observable's eigenbasis.
        
        Args:
            observable: Observable to diagonalize
            
        Returns:
            Projective measurement in eigenbasis
        """
        eigenvalues, eigenvectors = observable.eigendecomposition()
        
        projectors = []
        labels = []
        for i, eigenvalue in enumerate(eigenvalues):
            v = eigenvectors[:, i]
            proj = torch.outer(v, v.conj())
            projectors.append(proj)
            labels.append(f"λ={eigenvalue.item():.3f}")
        
        return cls(projectors=projectors, labels=labels)
    
    @property
    def n_outcomes(self) -> int:
        """Number of possible measurement outcomes."""
        return len(self.projectors)
    
    @property
    def dim(self) -> int:
        """Dimension of Hilbert space."""
        return self.projectors[0].shape[0]
    
    def probability(self, outcome: int, rho: torch.Tensor) -> float:
        """
        Compute probability of specific outcome.
        
        p(i) = Tr(Πᵢ ρ)
        
        Args:
            outcome: Outcome index
            rho: Density matrix
            
        Returns:
            Probability
        """
        return torch.trace(self.projectors[outcome] @ rho).real.item()
    
    def probabilities(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute all outcome probabilities.
        
        Args:
            rho: Density matrix
            
        Returns:
            Probability vector
        """
        probs = torch.tensor([self.probability(i, rho) for i in range(self.n_outcomes)])
        return probs
    
    def post_measurement_state(
        self,
        outcome: int,
        rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute post-measurement state given outcome.
        
        ρ' = Πᵢ ρ Πᵢ / Tr(Πᵢ ρ)
        
        Args:
            outcome: Measurement outcome
            rho: Pre-measurement density matrix
            
        Returns:
            Post-measurement density matrix
        """
        proj = self.projectors[outcome]
        rho_post = proj @ rho @ proj
        
        # Normalize
        trace = torch.trace(rho_post).real
        if trace < 1e-10:
            raise ValueError(f"Outcome {outcome} has zero probability")
        
        return rho_post / trace
    
    def measure(
        self,
        rho: torch.Tensor,
        seed: Optional[int] = None
    ) -> Tuple[int, torch.Tensor]:
        """
        Perform measurement, returning outcome and collapsed state.
        
        Args:
            rho: Density matrix
            seed: Random seed for reproducibility
            
        Returns:
            (outcome_index, post_measurement_state)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Sample outcome
        probs = self.probabilities(rho)
        outcome = torch.multinomial(probs, 1).item()
        
        # Collapse state
        rho_post = self.post_measurement_state(outcome, rho)
        
        return outcome, rho_post
    
    def measure_density_matrix(
        self,
        dm: DensityMatrix,
        seed: Optional[int] = None
    ) -> Tuple[int, DensityMatrix]:
        """
        Perform measurement on DensityMatrix object.
        
        Args:
            dm: DensityMatrix instance
            seed: Random seed
            
        Returns:
            (outcome_index, collapsed_density_matrix)
        """
        outcome, rho_post = self.measure(dm.matrix, seed)
        
        config = DensityMatrixConfig(
            n_dims=self.dim,
            dtype=dm.config.dtype,
            device=dm.config.device,
            tolerance=dm.config.tolerance
        )
        
        return outcome, DensityMatrix(rho_post, config, validate=False)


@dataclass
class POVMElement:
    """Single element of a POVM."""
    operator: torch.Tensor
    label: str = "element"


@dataclass
class POVM:
    """
    Positive Operator-Valued Measure (generalized measurement).
    
    A POVM is a set of positive operators {Eᵢ} that sum to identity.
    Unlike projective measurements, POVM elements need not be orthogonal
    or projectors.
    
    POVMs are useful for:
    - Optimal state discrimination
    - Unambiguous measurements
    - Weak measurements
    
    For emotional modeling, POVMs can represent:
    - Partial observation (glimpse of facial expression)
    - Ambiguous expressions that partially reveal state
    - Soft classification (probabilities without collapse)
    """
    
    elements: List[POVMElement]
    
    def __post_init__(self):
        # Validate: elements sum to identity
        total = sum(e.operator for e in self.elements)
        dim = self.elements[0].operator.shape[0]
        identity = torch.eye(dim, dtype=self.elements[0].operator.dtype)
        
        if not torch.allclose(total, identity, atol=1e-5):
            raise ValueError("POVM elements must sum to identity")
    
    @classmethod
    def from_projective(cls, measurement: ProjectiveMeasurement) -> "POVM":
        """Convert projective measurement to POVM."""
        elements = [
            POVMElement(proj, label)
            for proj, label in zip(measurement.projectors, measurement.labels)
        ]
        return cls(elements=elements)
    
    @property
    def n_outcomes(self) -> int:
        return len(self.elements)
    
    @property
    def dim(self) -> int:
        return self.elements[0].operator.shape[0]
    
    def probability(self, outcome: int, rho: torch.Tensor) -> float:
        """p(i) = Tr(Eᵢ ρ)"""
        return torch.trace(self.elements[outcome].operator @ rho).real.item()
    
    def probabilities(self, rho: torch.Tensor) -> torch.Tensor:
        """All outcome probabilities."""
        return torch.tensor([self.probability(i, rho) for i in range(self.n_outcomes)])
    
    def measure(
        self,
        rho: torch.Tensor,
        seed: Optional[int] = None
    ) -> int:
        """
        Perform POVM measurement, returning outcome.
        
        Note: POVM doesn't specify post-measurement state uniquely.
        Use Kraus operators for that.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        probs = self.probabilities(rho)
        return torch.multinomial(probs, 1).item()
