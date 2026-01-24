"""
Quantum observables (Hermitian operators).

In quantum mechanics, observables are represented by Hermitian operators.
The eigenvalues are possible measurement outcomes, and eigenvectors are
the states that yield definite outcomes.

For emotional modeling, observables can represent:
- Emotion category (which emotion is expressed)
- Valence (positive/negative affect)
- Arousal (activation level)
"""

import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass

from ...core.utils import is_hermitian


@dataclass
class Observable:
    """
    Quantum observable (Hermitian operator).
    
    Attributes:
        matrix: Hermitian matrix representation
        name: Human-readable name
        eigenvalues: Cached eigenvalues
        eigenvectors: Cached eigenvectors
    
    Example:
        >>> # Pauli Z (spin measurement)
        >>> sigma_z = Observable.pauli_z()
        >>> sigma_z.eigenvalues
        tensor([-1.,  1.])
    """
    
    matrix: torch.Tensor
    name: str = "Observable"
    _eigenvalues: Optional[torch.Tensor] = None
    _eigenvectors: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        if not is_hermitian(self.matrix):
            raise ValueError(f"Observable {self.name} must be Hermitian")
    
    @classmethod
    def from_eigendecomposition(
        cls,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        name: str = "Observable"
    ) -> "Observable":
        """
        Construct observable from eigendecomposition.
        
        O = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|
        
        Args:
            eigenvalues: Real eigenvalues
            eigenvectors: Orthonormal eigenvectors (columns)
            name: Observable name
            
        Returns:
            Observable instance
        """
        # Reconstruct matrix: O = V Λ V†
        matrix = eigenvectors @ torch.diag(eigenvalues.to(eigenvectors.dtype)) @ eigenvectors.conj().T
        
        obs = cls(matrix=matrix, name=name)
        obs._eigenvalues = eigenvalues
        obs._eigenvectors = eigenvectors
        return obs
    
    @classmethod
    def identity(cls, dim: int, dtype: torch.dtype = torch.complex64) -> "Observable":
        """Create identity operator."""
        return cls(
            matrix=torch.eye(dim, dtype=dtype),
            name="Identity"
        )
    
    @classmethod
    def pauli_x(cls) -> "Observable":
        """Pauli X (bit flip)."""
        return cls(
            matrix=torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
            name="Pauli-X"
        )
    
    @classmethod
    def pauli_y(cls) -> "Observable":
        """Pauli Y."""
        return cls(
            matrix=torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
            name="Pauli-Y"
        )
    
    @classmethod
    def pauli_z(cls) -> "Observable":
        """Pauli Z (phase flip / measurement)."""
        return cls(
            matrix=torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
            name="Pauli-Z"
        )
    
    @classmethod
    def number_operator(cls, dim: int) -> "Observable":
        """Number operator N = diag(0, 1, 2, ..., n-1)."""
        return cls(
            matrix=torch.diag(torch.arange(dim, dtype=torch.complex64)),
            name="Number"
        )
    
    @property
    def dim(self) -> int:
        """Dimension of the operator."""
        return self.matrix.shape[0]
    
    @property
    def eigenvalues(self) -> torch.Tensor:
        """Get eigenvalues (cached)."""
        if self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors = torch.linalg.eigh(self.matrix)
        return self._eigenvalues
    
    @property
    def eigenvectors(self) -> torch.Tensor:
        """Get eigenvectors as columns (cached)."""
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = torch.linalg.eigh(self.matrix)
        return self._eigenvectors
    
    def eigendecomposition(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (eigenvalues, eigenvectors)."""
        return self.eigenvalues, self.eigenvectors
    
    def expectation(self, rho: torch.Tensor) -> float:
        """
        Compute expectation value: ⟨O⟩ = Tr(ρO).
        
        Args:
            rho: Density matrix
            
        Returns:
            Real expectation value
        """
        return torch.trace(rho @ self.matrix).real.item()
    
    def variance(self, rho: torch.Tensor) -> float:
        """
        Compute variance: Var(O) = ⟨O²⟩ - ⟨O⟩².
        
        Args:
            rho: Density matrix
            
        Returns:
            Variance (always non-negative)
        """
        exp_O = self.expectation(rho)
        exp_O2 = torch.trace(rho @ self.matrix @ self.matrix).real.item()
        return exp_O2 - exp_O ** 2
    
    def uncertainty(self, rho: torch.Tensor) -> float:
        """
        Compute standard deviation: ΔO = √Var(O).
        
        This quantifies the spread of measurement outcomes.
        """
        return self.variance(rho) ** 0.5
    
    def commutator(self, other: "Observable") -> torch.Tensor:
        """
        Compute commutator: [A, B] = AB - BA.
        
        Non-zero commutator indicates incompatible observables
        (cannot be measured simultaneously with certainty).
        """
        return self.matrix @ other.matrix - other.matrix @ self.matrix
    
    def commutes_with(self, other: "Observable", tol: float = 1e-6) -> bool:
        """Check if observables commute: [A, B] = 0."""
        comm = self.commutator(other)
        return torch.allclose(comm, torch.zeros_like(comm), atol=tol)
    
    def anticommutator(self, other: "Observable") -> torch.Tensor:
        """Compute anticommutator: {A, B} = AB + BA."""
        return self.matrix @ other.matrix + other.matrix @ self.matrix
    
    def projector(self, eigenvalue_index: int) -> torch.Tensor:
        """
        Get projector onto eigenspace.
        
        Πᵢ = |vᵢ⟩⟨vᵢ|
        
        Args:
            eigenvalue_index: Index of eigenvalue
            
        Returns:
            Projection operator
        """
        v = self.eigenvectors[:, eigenvalue_index]
        return torch.outer(v, v.conj())
    
    def spectral_decomposition(self) -> List[Tuple[float, torch.Tensor]]:
        """
        Return spectral decomposition: O = Σᵢ λᵢ Πᵢ.
        
        Returns:
            List of (eigenvalue, projector) pairs
        """
        decomp = []
        for i, eigenvalue in enumerate(self.eigenvalues):
            projector = self.projector(i)
            decomp.append((eigenvalue.item(), projector))
        return decomp
    
    def function(self, f) -> "Observable":
        """
        Apply function to observable: f(O) = Σᵢ f(λᵢ) Πᵢ.
        
        Args:
            f: Scalar function to apply to eigenvalues
            
        Returns:
            New observable f(O)
        """
        new_eigenvalues = torch.tensor([f(e.item()) for e in self.eigenvalues])
        return Observable.from_eigendecomposition(
            new_eigenvalues,
            self.eigenvectors,
            name=f"f({self.name})"
        )
    
    def __add__(self, other: "Observable") -> "Observable":
        """Add observables."""
        return Observable(
            matrix=self.matrix + other.matrix,
            name=f"({self.name}+{other.name})"
        )
    
    def __sub__(self, other: "Observable") -> "Observable":
        """Subtract observables."""
        return Observable(
            matrix=self.matrix - other.matrix,
            name=f"({self.name}-{other.name})"
        )
    
    def __mul__(self, scalar: float) -> "Observable":
        """Multiply by scalar."""
        return Observable(
            matrix=scalar * self.matrix,
            name=f"{scalar}*{self.name}"
        )
    
    def __rmul__(self, scalar: float) -> "Observable":
        """Multiply by scalar from left."""
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        return f"Observable({self.name}, dim={self.dim})"
