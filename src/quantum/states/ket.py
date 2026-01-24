"""
Ket (pure state) representation.

A ket |ψ⟩ represents a pure quantum state as a normalized complex vector.
This is the simplest quantum state representation, suitable for
unambiguous emotional states or genuine superpositions.

For mixed states (classical uncertainty), use DensityMatrix instead.
"""

import torch
from typing import Optional, List, Union
from dataclasses import dataclass

from .hilbert_space import HilbertSpace


@dataclass
class Ket:
    """
    Pure quantum state |ψ⟩.
    
    A ket represents a normalized complex vector in Hilbert space.
    
    Attributes:
        vector: The state vector (normalized)
        space: The Hilbert space this state lives in
    
    Example:
        >>> space = HilbertSpace.from_labels(["happy", "sad"])
        >>> psi = Ket.from_amplitudes([1, 1], space)  # |ψ⟩ = (|happy⟩ + |sad⟩)/√2
        >>> psi.probability(0)  # P(happy)
        0.5
    """
    
    vector: torch.Tensor
    space: HilbertSpace
    
    def __post_init__(self):
        # Ensure normalization
        norm = torch.norm(self.vector)
        if norm < 1e-10:
            raise ValueError("State vector cannot be zero")
        self.vector = self.vector / norm
    
    @classmethod
    def from_amplitudes(
        cls,
        amplitudes: Union[List[complex], torch.Tensor],
        space: HilbertSpace
    ) -> "Ket":
        """
        Create ket from amplitudes.
        
        Args:
            amplitudes: Complex amplitudes for each basis state
            space: Hilbert space
            
        Returns:
            Normalized ket
        """
        if isinstance(amplitudes, list):
            amplitudes = torch.tensor(amplitudes, dtype=space.dtype, device=space.device)
        return cls(vector=amplitudes.to(space.dtype).to(space.device), space=space)
    
    @classmethod
    def from_basis(cls, index: int, space: HilbertSpace) -> "Ket":
        """
        Create ket from basis state |i⟩.
        
        Args:
            index: Basis index
            space: Hilbert space
            
        Returns:
            Basis ket
        """
        return cls(vector=space.basis_vector(index), space=space)
    
    @classmethod
    def from_label(cls, label: str, space: HilbertSpace) -> "Ket":
        """
        Create ket from basis label.
        
        Args:
            label: Basis state label (e.g., "happy")
            space: Hilbert space
            
        Returns:
            Basis ket
        """
        index = space.label_to_index(label)
        return cls.from_basis(index, space)
    
    @classmethod
    def uniform_superposition(cls, space: HilbertSpace) -> "Ket":
        """Create uniform superposition over all basis states."""
        return cls(vector=space.uniform_superposition(), space=space)
    
    @classmethod
    def random(cls, space: HilbertSpace, seed: Optional[int] = None) -> "Ket":
        """Create random (Haar-distributed) pure state."""
        return cls(vector=space.random_state(seed), space=space)
    
    @property
    def dim(self) -> int:
        """Dimension of the state."""
        return self.space.dim
    
    def amplitude(self, index: int) -> complex:
        """Get amplitude for basis state |i⟩."""
        return self.vector[index].item()
    
    def probability(self, index: int) -> float:
        """Get probability of measuring basis state |i⟩."""
        return abs(self.vector[index].item()) ** 2
    
    def probabilities(self) -> torch.Tensor:
        """Get all measurement probabilities."""
        return torch.abs(self.vector) ** 2
    
    def inner_product(self, other: "Ket") -> complex:
        """Compute ⟨self|other⟩."""
        return torch.dot(self.vector.conj(), other.vector).item()
    
    def overlap(self, other: "Ket") -> float:
        """Compute |⟨self|other⟩|² (probability of measuring same state)."""
        return abs(self.inner_product(other)) ** 2
    
    def to_density_matrix(self) -> "DensityMatrix":
        """
        Convert to density matrix: ρ = |ψ⟩⟨ψ|.
        
        Returns:
            Pure state density matrix
        """
        from .density_matrix import DensityMatrix, DensityMatrixConfig
        
        config = DensityMatrixConfig(
            n_dims=self.dim,
            dtype=self.space.dtype,
            device=self.space.device
        )
        return DensityMatrix.from_pure_state(self.vector, config)
    
    def tensor_product(self, other: "Ket") -> "Ket":
        """
        Compute tensor product |ψ⟩ ⊗ |φ⟩.
        
        Args:
            other: Another ket
            
        Returns:
            Product state in combined Hilbert space
        """
        combined_vector = torch.kron(self.vector, other.vector)
        combined_space = self.space.tensor_product(other.space)
        return Ket(vector=combined_vector, space=combined_space)
    
    def apply_operator(self, operator: torch.Tensor) -> "Ket":
        """
        Apply operator to state: |ψ'⟩ = O|ψ⟩.
        
        Args:
            operator: Matrix operator
            
        Returns:
            Transformed (and renormalized) ket
        """
        new_vector = operator @ self.vector
        return Ket(vector=new_vector, space=self.space)
    
    def phase(self, phi: float) -> "Ket":
        """
        Apply global phase: |ψ⟩ → e^{iφ}|ψ⟩.
        
        Note: Global phase has no physical effect but may be
        useful for mathematical manipulations.
        """
        return Ket(
            vector=self.vector * torch.exp(1j * phi),
            space=self.space
        )
    
    def conjugate(self) -> "Ket":
        """Return complex conjugate state."""
        return Ket(vector=self.vector.conj(), space=self.space)
    
    def copy(self) -> "Ket":
        """Create a copy of this ket."""
        return Ket(vector=self.vector.clone(), space=self.space)
    
    def __repr__(self) -> str:
        # Show dominant components
        probs = self.probabilities()
        top_indices = torch.argsort(probs, descending=True)[:3]
        
        components = []
        for idx in top_indices:
            p = probs[idx].item()
            if p > 0.01:
                label = self.space.basis_labels[idx]
                components.append(f"{p:.2f}|{label}⟩")
        
        return f"Ket({' + '.join(components)})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ket):
            return False
        # States equal up to global phase
        overlap = self.overlap(other)
        return abs(overlap - 1.0) < 1e-6
    
    def __add__(self, other: "Ket") -> "Ket":
        """Add kets (result is renormalized)."""
        return Ket(vector=self.vector + other.vector, space=self.space)
    
    def __mul__(self, scalar: complex) -> "Ket":
        """Multiply by scalar (result is renormalized)."""
        return Ket(vector=scalar * self.vector, space=self.space)
    
    def __rmul__(self, scalar: complex) -> "Ket":
        """Multiply by scalar from left."""
        return self.__mul__(scalar)
