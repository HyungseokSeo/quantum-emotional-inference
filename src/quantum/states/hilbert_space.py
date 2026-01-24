"""
Hilbert space construction for emotional state representations.

A Hilbert space provides the mathematical foundation for quantum states,
equipped with an inner product that enables:
- Measuring overlap between states
- Computing probabilities
- Defining orthogonal bases

For emotional modeling, we construct finite-dimensional Hilbert spaces
where each basis vector corresponds to a discrete emotion category.
"""

import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HilbertSpace:
    """
    Finite-dimensional Hilbert space for emotional states.
    
    Attributes:
        dim: Dimension of the space
        basis_labels: Human-readable labels for basis states
        dtype: Complex dtype for vectors
        device: Computation device
    
    Example:
        >>> space = HilbertSpace.from_labels(["happy", "sad", "angry"])
        >>> space.dim
        3
        >>> space.basis_vector(0)  # |happy⟩
        tensor([1.+0.j, 0.+0.j, 0.+0.j])
    """
    
    dim: int
    basis_labels: Optional[List[str]] = None
    dtype: torch.dtype = torch.complex64
    device: str = "cpu"
    
    def __post_init__(self):
        if self.basis_labels is None:
            self.basis_labels = [f"|{i}⟩" for i in range(self.dim)]
        elif len(self.basis_labels) != self.dim:
            raise ValueError(
                f"Number of labels ({len(self.basis_labels)}) must match "
                f"dimension ({self.dim})"
            )
    
    @classmethod
    def from_labels(
        cls, 
        labels: List[str],
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu"
    ) -> "HilbertSpace":
        """
        Create Hilbert space from basis labels.
        
        Args:
            labels: Names for each basis state (e.g., emotions)
            dtype: Complex dtype
            device: Computation device
            
        Returns:
            HilbertSpace with dimension = len(labels)
        """
        return cls(
            dim=len(labels),
            basis_labels=labels,
            dtype=dtype,
            device=device
        )
    
    def basis_vector(self, index: int) -> torch.Tensor:
        """
        Get the i-th computational basis vector |i⟩.
        
        Args:
            index: Basis index
            
        Returns:
            Unit vector with 1 at position index
        """
        if not 0 <= index < self.dim:
            raise IndexError(f"Index {index} out of range [0, {self.dim})")
        
        vec = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        vec[index] = 1.0
        return vec
    
    def basis_vectors(self) -> torch.Tensor:
        """
        Get all basis vectors as columns of a matrix.
        
        Returns:
            Identity matrix (each column is a basis vector)
        """
        return torch.eye(self.dim, dtype=self.dtype, device=self.device)
    
    def label_to_index(self, label: str) -> int:
        """Convert basis label to index."""
        if label not in self.basis_labels:
            raise ValueError(f"Unknown label: {label}")
        return self.basis_labels.index(label)
    
    def index_to_label(self, index: int) -> str:
        """Convert index to basis label."""
        if not 0 <= index < self.dim:
            raise IndexError(f"Index {index} out of range [0, {self.dim})")
        return self.basis_labels[index]
    
    def inner_product(self, psi: torch.Tensor, phi: torch.Tensor) -> complex:
        """
        Compute inner product ⟨ψ|φ⟩.
        
        Args:
            psi, phi: State vectors
            
        Returns:
            Complex inner product
        """
        return torch.dot(psi.conj(), phi).item()
    
    def norm(self, psi: torch.Tensor) -> float:
        """Compute norm ||ψ|| = √⟨ψ|ψ⟩."""
        return torch.norm(psi).item()
    
    def normalize(self, psi: torch.Tensor) -> torch.Tensor:
        """Normalize state vector to unit norm."""
        n = self.norm(psi)
        if n < 1e-10:
            raise ValueError("Cannot normalize zero vector")
        return psi / n
    
    def superposition(
        self,
        amplitudes: List[complex],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Create superposition state from amplitudes.
        
        |ψ⟩ = Σᵢ αᵢ|i⟩
        
        Args:
            amplitudes: Complex amplitudes for each basis state
            normalize: Whether to normalize the result
            
        Returns:
            Superposition state vector
        """
        if len(amplitudes) != self.dim:
            raise ValueError(
                f"Number of amplitudes ({len(amplitudes)}) must match "
                f"dimension ({self.dim})"
            )
        
        psi = torch.tensor(amplitudes, dtype=self.dtype, device=self.device)
        
        if normalize:
            psi = self.normalize(psi)
        
        return psi
    
    def uniform_superposition(self) -> torch.Tensor:
        """
        Create uniform superposition over all basis states.
        
        |+⟩ = (1/√n) Σᵢ|i⟩
        """
        return torch.ones(self.dim, dtype=self.dtype, device=self.device) / (self.dim ** 0.5)
    
    def random_state(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random normalized state (Haar-distributed).
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Random unit vector
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Random complex vector
        real = torch.randn(self.dim, device=self.device)
        imag = torch.randn(self.dim, device=self.device)
        psi = (real + 1j * imag).to(self.dtype)
        
        return self.normalize(psi)
    
    def tensor_product(self, other: "HilbertSpace") -> "HilbertSpace":
        """
        Create tensor product space H₁ ⊗ H₂.
        
        Args:
            other: Another Hilbert space
            
        Returns:
            Combined Hilbert space
        """
        new_dim = self.dim * other.dim
        new_labels = [
            f"{l1}⊗{l2}" 
            for l1 in self.basis_labels 
            for l2 in other.basis_labels
        ]
        return HilbertSpace(
            dim=new_dim,
            basis_labels=new_labels,
            dtype=self.dtype,
            device=self.device
        )
    
    def __repr__(self) -> str:
        return f"HilbertSpace(dim={self.dim}, labels={self.basis_labels})"
