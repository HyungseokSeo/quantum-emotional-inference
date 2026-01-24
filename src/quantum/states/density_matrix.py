"""
Density matrix representation for quantum emotional states.

The density matrix formalism generalizes pure quantum states to handle:
- Pure states: genuine superposition (coherent)
- Mixed states: classical uncertainty (incoherent)

This distinction is crucial for emotional modeling:
- Pure superposition = felt ambivalence (experiencing multiple emotions simultaneously)
- Mixed state = uncertainty about which emotion one feels

References:
    Nielsen & Chuang, "Quantum Computation and Quantum Information", Ch. 2.4
    Busemeyer & Bruza, "Quantum Models of Cognition and Decision", Ch. 3
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from enum import Enum

from ...core.types import StateType
from ...core.utils import (
    to_complex, normalize_vector, is_hermitian, 
    is_positive_semidefinite, is_unit_trace, matrix_log
)


@dataclass
class DensityMatrixConfig:
    """Configuration for density matrix representation."""
    n_dims: int
    dtype: torch.dtype = torch.complex64
    device: str = "cpu"
    tolerance: float = 1e-6


class DensityMatrix:
    """
    Quantum density matrix representation for emotional states.
    
    Properties guaranteed:
        - Hermiticity: ρ = ρ†
        - Positive semi-definiteness: ρ ≥ 0
        - Unit trace: Tr(ρ) = 1
    
    The density matrix ρ encodes:
        - Diagonal elements ρᵢᵢ: probabilities of each basis state
        - Off-diagonal elements ρᵢⱼ (i≠j): coherences (superposition)
    
    For emotional states:
        - High off-diagonal magnitude → genuine ambivalence
        - Zero off-diagonals → classical uncertainty
    
    Example:
        >>> config = DensityMatrixConfig(n_dims=2)
        >>> # Pure superposition: |ψ⟩ = (|happy⟩ + |sad⟩)/√2
        >>> psi = torch.tensor([1.0, 1.0]) / np.sqrt(2)
        >>> rho_pure = DensityMatrix.from_pure_state(psi, config)
        >>> rho_pure.l1_coherence()  # Non-zero: ambivalence
        0.5
        >>> 
        >>> # Mixed state: 50% happy, 50% sad (uncertainty)
        >>> rho_mixed = DensityMatrix.from_probabilities(torch.tensor([0.5, 0.5]), config)
        >>> rho_mixed.l1_coherence()  # Zero: no ambivalence
        0.0
    """
    
    def __init__(
        self,
        matrix: torch.Tensor,
        config: DensityMatrixConfig,
        validate: bool = True
    ):
        """
        Initialize density matrix.
        
        Args:
            matrix: The density matrix tensor
            config: Configuration parameters
            validate: Whether to validate physical constraints
        """
        self.config = config
        self._matrix = matrix.to(config.dtype).to(config.device)
        
        if validate:
            self._validate()
    
    def _validate(self) -> None:
        """Ensure physical validity of density matrix."""
        tol = self.config.tolerance
        
        # Check Hermiticity: ρ = ρ†
        if not is_hermitian(self._matrix, tol):
            raise ValueError("Density matrix must be Hermitian (ρ = ρ†)")
        
        # Check unit trace: Tr(ρ) = 1
        if not is_unit_trace(self._matrix, tol):
            trace = torch.trace(self._matrix).real.item()
            raise ValueError(f"Density matrix must have unit trace, got Tr(ρ) = {trace}")
        
        # Check positive semi-definiteness: ρ ≥ 0
        if not is_positive_semidefinite(self._matrix, tol):
            eigenvalues = torch.linalg.eigvalsh(self._matrix)
            raise ValueError(
                f"Density matrix must be positive semi-definite. "
                f"Eigenvalues: {eigenvalues.tolist()}"
            )
    
    @property
    def matrix(self) -> torch.Tensor:
        """The underlying matrix tensor."""
        return self._matrix
    
    @property
    def n_dims(self) -> int:
        """Dimension of the Hilbert space."""
        return self._matrix.shape[0]
    
    # ==================== Constructors ====================
    
    @classmethod
    def from_pure_state(
        cls, 
        psi: torch.Tensor, 
        config: DensityMatrixConfig
    ) -> DensityMatrix:
        """
        Create density matrix from pure state: ρ = |ψ⟩⟨ψ|.
        
        Args:
            psi: State vector (will be normalized)
            config: Configuration
            
        Returns:
            Pure state density matrix
        """
        psi = to_complex(psi)
        psi = normalize_vector(psi)
        rho = torch.outer(psi, psi.conj())
        return cls(rho, config)
    
    @classmethod
    def from_probabilities(
        cls,
        probabilities: torch.Tensor,
        config: DensityMatrixConfig
    ) -> DensityMatrix:
        """
        Create diagonal density matrix from probabilities.
        
        This represents classical uncertainty (no coherence).
        
        Args:
            probabilities: Probability distribution (must sum to 1)
            config: Configuration
            
        Returns:
            Mixed state density matrix (diagonal)
        """
        probabilities = probabilities / probabilities.sum()  # Normalize
        rho = torch.diag(probabilities.to(config.dtype))
        return cls(rho, config)
    
    @classmethod
    def from_ensemble(
        cls,
        states: List[torch.Tensor],
        probabilities: torch.Tensor,
        config: DensityMatrixConfig
    ) -> DensityMatrix:
        """
        Create density matrix from statistical ensemble.
        
        ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
        
        Args:
            states: List of state vectors
            probabilities: Probability weights
            config: Configuration
            
        Returns:
            Mixed state density matrix
        """
        probabilities = probabilities / probabilities.sum()
        
        rho = torch.zeros(config.n_dims, config.n_dims, dtype=config.dtype)
        for p, psi in zip(probabilities, states):
            psi = to_complex(psi)
            psi = normalize_vector(psi)
            rho += p * torch.outer(psi, psi.conj())
        
        return cls(rho, config)
    
    @classmethod
    def maximally_mixed(cls, config: DensityMatrixConfig) -> DensityMatrix:
        """
        Create maximally mixed state: ρ = I/n.
        
        This represents complete ignorance about the state.
        """
        rho = torch.eye(config.n_dims, dtype=config.dtype) / config.n_dims
        return cls(rho, config)
    
    @classmethod
    def from_bloch_vector(
        cls,
        r: torch.Tensor,
        config: Optional[DensityMatrixConfig] = None
    ) -> DensityMatrix:
        """
        Create qubit state from Bloch vector.
        
        ρ = (I + r·σ) / 2
        
        where σ = (σₓ, σᵧ, σᵤ) are Pauli matrices.
        
        Args:
            r: Bloch vector (3D, |r| ≤ 1)
            config: Configuration (defaults to 2D)
            
        Returns:
            Qubit density matrix
        """
        if config is None:
            config = DensityMatrixConfig(n_dims=2)
        
        if config.n_dims != 2:
            raise ValueError("Bloch representation only for 2-level systems")
        
        # Pauli matrices
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=config.dtype)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=config.dtype)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=config.dtype)
        
        I = torch.eye(2, dtype=config.dtype)
        rho = (I + r[0]*sigma_x + r[1]*sigma_y + r[2]*sigma_z) / 2
        
        return cls(rho, config)
    
    # ==================== State Properties ====================
    
    def purity(self) -> float:
        """
        Compute purity: Tr(ρ²).
        
        - Pure state: Tr(ρ²) = 1
        - Maximally mixed: Tr(ρ²) = 1/n
        - Mixed: 1/n < Tr(ρ²) < 1
        """
        return torch.trace(self._matrix @ self._matrix).real.item()
    
    def von_neumann_entropy(self) -> float:
        """
        Compute von Neumann entropy: S(ρ) = -Tr(ρ ln ρ).
        
        - Pure state: S = 0
        - Maximally mixed: S = ln(n)
        
        For emotional states, entropy measures the "spread" of the
        probability distribution over emotions.
        """
        eigenvalues = torch.linalg.eigvalsh(self._matrix).real
        eigenvalues = eigenvalues[eigenvalues > self.config.tolerance]
        return -torch.sum(eigenvalues * torch.log(eigenvalues)).item()
    
    def linear_entropy(self) -> float:
        """
        Compute linear entropy: S_L(ρ) = 1 - Tr(ρ²).
        
        Computationally simpler approximation to von Neumann entropy.
        """
        return 1 - self.purity()
    
    def state_type(self) -> StateType:
        """Classify state as pure, mixed, or maximally mixed."""
        p = self.purity()
        tol = self.config.tolerance
        
        if abs(p - 1.0) < tol:
            return StateType.PURE
        elif abs(p - 1.0/self.n_dims) < tol:
            return StateType.MAXIMALLY_MIXED
        return StateType.MIXED
    
    # ==================== Coherence Measures ====================
    
    def l1_coherence(self, basis: Optional[torch.Tensor] = None) -> float:
        """
        Compute l1-norm of coherence: C_l1(ρ) = Σᵢ≠ⱼ |ρᵢⱼ|.
        
        This measures the total "quantumness" or superposition magnitude.
        For emotional states:
            - C > 0: genuine ambivalence (feeling multiple emotions)
            - C = 0: classical uncertainty (unsure which emotion)
        
        Args:
            basis: Optional basis transformation matrix
            
        Returns:
            l1 coherence value
        """
        if basis is not None:
            rho = basis.conj().T @ self._matrix @ basis
        else:
            rho = self._matrix
        
        # Extract off-diagonal elements
        off_diagonal = rho - torch.diag(torch.diag(rho))
        return torch.sum(torch.abs(off_diagonal)).item()
    
    def relative_entropy_of_coherence(self) -> float:
        """
        Compute relative entropy of coherence: C_RE(ρ) = S(ρ_diag) - S(ρ).
        
        Information-theoretic measure of coherence.
        """
        # Diagonal (dephased) version
        diagonal = torch.diag(torch.diag(self._matrix))
        
        # Entropy of diagonal state
        diag_eigenvalues = torch.diag(diagonal).real
        diag_eigenvalues = diag_eigenvalues[diag_eigenvalues > self.config.tolerance]
        S_diag = -torch.sum(diag_eigenvalues * torch.log(diag_eigenvalues)).item()
        
        # Entropy of original state
        S_rho = self.von_neumann_entropy()
        
        return S_diag - S_rho
    
    def coherence_matrix(self) -> torch.Tensor:
        """Return the off-diagonal (coherence) part of the density matrix."""
        return self._matrix - torch.diag(torch.diag(self._matrix))
    
    # ==================== Operations ====================
    
    def expectation(self, observable: torch.Tensor) -> float:
        """
        Compute expectation value: ⟨O⟩ = Tr(ρO).
        
        Args:
            observable: Hermitian operator
            
        Returns:
            Real expectation value
        """
        return torch.trace(self._matrix @ observable).real.item()
    
    def probabilities(self) -> torch.Tensor:
        """Return diagonal elements (measurement probabilities)."""
        return torch.diag(self._matrix).real
    
    def measure(self, basis_index: int) -> DensityMatrix:
        """
        Apply projective measurement onto basis state.
        
        Collapses the state: ρ → |i⟩⟨i|
        
        Args:
            basis_index: Index of basis state to project onto
            
        Returns:
            Collapsed (pure) density matrix
        """
        result = torch.zeros_like(self._matrix)
        result[basis_index, basis_index] = 1.0
        return DensityMatrix(result, self.config, validate=False)
    
    def partial_trace(self, dims: Tuple[int, int], trace_out: int) -> DensityMatrix:
        """
        Compute partial trace for bipartite systems.
        
        Args:
            dims: Dimensions of subsystems (d1, d2)
            trace_out: Which subsystem to trace out (0 or 1)
            
        Returns:
            Reduced density matrix
        """
        from ...core.utils import partial_trace as pt
        
        reduced = pt(self._matrix, dims, trace_out)
        new_dim = dims[1] if trace_out == 0 else dims[0]
        new_config = DensityMatrixConfig(
            n_dims=new_dim,
            dtype=self.config.dtype,
            device=self.config.device,
            tolerance=self.config.tolerance
        )
        return DensityMatrix(reduced, new_config, validate=False)
    
    # ==================== Bloch Representation ====================
    
    def to_bloch_vector(self) -> torch.Tensor:
        """
        Extract Bloch vector for 2-level systems.
        
        Returns:
            3D Bloch vector r where ρ = (I + r·σ)/2
        """
        if self.n_dims != 2:
            raise ValueError("Bloch representation only for 2-level systems")
        
        # r = Tr(ρ σ) for each Pauli matrix
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=self.config.dtype)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=self.config.dtype)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=self.config.dtype)
        
        rx = torch.trace(self._matrix @ sigma_x).real
        ry = torch.trace(self._matrix @ sigma_y).real
        rz = torch.trace(self._matrix @ sigma_z).real
        
        return torch.tensor([rx, ry, rz])
    
    # ==================== Utility ====================
    
    def copy(self) -> DensityMatrix:
        """Create a copy of this density matrix."""
        return DensityMatrix(self._matrix.clone(), self.config, validate=False)
    
    def to(self, device: str) -> DensityMatrix:
        """Move to specified device."""
        new_config = DensityMatrixConfig(
            n_dims=self.config.n_dims,
            dtype=self.config.dtype,
            device=device,
            tolerance=self.config.tolerance
        )
        return DensityMatrix(self._matrix.to(device), new_config, validate=False)
    
    def __repr__(self) -> str:
        return (
            f"DensityMatrix(n={self.n_dims}, "
            f"purity={self.purity():.4f}, "
            f"S={self.von_neumann_entropy():.4f}, "
            f"C={self.l1_coherence():.4f})"
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DensityMatrix):
            return False
        return torch.allclose(
            self._matrix, other._matrix, 
            atol=self.config.tolerance
        )
