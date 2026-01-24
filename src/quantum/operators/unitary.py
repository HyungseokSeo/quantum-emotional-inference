"""
Unitary operators for coherent quantum evolution.

Unitary operators U satisfy U†U = UU† = I and preserve:
- Normalization of states
- Inner products
- Purity of states

For emotional modeling, unitary evolution represents:
- Internal emotional dynamics (before measurement/expression)
- Coherent transitions between emotional states
- Time evolution of emotional superpositions
"""

import torch
from typing import Optional
from dataclasses import dataclass
import math


@dataclass
class UnitaryOperator:
    """
    Unitary operator for coherent state evolution.
    
    Attributes:
        matrix: Unitary matrix (U†U = I)
        name: Human-readable name
    
    Example:
        >>> # Hadamard gate (creates superposition)
        >>> H = UnitaryOperator.hadamard()
        >>> # Apply to state
        >>> psi_new = H @ psi
    """
    
    matrix: torch.Tensor
    name: str = "Unitary"
    
    def __post_init__(self):
        # Validate unitarity
        identity = torch.eye(self.dim, dtype=self.matrix.dtype, device=self.matrix.device)
        product = self.matrix.conj().T @ self.matrix
        
        if not torch.allclose(product, identity, atol=1e-5):
            raise ValueError(f"Operator {self.name} is not unitary")
    
    @classmethod
    def identity(cls, dim: int, dtype: torch.dtype = torch.complex64) -> "UnitaryOperator":
        """Identity operator."""
        return cls(
            matrix=torch.eye(dim, dtype=dtype),
            name="I"
        )
    
    @classmethod
    def hadamard(cls) -> "UnitaryOperator":
        """
        Hadamard gate (2D).
        
        H = (1/√2) [[1, 1], [1, -1]]
        
        Creates equal superposition from basis states.
        """
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)
        return cls(matrix=H, name="Hadamard")
    
    @classmethod
    def phase(cls, phi: float, dim: int = 2) -> "UnitaryOperator":
        """
        Phase gate.
        
        For qubits: diag(1, e^{iφ})
        """
        phases = torch.zeros(dim, dtype=torch.complex64)
        phases[0] = 1.0
        for i in range(1, dim):
            phases[i] = torch.exp(1j * phi * i)
        return cls(matrix=torch.diag(phases), name=f"Phase({phi:.3f})")
    
    @classmethod
    def rotation_x(cls, theta: float) -> "UnitaryOperator":
        """
        Rotation around X axis (qubit).
        
        Rx(θ) = exp(-iθσx/2)
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        matrix = torch.tensor([
            [c, -1j * s],
            [-1j * s, c]
        ], dtype=torch.complex64)
        return cls(matrix=matrix, name=f"Rx({theta:.3f})")
    
    @classmethod
    def rotation_y(cls, theta: float) -> "UnitaryOperator":
        """
        Rotation around Y axis (qubit).
        
        Ry(θ) = exp(-iθσy/2)
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        matrix = torch.tensor([
            [c, -s],
            [s, c]
        ], dtype=torch.complex64)
        return cls(matrix=matrix, name=f"Ry({theta:.3f})")
    
    @classmethod
    def rotation_z(cls, theta: float) -> "UnitaryOperator":
        """
        Rotation around Z axis (qubit).
        
        Rz(θ) = exp(-iθσz/2)
        """
        matrix = torch.tensor([
            [torch.exp(-1j * theta / 2), 0],
            [0, torch.exp(1j * theta / 2)]
        ], dtype=torch.complex64)
        return cls(matrix=matrix, name=f"Rz({theta:.3f})")
    
    @classmethod
    def from_generator(
        cls,
        generator: torch.Tensor,
        t: float = 1.0,
        name: str = "exp(-iHt)"
    ) -> "UnitaryOperator":
        """
        Create unitary from Hermitian generator.
        
        U = exp(-iHt)
        
        Args:
            generator: Hermitian matrix H
            t: Time parameter
            name: Operator name
            
        Returns:
            Unitary operator
        """
        matrix = torch.linalg.matrix_exp(-1j * generator * t)
        return cls(matrix=matrix, name=name)
    
    @classmethod
    def random(cls, dim: int, seed: Optional[int] = None) -> "UnitaryOperator":
        """
        Generate random unitary (Haar-distributed).
        
        Uses QR decomposition of random complex matrix.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Random complex matrix
        real = torch.randn(dim, dim)
        imag = torch.randn(dim, dim)
        A = (real + 1j * imag).to(torch.complex64)
        
        # QR decomposition
        Q, R = torch.linalg.qr(A)
        
        # Make unique by fixing phase
        diag_R = torch.diag(R)
        phases = diag_R / torch.abs(diag_R)
        Q = Q @ torch.diag(phases.conj())
        
        return cls(matrix=Q, name="Random")
    
    @property
    def dim(self) -> int:
        """Dimension of operator."""
        return self.matrix.shape[0]
    
    def apply_to_ket(self, psi: torch.Tensor) -> torch.Tensor:
        """Apply to state vector: |ψ'⟩ = U|ψ⟩."""
        return self.matrix @ psi
    
    def apply_to_density_matrix(self, rho: torch.Tensor) -> torch.Tensor:
        """Apply to density matrix: ρ' = UρU†."""
        return self.matrix @ rho @ self.matrix.conj().T
    
    def inverse(self) -> "UnitaryOperator":
        """Return inverse (adjoint) operator: U† = U⁻¹."""
        return UnitaryOperator(
            matrix=self.matrix.conj().T,
            name=f"{self.name}†"
        )
    
    def power(self, n: int) -> "UnitaryOperator":
        """Return U^n."""
        result = torch.eye(self.dim, dtype=self.matrix.dtype)
        for _ in range(abs(n)):
            result = result @ self.matrix
        if n < 0:
            result = result.conj().T
        return UnitaryOperator(matrix=result, name=f"{self.name}^{n}")
    
    def tensor_product(self, other: "UnitaryOperator") -> "UnitaryOperator":
        """Tensor product U ⊗ V."""
        return UnitaryOperator(
            matrix=torch.kron(self.matrix, other.matrix),
            name=f"{self.name}⊗{other.name}"
        )
    
    def __matmul__(self, other):
        """Matrix multiplication U @ V or U @ |ψ⟩."""
        if isinstance(other, UnitaryOperator):
            return UnitaryOperator(
                matrix=self.matrix @ other.matrix,
                name=f"{self.name}·{other.name}"
            )
        elif isinstance(other, torch.Tensor):
            if other.dim() == 1:
                return self.apply_to_ket(other)
            else:
                return self.apply_to_density_matrix(other)
        else:
            raise TypeError(f"Cannot multiply UnitaryOperator with {type(other)}")
    
    def __repr__(self) -> str:
        return f"UnitaryOperator({self.name}, dim={self.dim})"
