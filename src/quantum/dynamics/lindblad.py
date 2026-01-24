"""
Lindblad master equation for open quantum system dynamics.

The Lindblad equation describes the evolution of a quantum system
interacting with its environment:

    dρ/dt = -i[H, ρ] + Σₖ γₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})

Components:
- Hamiltonian H: drives coherent evolution
- Lindblad operators Lₖ: model environmental coupling
- Rates γₖ: strength of each dissipation channel

For emotional modeling, Lindblad dynamics represents:
- Internal emotional dynamics (H)
- "Measurement" by environment causing decoherence (Lₖ)
- The gradual collapse of emotional superposition over time

This is central to the hierarchical decoherence cascade:
- Reactive level: weak Lindblad terms, high coherence preserved
- Adaptive level: stronger terms, partial decoherence  
- Reflective level: strong terms, near-complete collapse
"""

import torch
from typing import List, Optional, Callable
from dataclasses import dataclass

from ..states.density_matrix import DensityMatrix, DensityMatrixConfig


@dataclass
class LindbladOperator:
    """
    Single Lindblad (jump) operator with associated rate.
    
    Represents one channel of environmental coupling.
    
    Attributes:
        operator: The Lindblad operator L
        rate: Coupling strength γ
        name: Description of this channel
    """
    operator: torch.Tensor
    rate: float
    name: str = "jump"
    
    def __post_init__(self):
        # Precompute L†L for efficiency
        self.L_dag = self.operator.conj().T
        self.L_dag_L = self.L_dag @ self.operator


class LindbladDynamics:
    """
    Lindblad master equation solver.
    
    Solves: dρ/dt = -i[H, ρ] + Σₖ γₖ D[Lₖ](ρ)
    
    where D[L](ρ) = L ρ L† - ½{L†L, ρ} is the dissipator.
    
    Example:
        >>> # Pure dephasing dynamics
        >>> H = torch.zeros(2, 2, dtype=torch.complex64)
        >>> L = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)  # σz
        >>> dynamics = LindbladDynamics(
        ...     hamiltonian=H,
        ...     lindblad_operators=[LindbladOperator(L, rate=1.0, name="dephasing")]
        ... )
        >>> trajectory = dynamics.evolve(rho_init, t_span)
    """
    
    def __init__(
        self,
        hamiltonian: torch.Tensor,
        lindblad_operators: List[LindbladOperator],
        config: Optional[DensityMatrixConfig] = None
    ):
        """
        Initialize Lindblad dynamics.
        
        Args:
            hamiltonian: Hermitian Hamiltonian matrix
            lindblad_operators: List of Lindblad operators with rates
            config: Configuration for density matrices
        """
        self.H = hamiltonian
        self.lindblad_ops = lindblad_operators
        self.config = config or DensityMatrixConfig(n_dims=hamiltonian.shape[0])
        
        # Validate Hamiltonian is Hermitian
        if not torch.allclose(self.H, self.H.conj().T, atol=1e-6):
            raise ValueError("Hamiltonian must be Hermitian")
    
    @classmethod
    def dephasing(
        cls,
        n_dims: int,
        rate: float,
        hamiltonian: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.complex64
    ) -> "LindbladDynamics":
        """
        Create pure dephasing dynamics.
        
        Dephasing destroys off-diagonal coherences without affecting
        populations. This models "which-path" information leaking to
        environment.
        
        Args:
            n_dims: Hilbert space dimension
            rate: Dephasing rate
            hamiltonian: Optional Hamiltonian (default: zero)
            dtype: Complex dtype
            
        Returns:
            LindbladDynamics for dephasing
        """
        if hamiltonian is None:
            hamiltonian = torch.zeros(n_dims, n_dims, dtype=dtype)
        
        # Dephasing operators: projectors |i⟩⟨i| for each basis state
        lindblad_ops = []
        for i in range(n_dims):
            proj = torch.zeros(n_dims, n_dims, dtype=dtype)
            proj[i, i] = 1.0
            lindblad_ops.append(LindbladOperator(proj, rate, f"dephase_{i}"))
        
        return cls(hamiltonian, lindblad_ops)
    
    @classmethod
    def amplitude_damping(
        cls,
        n_dims: int,
        rates: List[float],
        hamiltonian: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.complex64
    ) -> "LindbladDynamics":
        """
        Create amplitude damping (spontaneous emission) dynamics.
        
        Models decay from excited states to ground state.
        
        Args:
            n_dims: Hilbert space dimension
            rates: Decay rates for each level → ground
            hamiltonian: Optional Hamiltonian
            dtype: Complex dtype
            
        Returns:
            LindbladDynamics for amplitude damping
        """
        if hamiltonian is None:
            hamiltonian = torch.zeros(n_dims, n_dims, dtype=dtype)
        
        # Lowering operators: |0⟩⟨i| for each excited state
        lindblad_ops = []
        for i in range(1, n_dims):
            lower = torch.zeros(n_dims, n_dims, dtype=dtype)
            lower[0, i] = 1.0  # |ground⟩⟨excited_i|
            rate = rates[i-1] if i-1 < len(rates) else rates[-1]
            lindblad_ops.append(LindbladOperator(lower, rate, f"decay_{i}"))
        
        return cls(hamiltonian, lindblad_ops)
    
    def coherent_term(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute coherent evolution: -i[H, ρ].
        
        This is the standard Schrödinger-like term that preserves purity.
        """
        commutator = self.H @ rho - rho @ self.H
        return -1j * commutator
    
    def dissipator(self, L: LindbladOperator, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute single dissipator: γ(L ρ L† - ½{L†L, ρ}).
        
        This term drives decoherence and thermalization.
        """
        # L ρ L†
        sandwich = L.operator @ rho @ L.L_dag
        
        # ½{L†L, ρ} = ½(L†L ρ + ρ L†L)
        anticommutator = 0.5 * (L.L_dag_L @ rho + rho @ L.L_dag_L)
        
        return L.rate * (sandwich - anticommutator)
    
    def dissipative_term(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute total dissipative term: Σₖ D[Lₖ](ρ)."""
        result = torch.zeros_like(rho)
        for L in self.lindblad_ops:
            result += self.dissipator(L, rho)
        return result
    
    def drho_dt(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Full Lindblad equation right-hand side.
        
        dρ/dt = -i[H, ρ] + Σₖ γₖ D[Lₖ](ρ)
        """
        return self.coherent_term(rho) + self.dissipative_term(rho)
    
    def step_euler(self, rho: torch.Tensor, dt: float) -> torch.Tensor:
        """Single Euler integration step."""
        return rho + dt * self.drho_dt(rho)
    
    def step_rk4(self, rho: torch.Tensor, dt: float) -> torch.Tensor:
        """Single 4th-order Runge-Kutta step."""
        k1 = self.drho_dt(rho)
        k2 = self.drho_dt(rho + 0.5 * dt * k1)
        k3 = self.drho_dt(rho + 0.5 * dt * k2)
        k4 = self.drho_dt(rho + dt * k3)
        return rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def enforce_physicality(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Project density matrix back to valid state if needed.
        
        Numerical errors can violate positivity or trace normalization.
        This repairs such violations.
        """
        # Enforce Hermiticity
        rho = 0.5 * (rho + rho.conj().T)
        
        # Enforce positivity via eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(rho)
        eigenvalues = torch.clamp(eigenvalues.real, min=0)
        
        # Renormalize to unit trace
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        # Reconstruct
        rho = eigenvectors @ torch.diag(eigenvalues.to(rho.dtype)) @ eigenvectors.conj().T
        
        return rho
    
    def evolve(
        self,
        rho_init: DensityMatrix,
        t_span: torch.Tensor,
        method: str = "rk4",
        enforce_physical: bool = True
    ) -> List[DensityMatrix]:
        """
        Evolve density matrix over time.
        
        Args:
            rho_init: Initial density matrix
            t_span: Time points for evolution
            method: Integration method ("euler" or "rk4")
            enforce_physical: Whether to project to physical states
            
        Returns:
            List of density matrices at each time point
        """
        trajectory = [rho_init]
        rho = rho_init.matrix.clone()
        
        for i in range(1, len(t_span)):
            dt = (t_span[i] - t_span[i-1]).item()
            
            if method == "euler":
                rho = self.step_euler(rho, dt)
            elif method == "rk4":
                rho = self.step_rk4(rho, dt)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if enforce_physical:
                rho = self.enforce_physicality(rho)
            
            trajectory.append(DensityMatrix(rho.clone(), self.config, validate=False))
        
        return trajectory
    
    def steady_state(
        self,
        max_time: float = 100.0,
        dt: float = 0.1,
        tol: float = 1e-6
    ) -> DensityMatrix:
        """
        Find steady state by evolving until convergence.
        
        Args:
            max_time: Maximum evolution time
            dt: Time step
            tol: Convergence tolerance
            
        Returns:
            Steady state density matrix
        """
        # Start from maximally mixed state
        rho = torch.eye(self.config.n_dims, dtype=self.config.dtype) / self.config.n_dims
        
        t = 0.0
        while t < max_time:
            rho_new = self.step_rk4(rho, dt)
            rho_new = self.enforce_physicality(rho_new)
            
            # Check convergence
            diff = torch.norm(rho_new - rho).item()
            if diff < tol:
                break
            
            rho = rho_new
            t += dt
        
        return DensityMatrix(rho, self.config, validate=False)
    
    def coherence_decay_rate(self) -> float:
        """
        Estimate characteristic decoherence rate.
        
        For simple dephasing, this is the sum of Lindblad rates.
        """
        return sum(op.rate for op in self.lindblad_ops)
    
    def decoherence_time(self) -> float:
        """Characteristic decoherence time T2 ≈ 1/γ."""
        rate = self.coherence_decay_rate()
        return 1.0 / rate if rate > 0 else float('inf')


class EmotionalLindbladDynamics(LindbladDynamics):
    """
    Lindblad dynamics specialized for emotional decoherence.
    
    Models how emotional superpositions collapse through:
    - Self-observation (introspection)
    - External observation (expression)
    - Environmental coupling (social context)
    """
    
    def __init__(
        self,
        n_emotions: int,
        hamiltonian: Optional[torch.Tensor] = None,
        decoherence_rate: float = 1.0,
        emotion_labels: Optional[List[str]] = None,
        dtype: torch.dtype = torch.complex64
    ):
        """
        Initialize emotional Lindblad dynamics.
        
        Args:
            n_emotions: Number of emotion basis states
            hamiltonian: Emotional transition Hamiltonian (default: none)
            decoherence_rate: Base decoherence rate
            emotion_labels: Names of emotions
            dtype: Complex dtype
        """
        if hamiltonian is None:
            hamiltonian = torch.zeros(n_emotions, n_emotions, dtype=dtype)
        
        # Dephasing in emotion basis
        lindblad_ops = []
        for i in range(n_emotions):
            proj = torch.zeros(n_emotions, n_emotions, dtype=dtype)
            proj[i, i] = 1.0
            label = emotion_labels[i] if emotion_labels else f"emotion_{i}"
            lindblad_ops.append(LindbladOperator(proj, decoherence_rate, f"dephase_{label}"))
        
        config = DensityMatrixConfig(n_dims=n_emotions, dtype=dtype)
        super().__init__(hamiltonian, lindblad_ops, config)
        
        self.n_emotions = n_emotions
        self.emotion_labels = emotion_labels
    
    def with_transition_rates(
        self,
        transitions: dict
    ) -> "EmotionalLindbladDynamics":
        """
        Add coherent transitions between emotions.
        
        Args:
            transitions: Dict of {(from, to): rate} for Hamiltonian coupling
            
        Returns:
            New dynamics with transitions
        """
        H = self.H.clone()
        
        for (i, j), rate in transitions.items():
            H[i, j] += rate
            H[j, i] += rate.conj() if isinstance(rate, complex) else rate
        
        return EmotionalLindbladDynamics(
            n_emotions=self.n_emotions,
            hamiltonian=H,
            decoherence_rate=self.lindblad_ops[0].rate,
            emotion_labels=self.emotion_labels,
            dtype=H.dtype
        )
