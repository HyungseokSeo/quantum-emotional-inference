"""
Existential stakes: tying system existence to free energy.

Core theoretical claim: Genuine emotion requires that the system's
continued operation depends on successful inference.

If F exceeds a threshold, the system "dies" (stops functioning).
This creates genuine stakes that classical emotion recognition lacks.
"""
import torch
from typing import Optional, Callable
from dataclasses import dataclass

from ..quantum.quantum_variational import QuantumVariationalFreeEnergy
from ...quantum.states.density_matrix import DensityMatrix


@dataclass
class ExistentialDynamics:
    """
    Existential stakes for emotional inference.
    
    The system maintains a "life signal" that depends on free energy:
    - F < threshold: system operates normally
    - F ≥ threshold: system "dies" (cannot continue)
    
    This grounds emotion in survival-relevant computation.
    """
    
    fe_calculator: QuantumVariationalFreeEnergy
    hamiltonian: torch.Tensor
    existence_threshold: float
    recovery_rate: float = 0.1
    
    def __post_init__(self):
        self.vitality: float = 1.0  # 0 = dead, 1 = fully alive
        self.alive: bool = True
    
    def update(self, rho: DensityMatrix, dt: float) -> float:
        """
        Update existential state based on free energy.
        
        Returns current vitality.
        """
        if not self.alive:
            return 0.0
        
        F = self.fe_calculator.compute(rho, self.hamiltonian)
        
        # Vitality dynamics: decreases when F > threshold
        if F > self.existence_threshold:
            damage = (F - self.existence_threshold) * dt
            self.vitality -= damage
        else:
            # Recovery when F is low
            recovery = self.recovery_rate * (self.existence_threshold - F) * dt
            self.vitality = min(1.0, self.vitality + recovery)
        
        # Check for death
        if self.vitality <= 0:
            self.alive = False
            self.vitality = 0.0
        
        return self.vitality
    
    def existential_urgency(self, rho: DensityMatrix) -> float:
        """
        Measure urgency of free energy minimization.
        
        Higher when close to death threshold.
        """
        F = self.fe_calculator.compute(rho, self.hamiltonian)
        
        if F >= self.existence_threshold:
            return 1.0  # Maximum urgency
        
        # Urgency increases as F approaches threshold
        return F / self.existence_threshold
    
    def survival_gradient(self, rho: DensityMatrix) -> torch.Tensor:
        """
        Gradient that points toward survival.
        
        The system should minimize this to stay alive.
        """
        return self.fe_calculator.free_energy_gradient(rho, self.hamiltonian)
    
    def reset(self):
        """Resurrect the system."""
        self.vitality = 1.0
        self.alive = True
