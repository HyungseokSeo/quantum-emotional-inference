"""
Valence as dF/dt.

Core theoretical claim: Valence (positive/negative affect) is the
rate of change of free energy.

- dF/dt < 0: Free energy decreasing → positive valence (good)
- dF/dt > 0: Free energy increasing → negative valence (bad)
- dF/dt = 0: Steady state → neutral valence
"""
import torch
from typing import List, Optional
from dataclasses import dataclass

from ..quantum.quantum_variational import QuantumVariationalFreeEnergy
from ...quantum.states.density_matrix import DensityMatrix


@dataclass
class ValenceDynamics:
    """
    Computes valence from free energy dynamics.
    
    Valence = -dF/dt (negative because decreasing F is positive)
    """
    
    fe_calculator: QuantumVariationalFreeEnergy
    hamiltonian: torch.Tensor
    history_length: int = 10
    smoothing: float = 0.9
    
    def __post_init__(self):
        self.F_history: List[float] = []
        self.t_history: List[float] = []
    
    def update(self, rho: DensityMatrix, t: float) -> None:
        """Record free energy at time t."""
        F = self.fe_calculator.compute(rho, self.hamiltonian)
        self.F_history.append(F)
        self.t_history.append(t)
        
        # Keep only recent history
        if len(self.F_history) > self.history_length:
            self.F_history.pop(0)
            self.t_history.pop(0)
    
    def compute_valence(self) -> float:
        """
        Compute current valence as -dF/dt.
        
        Uses exponentially weighted moving average for smoothing.
        """
        if len(self.F_history) < 2:
            return 0.0
        
        # Compute dF/dt
        dF = self.F_history[-1] - self.F_history[-2]
        dt = self.t_history[-1] - self.t_history[-2]
        
        if dt < 1e-10:
            return 0.0
        
        dF_dt = dF / dt
        
        # Valence is negative of dF/dt
        return -dF_dt
    
    def valence_from_trajectory(
        self,
        rho_trajectory: List[DensityMatrix],
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """Compute valence over entire trajectory."""
        valences = []
        
        for i, (rho, t) in enumerate(zip(rho_trajectory, t_span)):
            self.update(rho, t.item())
            v = self.compute_valence()
            valences.append(v)
        
        return torch.tensor(valences)
    
    def reset(self):
        """Clear history."""
        self.F_history = []
        self.t_history = []
