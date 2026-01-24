"""
Arousal as d²F/dt².

Core theoretical claim: Arousal (activation level) is the
acceleration of free energy change.

- |d²F/dt²| large: Rapid changes → high arousal
- |d²F/dt²| small: Stable dynamics → low arousal
"""
import torch
from typing import List
from dataclasses import dataclass

from ..quantum.quantum_variational import QuantumVariationalFreeEnergy
from ...quantum.states.density_matrix import DensityMatrix


@dataclass
class ArousalDynamics:
    """
    Computes arousal from free energy dynamics.
    
    Arousal = |d²F/dt²| (magnitude of acceleration)
    """
    
    fe_calculator: QuantumVariationalFreeEnergy
    hamiltonian: torch.Tensor
    history_length: int = 10
    
    def __post_init__(self):
        self.F_history: List[float] = []
        self.t_history: List[float] = []
        self.dF_dt_history: List[float] = []
    
    def update(self, rho: DensityMatrix, t: float) -> None:
        """Record free energy and compute derivatives."""
        F = self.fe_calculator.compute(rho, self.hamiltonian)
        self.F_history.append(F)
        self.t_history.append(t)
        
        # Compute dF/dt
        if len(self.F_history) >= 2:
            dF = self.F_history[-1] - self.F_history[-2]
            dt = self.t_history[-1] - self.t_history[-2]
            if dt > 1e-10:
                self.dF_dt_history.append(dF / dt)
        
        # Keep only recent history
        if len(self.F_history) > self.history_length:
            self.F_history.pop(0)
            self.t_history.pop(0)
        if len(self.dF_dt_history) > self.history_length:
            self.dF_dt_history.pop(0)
    
    def compute_arousal(self) -> float:
        """
        Compute current arousal as |d²F/dt²|.
        """
        if len(self.dF_dt_history) < 2:
            return 0.0
        
        # Compute d²F/dt²
        d_dFdt = self.dF_dt_history[-1] - self.dF_dt_history[-2]
        dt = self.t_history[-1] - self.t_history[-2]
        
        if dt < 1e-10:
            return 0.0
        
        d2F_dt2 = d_dFdt / dt
        
        # Arousal is magnitude of acceleration
        return abs(d2F_dt2)
    
    def arousal_from_trajectory(
        self,
        rho_trajectory: List[DensityMatrix],
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """Compute arousal over entire trajectory."""
        arousals = []
        
        for rho, t in zip(rho_trajectory, t_span):
            self.update(rho, t.item())
            a = self.compute_arousal()
            arousals.append(a)
        
        return torch.tensor(arousals)
    
    def reset(self):
        """Clear history."""
        self.F_history = []
        self.t_history = []
        self.dF_dt_history = []
