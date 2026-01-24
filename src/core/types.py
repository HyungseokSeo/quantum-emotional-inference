"""
Type definitions and protocols for the framework.
"""

from typing import Protocol, TypeVar, Union, Tuple, List, Optional, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np


# Type aliases
Tensor = torch.Tensor
Array = np.ndarray
TensorOrArray = Union[Tensor, Array]
Shape = Tuple[int, ...]


class StateType(Enum):
    """Classification of quantum state types."""
    PURE = "pure"
    MIXED = "mixed"
    MAXIMALLY_MIXED = "maximally_mixed"


class EmotionCategory(Enum):
    """Standard emotion categories (Ekman basic emotions)."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"


class HierarchyLevel(Enum):
    """Levels in the temporal hierarchy."""
    REACTIVE = "reactive"      # ~100ms, high coherence
    ADAPTIVE = "adaptive"      # ~1s, partial decoherence
    REFLECTIVE = "reflective"  # ~10s, classical limit


@dataclass
class EmotionalState:
    """Container for emotional state information."""
    probabilities: Tensor  # Diagonal probabilities
    coherences: Optional[Tensor] = None  # Off-diagonal elements
    hierarchy_level: HierarchyLevel = HierarchyLevel.REACTIVE
    timestamp: Optional[float] = None
    
    @property
    def is_ambivalent(self) -> bool:
        """True if state has genuine superposition (non-zero coherence)."""
        if self.coherences is None:
            return False
        return torch.sum(torch.abs(self.coherences)).item() > 1e-6


@dataclass
class FreeEnergyState:
    """Container for Free Energy dynamics."""
    F: float  # Current free energy
    dF_dt: float  # Valence (rate of change)
    d2F_dt2: float  # Arousal (acceleration)
    entropy: float  # Von Neumann or Shannon entropy
    
    @property
    def valence(self) -> float:
        """Valence as dF/dt."""
        return self.dF_dt
    
    @property
    def arousal(self) -> float:
        """Arousal as d²F/dt²."""
        return self.d2F_dt2


@runtime_checkable
class QuantumState(Protocol):
    """Protocol for quantum state representations."""
    
    @property
    def n_dims(self) -> int:
        """Dimension of the Hilbert space."""
        ...
    
    def purity(self) -> float:
        """Tr(ρ²) — 1 for pure, 1/n for maximally mixed."""
        ...
    
    def von_neumann_entropy(self) -> float:
        """S(ρ) = -Tr(ρ ln ρ)."""
        ...
    
    def expectation(self, observable: Tensor) -> float:
        """⟨O⟩ = Tr(ρO)."""
        ...


@runtime_checkable
class DynamicalSystem(Protocol):
    """Protocol for dynamical systems."""
    
    def step(self, state: Tensor, dt: float) -> Tensor:
        """Evolve state by time step dt."""
        ...
    
    def evolve(self, state: Tensor, t_span: Tensor) -> List[Tensor]:
        """Evolve state over time span."""
        ...


# Generic type variables
T = TypeVar('T')
StateT = TypeVar('StateT', bound=QuantumState)
