"""Continuous dimensional emotion representations (VAD space)."""
import torch
from typing import Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class DimensionalEmotionSpace:
    """
    Valence-Arousal-Dominance continuous emotion space.
    
    Maps emotional states to continuous coordinates:
    - Valence: positive ↔ negative
    - Arousal: high ↔ low activation
    - Dominance: in control ↔ controlled (optional)
    """
    
    use_dominance: bool = False
    
    @property
    def n_dims(self) -> int:
        return 3 if self.use_dominance else 2
    
    def to_cartesian(self, valence: float, arousal: float, dominance: float = 0.0) -> torch.Tensor:
        """Convert VAD to Cartesian coordinates."""
        if self.use_dominance:
            return torch.tensor([valence, arousal, dominance])
        return torch.tensor([valence, arousal])
    
    def to_polar(self, valence: float, arousal: float) -> Tuple[float, float]:
        """Convert to polar coordinates (intensity, angle)."""
        intensity = math.sqrt(valence**2 + arousal**2)
        angle = math.atan2(arousal, valence)
        return intensity, angle
    
    def from_polar(self, intensity: float, angle: float) -> Tuple[float, float]:
        """Convert from polar to Cartesian."""
        valence = intensity * math.cos(angle)
        arousal = intensity * math.sin(angle)
        return valence, arousal
    
    def distance(self, p1: torch.Tensor, p2: torch.Tensor) -> float:
        """Euclidean distance between emotional states."""
        return torch.norm(p1 - p2).item()
