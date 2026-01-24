"""Emotional setpoints for homeostatic regulation."""
import torch
from dataclasses import dataclass


@dataclass
class EmotionalSetpoint:
    """
    Target emotional state that system tries to maintain.
    
    Deviations from setpoint drive emotional dynamics.
    """
    
    target_valence: float = 0.2  # Slightly positive baseline
    target_arousal: float = 0.3  # Moderate activation
    tolerance: float = 0.1
    
    def deviation(self, current_valence: float, current_arousal: float) -> float:
        """Compute deviation from setpoint."""
        v_dev = abs(current_valence - self.target_valence)
        a_dev = abs(current_arousal - self.target_arousal)
        return (v_dev + a_dev) / 2
    
    def in_homeostasis(self, valence: float, arousal: float) -> bool:
        """Check if within homeostatic range."""
        return self.deviation(valence, arousal) < self.tolerance
