"""Homeostatic regulation dynamics."""
import torch
from dataclasses import dataclass

from .setpoints import EmotionalSetpoint


@dataclass
class HomestaticRegulator:
    """Regulates emotional state toward setpoint."""
    
    setpoint: EmotionalSetpoint
    regulation_strength: float = 0.1
    
    def regulate(
        self,
        current_valence: float,
        current_arousal: float
    ) -> tuple:
        """Compute regulation force toward setpoint."""
        v_force = self.regulation_strength * (self.setpoint.target_valence - current_valence)
        a_force = self.regulation_strength * (self.setpoint.target_arousal - current_arousal)
        return v_force, a_force
