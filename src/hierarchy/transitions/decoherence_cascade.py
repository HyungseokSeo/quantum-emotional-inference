"""
Decoherence cascade: progressive collapse through hierarchy.

As information flows up the hierarchy:
Reactive (coherent) → Adaptive (partial) → Reflective (classical)

Each level "measures" the state from below, causing decoherence.
"""
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..levels.reactive import ReactiveLevel
from ..levels.adaptive import AdaptiveLevel
from ..levels.reflective import ReflectiveLevel
from ...quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig
from ...quantum.states.emotional_basis import EmotionalBasis, EKMAN_EMOTIONS


@dataclass
class DecoherenceCascade:
    """
    Full decoherence cascade through temporal hierarchy.
    
    Models the progressive collapse of emotional superposition
    as information ascends from reactive to reflective processing.
    """
    
    reactive: ReactiveLevel
    adaptive: AdaptiveLevel
    reflective: ReflectiveLevel
    
    @classmethod
    def default(
        cls,
        n_emotions: int = 7,
        emotion_labels: Optional[List[str]] = None
    ) -> "DecoherenceCascade":
        """Create default cascade with Ekman emotions."""
        if emotion_labels is None:
            emotion_labels = EKMAN_EMOTIONS
        
        return cls(
            reactive=ReactiveLevel(n_emotions),
            adaptive=AdaptiveLevel(n_emotions),
            reflective=ReflectiveLevel(n_emotions, emotion_labels)
        )
    
    def cascade(
        self,
        rho_input: DensityMatrix,
        context: Optional[torch.Tensor] = None,
        return_trajectory: bool = False
    ) -> DensityMatrix:
        """
        Run full cascade from input to conscious output.
        
        Args:
            rho_input: Initial emotional state (can be superposition)
            context: Environmental context
            return_trajectory: If True, return all intermediate states
            
        Returns:
            Final (near-classical) emotional state
        """
        trajectory = [rho_input]
        
        # Reactive processing
        rho_reactive = self.reactive.process(rho_input)
        trajectory.append(rho_reactive)
        
        # Adaptive processing (with context)
        rho_adaptive = self.adaptive.process(rho_reactive, context)
        trajectory.append(rho_adaptive)
        
        # Reflective processing
        rho_reflective = self.reflective.process(rho_adaptive)
        trajectory.append(rho_reflective)
        
        if return_trajectory:
            return trajectory
        return rho_reflective
    
    def coherence_profile(self, rho_input: DensityMatrix) -> dict:
        """
        Track coherence through the cascade.
        """
        trajectory = self.cascade(rho_input, return_trajectory=True)
        
        return {
            "input": trajectory[0].l1_coherence(),
            "post_reactive": trajectory[1].l1_coherence(),
            "post_adaptive": trajectory[2].l1_coherence(),
            "post_reflective": trajectory[3].l1_coherence(),
        }
    
    def characteristic_times(self) -> dict:
        """Return characteristic timescales of each level."""
        return {
            "reactive": self.reactive.timescale,
            "adaptive": self.adaptive.timescale,
            "reflective": self.reflective.timescale,
            "total": (self.reactive.timescale + 
                     self.adaptive.timescale + 
                     self.reflective.timescale)
        }
