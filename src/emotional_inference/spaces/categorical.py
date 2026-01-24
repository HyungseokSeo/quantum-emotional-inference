"""Discrete emotion basis representations."""
import torch
from typing import List, Optional
from dataclasses import dataclass

from ...quantum.states.emotional_basis import EmotionalBasis, EKMAN_EMOTIONS


@dataclass
class CategoricalEmotionSpace:
    """
    Categorical emotion space with discrete basis states.
    
    Each emotion is a basis vector: |happy⟩, |sad⟩, etc.
    Superpositions represent genuine ambivalence.
    """
    
    basis: EmotionalBasis
    
    @classmethod
    def ekman(cls, include_neutral: bool = True) -> "CategoricalEmotionSpace":
        return cls(basis=EmotionalBasis.ekman(include_neutral))
    
    @classmethod
    def from_labels(cls, labels: List[str]) -> "CategoricalEmotionSpace":
        return cls(basis=EmotionalBasis.from_emotions(labels))
    
    @property
    def n_emotions(self) -> int:
        return self.basis.n_emotions
    
    @property
    def emotions(self) -> List[str]:
        return self.basis.emotion_labels
    
    def pure_state(self, emotion: str):
        """Get pure state |emotion⟩."""
        return self.basis.basis_state(emotion)
    
    def superposition(self, emotions: List[str], weights: List[float]):
        """Create coherent superposition (ambivalence)."""
        return self.basis.ambivalent(emotions, weights)
    
    def mixture(self, emotions: List[str], probabilities: List[float]):
        """Create classical mixture (uncertainty)."""
        return self.basis.uncertain(emotions, probabilities)
