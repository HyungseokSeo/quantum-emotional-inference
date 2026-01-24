"""
Domain-specific operators for emotional inference.

These operators map abstract quantum formalism to emotional constructs:
- ValenceOperator: Measures positive/negative affect
- ArousalOperator: Measures activation level
- EmotionalObservable: General emotion measurement
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

from .observable import Observable


# Standard valence-arousal mappings for Ekman emotions
DEFAULT_EMOTION_VA = {
    "happy": (0.8, 0.4),
    "sad": (-0.6, -0.4),
    "angry": (-0.6, 0.8),
    "fearful": (-0.6, 0.6),
    "surprised": (0.2, 0.8),
    "disgusted": (-0.6, 0.2),
    "neutral": (0.0, 0.0),
}


class EmotionalObservable(Observable):
    """
    Observable specialized for emotional measurements.
    
    Extends Observable with emotion-specific functionality.
    """
    
    def __init__(
        self,
        matrix: torch.Tensor,
        emotion_labels: List[str],
        name: str = "EmotionalObservable"
    ):
        super().__init__(matrix=matrix, name=name)
        self.emotion_labels = emotion_labels
    
    @classmethod
    def category_observable(
        cls,
        emotion_labels: List[str],
        dtype: torch.dtype = torch.complex64
    ) -> "EmotionalObservable":
        """
        Create observable for emotion category measurement.
        
        Eigenvalues are indices 0, 1, 2, ..., n-1.
        Eigenstates are basis states |emotionᵢ⟩.
        
        Args:
            emotion_labels: List of emotion names
            dtype: Complex dtype
            
        Returns:
            Emotion category observable
        """
        n = len(emotion_labels)
        matrix = torch.diag(torch.arange(n, dtype=dtype))
        return cls(matrix=matrix, emotion_labels=emotion_labels, name="EmotionCategory")
    
    def expectation_emotion(self, rho: torch.Tensor) -> Dict[str, float]:
        """
        Get probability of each emotion.
        
        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        probs = torch.diag(rho).real
        return {label: probs[i].item() for i, label in enumerate(self.emotion_labels)}
    
    def most_likely_emotion(self, rho: torch.Tensor) -> str:
        """Return emotion with highest probability."""
        probs = torch.diag(rho).real
        idx = torch.argmax(probs).item()
        return self.emotion_labels[idx]


class ValenceOperator(Observable):
    """
    Observable for measuring emotional valence (positive/negative).
    
    Valence eigenvalues range from -1 (negative) to +1 (positive).
    Eigenstates are emotion basis states weighted by their valence.
    
    Example:
        >>> V = ValenceOperator.from_emotions(EKMAN_EMOTIONS)
        >>> valence = V.expectation(rho)  # Expected valence
    """
    
    def __init__(
        self,
        matrix: torch.Tensor,
        emotion_labels: List[str],
        valence_map: Dict[str, float]
    ):
        super().__init__(matrix=matrix, name="Valence")
        self.emotion_labels = emotion_labels
        self.valence_map = valence_map
    
    @classmethod
    def from_emotions(
        cls,
        emotion_labels: List[str],
        valence_map: Optional[Dict[str, float]] = None,
        dtype: torch.dtype = torch.complex64
    ) -> "ValenceOperator":
        """
        Create valence operator from emotion labels.
        
        Args:
            emotion_labels: List of emotion names
            valence_map: Mapping from emotion to valence [-1, 1]
            dtype: Complex dtype
            
        Returns:
            Valence observable
        """
        if valence_map is None:
            valence_map = {e: DEFAULT_EMOTION_VA.get(e, (0, 0))[0] 
                         for e in emotion_labels}
        
        n = len(emotion_labels)
        eigenvalues = torch.tensor(
            [valence_map.get(e, 0.0) for e in emotion_labels],
            dtype=dtype
        )
        matrix = torch.diag(eigenvalues)
        
        return cls(
            matrix=matrix,
            emotion_labels=emotion_labels,
            valence_map=valence_map
        )
    
    def valence(self, rho: torch.Tensor) -> float:
        """Compute expected valence from density matrix."""
        return self.expectation(rho)


class ArousalOperator(Observable):
    """
    Observable for measuring emotional arousal (activation level).
    
    Arousal eigenvalues range from -1 (low/calm) to +1 (high/excited).
    
    Example:
        >>> A = ArousalOperator.from_emotions(EKMAN_EMOTIONS)
        >>> arousal = A.expectation(rho)  # Expected arousal
    """
    
    def __init__(
        self,
        matrix: torch.Tensor,
        emotion_labels: List[str],
        arousal_map: Dict[str, float]
    ):
        super().__init__(matrix=matrix, name="Arousal")
        self.emotion_labels = emotion_labels
        self.arousal_map = arousal_map
    
    @classmethod
    def from_emotions(
        cls,
        emotion_labels: List[str],
        arousal_map: Optional[Dict[str, float]] = None,
        dtype: torch.dtype = torch.complex64
    ) -> "ArousalOperator":
        """
        Create arousal operator from emotion labels.
        
        Args:
            emotion_labels: List of emotion names
            arousal_map: Mapping from emotion to arousal [-1, 1]
            dtype: Complex dtype
            
        Returns:
            Arousal observable
        """
        if arousal_map is None:
            arousal_map = {e: DEFAULT_EMOTION_VA.get(e, (0, 0))[1] 
                         for e in emotion_labels}
        
        n = len(emotion_labels)
        eigenvalues = torch.tensor(
            [arousal_map.get(e, 0.0) for e in emotion_labels],
            dtype=dtype
        )
        matrix = torch.diag(eigenvalues)
        
        return cls(
            matrix=matrix,
            emotion_labels=emotion_labels,
            arousal_map=arousal_map
        )
    
    def arousal(self, rho: torch.Tensor) -> float:
        """Compute expected arousal from density matrix."""
        return self.expectation(rho)


def check_complementarity(
    valence_op: ValenceOperator,
    arousal_op: ArousalOperator
) -> bool:
    """
    Check if valence and arousal operators commute.
    
    In classical dimensional models, V and A are independent.
    Non-commuting operators would indicate "quantum" structure
    where measuring valence affects arousal measurement.
    
    For diagonal operators (standard construction), they always commute.
    """
    return valence_op.commutes_with(arousal_op)


def uncertainty_relation(
    obs1: Observable,
    obs2: Observable,
    rho: torch.Tensor
) -> float:
    """
    Compute Robertson-Schrödinger uncertainty relation.
    
    ΔA · ΔB ≥ |⟨[A,B]⟩|/2
    
    Returns the lower bound on product of uncertainties.
    """
    commutator = obs1.commutator(obs2)
    expectation_comm = torch.trace(rho @ commutator)
    return abs(expectation_comm.item()) / 2
