"""
Emotion-specific basis sets for quantum emotional inference.

This module defines standard emotional bases and provides utilities
for constructing custom emotional Hilbert spaces.

Supported basis types:
- Categorical: Discrete emotions (Ekman, Plutchik, etc.)
- Dimensional: Continuous dimensions (Valence-Arousal-Dominance)
- Hybrid: Combinations of categorical and dimensional
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .hilbert_space import HilbertSpace
from .ket import Ket
from .density_matrix import DensityMatrix, DensityMatrixConfig


# ==================== Standard Emotion Sets ====================

# Ekman's 6 basic emotions + neutral
EKMAN_EMOTIONS = [
    "happy",
    "sad", 
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "neutral"
]

# Plutchik's 8 primary emotions (emotion wheel)
PLUTCHIK_EMOTIONS = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation"
]

# FER2013 dataset emotions
FER2013_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# AffectNet emotions
AFFECTNET_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
    "contempt"
]


@dataclass
class EmotionalBasis:
    """
    Emotional basis for quantum state representation.
    
    This class provides emotion-specific utilities on top of the
    generic HilbertSpace, including:
    - Named access to emotional states
    - Predefined superpositions (ambivalent states)
    - Conversion to/from dimensional representations
    
    Example:
        >>> basis = EmotionalBasis.ekman()
        >>> happy_state = basis.basis_state("happy")
        >>> mixed = basis.ambivalent(["happy", "sad"], [0.6, 0.4])
    """
    
    space: HilbertSpace
    emotion_labels: List[str]
    
    @classmethod
    def from_emotions(
        cls,
        emotions: List[str],
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu"
    ) -> "EmotionalBasis":
        """
        Create emotional basis from list of emotion labels.
        
        Args:
            emotions: List of emotion names
            dtype: Complex dtype
            device: Computation device
            
        Returns:
            EmotionalBasis instance
        """
        space = HilbertSpace.from_labels(emotions, dtype, device)
        return cls(space=space, emotion_labels=emotions)
    
    @classmethod
    def ekman(cls, include_neutral: bool = True, **kwargs) -> "EmotionalBasis":
        """Create basis from Ekman's basic emotions."""
        emotions = EKMAN_EMOTIONS if include_neutral else EKMAN_EMOTIONS[:-1]
        return cls.from_emotions(emotions, **kwargs)
    
    @classmethod
    def plutchik(cls, **kwargs) -> "EmotionalBasis":
        """Create basis from Plutchik's emotion wheel."""
        return cls.from_emotions(PLUTCHIK_EMOTIONS, **kwargs)
    
    @classmethod
    def fer2013(cls, **kwargs) -> "EmotionalBasis":
        """Create basis matching FER2013 dataset."""
        return cls.from_emotions(FER2013_EMOTIONS, **kwargs)
    
    @classmethod
    def affectnet(cls, **kwargs) -> "EmotionalBasis":
        """Create basis matching AffectNet dataset."""
        return cls.from_emotions(AFFECTNET_EMOTIONS, **kwargs)
    
    @property
    def n_emotions(self) -> int:
        """Number of emotions in the basis."""
        return self.space.dim
    
    @property
    def config(self) -> DensityMatrixConfig:
        """Get density matrix config for this basis."""
        return DensityMatrixConfig(
            n_dims=self.n_emotions,
            dtype=self.space.dtype,
            device=self.space.device
        )
    
    def emotion_index(self, emotion: str) -> int:
        """Get index of emotion in basis."""
        return self.space.label_to_index(emotion)
    
    def index_emotion(self, index: int) -> str:
        """Get emotion label for index."""
        return self.space.index_to_label(index)
    
    def basis_state(self, emotion: str) -> Ket:
        """
        Get pure basis state for single emotion.
        
        Args:
            emotion: Emotion label
            
        Returns:
            Ket representing pure emotional state |emotion⟩
        """
        return Ket.from_label(emotion, self.space)
    
    def basis_density_matrix(self, emotion: str) -> DensityMatrix:
        """
        Get density matrix for pure emotional state.
        
        Args:
            emotion: Emotion label
            
        Returns:
            Density matrix ρ = |emotion⟩⟨emotion|
        """
        ket = self.basis_state(emotion)
        return ket.to_density_matrix()
    
    def superposition(
        self,
        emotions: List[str],
        amplitudes: Optional[List[complex]] = None,
        phases: Optional[List[float]] = None
    ) -> Ket:
        """
        Create superposition of emotions.
        
        |ψ⟩ = Σᵢ αᵢ e^{iφᵢ} |emotionᵢ⟩
        
        Args:
            emotions: List of emotions to include
            amplitudes: Magnitudes (default: uniform)
            phases: Phases in radians (default: zero)
            
        Returns:
            Superposition ket
        """
        n = len(emotions)
        
        if amplitudes is None:
            amplitudes = [1.0] * n
        if phases is None:
            phases = [0.0] * n
        
        # Build full amplitude vector
        full_amplitudes = torch.zeros(
            self.n_emotions, 
            dtype=self.space.dtype, 
            device=self.space.device
        )
        
        for emotion, amp, phase in zip(emotions, amplitudes, phases):
            idx = self.emotion_index(emotion)
            full_amplitudes[idx] = amp * torch.exp(torch.tensor(1j * phase))
        
        return Ket.from_amplitudes(full_amplitudes, self.space)
    
    def ambivalent(
        self,
        emotions: List[str],
        weights: List[float]
    ) -> DensityMatrix:
        """
        Create ambivalent state (coherent superposition).
        
        This represents genuinely feeling multiple emotions simultaneously,
        distinct from uncertainty about which emotion one feels.
        
        Args:
            emotions: List of emotions in superposition
            weights: Relative weights (will be normalized)
            
        Returns:
            Pure state density matrix with non-zero coherences
        """
        # Normalize weights to amplitudes
        weights = torch.tensor(weights, dtype=torch.float32)
        amplitudes = torch.sqrt(weights / weights.sum())
        
        ket = self.superposition(emotions, amplitudes.tolist())
        return ket.to_density_matrix()
    
    def uncertain(
        self,
        emotions: List[str],
        probabilities: List[float]
    ) -> DensityMatrix:
        """
        Create uncertain state (classical mixture).
        
        This represents uncertainty about which emotion one feels,
        without genuine ambivalence.
        
        Args:
            emotions: List of possible emotions
            probabilities: Probability of each emotion
            
        Returns:
            Mixed state density matrix (diagonal, no coherences)
        """
        probs = torch.tensor(probabilities, dtype=torch.float32)
        probs = probs / probs.sum()
        
        # Build full probability vector
        full_probs = torch.zeros(self.n_emotions, dtype=torch.float32)
        for emotion, p in zip(emotions, probs):
            idx = self.emotion_index(emotion)
            full_probs[idx] = p
        
        return DensityMatrix.from_probabilities(full_probs, self.config)
    
    def from_probabilities(self, probs: torch.Tensor) -> DensityMatrix:
        """
        Create density matrix from probability vector.
        
        Args:
            probs: Probabilities for each emotion
            
        Returns:
            Diagonal density matrix (classical, no coherence)
        """
        return DensityMatrix.from_probabilities(probs, self.config)
    
    def maximally_mixed(self) -> DensityMatrix:
        """
        Create maximally mixed state (complete uncertainty).
        
        Equal probability for all emotions, no coherences.
        """
        return DensityMatrix.maximally_mixed(self.config)
    
    # ==================== Dimensional Mappings ====================
    
    def to_valence_arousal(self, state: DensityMatrix) -> Tuple[float, float]:
        """
        Map emotional state to valence-arousal coordinates.
        
        Uses standard mappings:
        - Valence: positive (happy) to negative (sad, angry, fearful)
        - Arousal: high (angry, fearful, surprised) to low (sad, neutral)
        
        Args:
            state: Emotional density matrix
            
        Returns:
            (valence, arousal) tuple in [-1, 1] range
        """
        # Standard VA coordinates for Ekman emotions
        va_map = {
            "happy": (0.8, 0.4),
            "sad": (-0.6, -0.4),
            "angry": (-0.6, 0.8),
            "fearful": (-0.6, 0.6),
            "surprised": (0.2, 0.8),
            "disgusted": (-0.6, 0.2),
            "neutral": (0.0, 0.0),
            # Plutchik additions
            "joy": (0.8, 0.4),
            "trust": (0.4, 0.2),
            "fear": (-0.6, 0.6),
            "surprise": (0.2, 0.8),
            "sadness": (-0.6, -0.4),
            "disgust": (-0.6, 0.2),
            "anger": (-0.6, 0.8),
            "anticipation": (0.4, 0.6),
            # AffectNet additions
            "contempt": (-0.4, 0.2),
        }
        
        probs = state.probabilities()
        valence = 0.0
        arousal = 0.0
        
        for i, emotion in enumerate(self.emotion_labels):
            if emotion in va_map:
                v, a = va_map[emotion]
                p = probs[i].item()
                valence += p * v
                arousal += p * a
        
        return (valence, arousal)
    
    def from_valence_arousal(
        self,
        valence: float,
        arousal: float,
        method: str = "softmax"
    ) -> DensityMatrix:
        """
        Create emotional state from valence-arousal coordinates.
        
        Args:
            valence: Valence in [-1, 1]
            arousal: Arousal in [-1, 1]
            method: Mapping method ("softmax" or "nearest")
            
        Returns:
            Density matrix
        """
        # VA coordinates for mapping
        va_map = {
            "happy": (0.8, 0.4),
            "sad": (-0.6, -0.4),
            "angry": (-0.6, 0.8),
            "fearful": (-0.6, 0.6),
            "surprised": (0.2, 0.8),
            "disgusted": (-0.6, 0.2),
            "neutral": (0.0, 0.0),
        }
        
        # Compute distances
        distances = []
        for emotion in self.emotion_labels:
            if emotion in va_map:
                v, a = va_map[emotion]
                d = ((valence - v) ** 2 + (arousal - a) ** 2) ** 0.5
            else:
                d = float('inf')
            distances.append(d)
        
        distances = torch.tensor(distances)
        
        if method == "softmax":
            # Convert distances to probabilities via softmax
            probs = torch.softmax(-distances * 2, dim=0)  # Temperature = 0.5
        elif method == "nearest":
            # Winner-take-all
            probs = torch.zeros(self.n_emotions)
            probs[torch.argmin(distances)] = 1.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return DensityMatrix.from_probabilities(probs, self.config)
    
    def __repr__(self) -> str:
        return f"EmotionalBasis({self.emotion_labels})"
