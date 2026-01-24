"""Combined categorical-dimensional emotion representations."""
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .categorical import CategoricalEmotionSpace
from .dimensional import DimensionalEmotionSpace
from ...quantum.states.density_matrix import DensityMatrix


@dataclass
class HybridEmotionSpace:
    """
    Hybrid space combining categorical and dimensional representations.
    
    Maintains both:
    - Quantum state in categorical basis (for ambivalence)
    - Continuous VAD coordinates (for dimensional analysis)
    """
    
    categorical: CategoricalEmotionSpace
    dimensional: DimensionalEmotionSpace
    
    # Mapping from categories to VA coordinates
    va_mapping: dict = None
    
    def __post_init__(self):
        if self.va_mapping is None:
            self.va_mapping = {
                "happy": (0.8, 0.4),
                "sad": (-0.6, -0.4),
                "angry": (-0.6, 0.8),
                "fearful": (-0.6, 0.6),
                "surprised": (0.2, 0.8),
                "disgusted": (-0.6, 0.2),
                "neutral": (0.0, 0.0),
            }
    
    @classmethod
    def default(cls) -> "HybridEmotionSpace":
        return cls(
            categorical=CategoricalEmotionSpace.ekman(),
            dimensional=DimensionalEmotionSpace()
        )
    
    def categorical_to_dimensional(self, rho: DensityMatrix) -> Tuple[float, float]:
        """Map density matrix to VA coordinates."""
        probs = rho.probabilities()
        valence, arousal = 0.0, 0.0
        
        for i, emotion in enumerate(self.categorical.emotions):
            if emotion in self.va_mapping:
                v, a = self.va_mapping[emotion]
                p = probs[i].item()
                valence += p * v
                arousal += p * a
        
        return valence, arousal
    
    def dimensional_to_categorical(
        self, 
        valence: float, 
        arousal: float,
        temperature: float = 1.0
    ) -> DensityMatrix:
        """Map VA coordinates to density matrix via softmax over distances."""
        distances = []
        for emotion in self.categorical.emotions:
            if emotion in self.va_mapping:
                v, a = self.va_mapping[emotion]
                d = ((valence - v)**2 + (arousal - a)**2)**0.5
            else:
                d = float('inf')
            distances.append(d)
        
        distances = torch.tensor(distances)
        probs = torch.softmax(-distances / temperature, dim=0)
        
        return self.categorical.basis.from_probabilities(probs)
