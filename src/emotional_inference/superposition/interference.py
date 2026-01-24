"""
Emotional interference effects.

When emotions are in superposition, they can interfere constructively
or destructively, affecting measurement probabilities.
"""
import torch
from typing import List, Optional
from dataclasses import dataclass
import math

from ...quantum.states.ket import Ket
from ...quantum.states.hilbert_space import HilbertSpace


@dataclass
class EmotionalInterference:
    """
    Models interference between emotional states.
    
    When two paths lead to same outcome, amplitudes add:
    P = |α₁ + α₂|² = |α₁|² + |α₂|² + 2Re(α₁*α₂)
    
    The cross term 2Re(α₁*α₂) is the interference.
    """
    
    space: HilbertSpace
    
    def create_superposition(
        self,
        states: List[int],
        amplitudes: List[complex],
        phases: Optional[List[float]] = None
    ) -> Ket:
        """Create superposition with specified amplitudes and phases."""
        if phases is None:
            phases = [0.0] * len(states)
        
        full_amps = torch.zeros(self.space.dim, dtype=torch.complex64)
        for state, amp, phase in zip(states, amplitudes, phases):
            full_amps[state] = amp * torch.exp(1j * torch.tensor(phase))
        
        return Ket.from_amplitudes(full_amps, self.space)
    
    def interference_term(
        self,
        amp1: complex,
        amp2: complex
    ) -> float:
        """Compute interference term: 2Re(α₁*α₂)."""
        return 2 * (amp1.conjugate() * amp2).real
    
    def probability_with_interference(
        self,
        amp1: complex,
        amp2: complex
    ) -> float:
        """
        Total probability with interference:
        P = |α₁|² + |α₂|² + 2Re(α₁*α₂)
        """
        return abs(amp1)**2 + abs(amp2)**2 + self.interference_term(amp1, amp2)
    
    def demonstrate_interference(
        self,
        phase_diff: float = 0.0
    ) -> dict:
        """
        Demonstrate constructive/destructive interference.
        
        Args:
            phase_diff: Phase difference between paths
            
        Returns:
            Dict with probabilities and interference analysis
        """
        amp = 1 / math.sqrt(2)
        amp1 = complex(amp, 0)
        amp2 = amp * complex(math.cos(phase_diff), math.sin(phase_diff))
        
        classical = abs(amp1)**2 + abs(amp2)**2
        interference = self.interference_term(amp1, amp2)
        quantum = self.probability_with_interference(amp1, amp2)
        
        return {
            "classical_prediction": classical,
            "interference_term": interference,
            "quantum_prediction": quantum,
            "phase_difference": phase_diff,
            "constructive": interference > 0,
            "destructive": interference < 0,
        }
