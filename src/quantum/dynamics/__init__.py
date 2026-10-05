"""
Quantum dynamics for emotional state evolution.

This module implements:
- Schrödinger dynamics (coherent evolution)
- Lindblad dynamics (open system with decoherence)
- Decoherence models for emotional collapse
- Quantum channels (general CPTP maps)
"""

from .schrodinger import SchrodingerDynamics
from .lindblad import LindbladDynamics, LindbladOperator
from .decoherence import Decoherence, DephasingChannel, AmplitudeDamping
from .quantum_channel import QuantumChannel, KrausChannel
from . import liouvillian

__all__ = [
    'SchrodingerDynamics',
    'LindbladDynamics',
    'LindbladOperator',
    'Decoherence',
    'DephasingChannel',
    'AmplitudeDamping',
    'QuantumChannel',
    'KrausChannel',
    'liouvillian',
]
