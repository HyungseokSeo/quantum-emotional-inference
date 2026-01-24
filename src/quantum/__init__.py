"""
Quantum formalism module.

Provides rigorous quantum-inspired representations for emotional states,
including density matrices, operators, dynamics, and information measures.
"""

from . import states
from . import operators
from . import dynamics
from . import information
from . import validation

from .states.density_matrix import DensityMatrix
from .states.ket import Ket
from .states.hilbert_space import HilbertSpace
from .states.emotional_basis import EmotionalBasis

from .operators.observable import Observable
from .operators.measurement import ProjectiveMeasurement, POVM
from .operators.unitary import UnitaryOperator

from .dynamics.lindblad import LindbladDynamics
from .dynamics.decoherence import Decoherence

from .information.entropy import von_neumann_entropy, linear_entropy
from .information.coherence import l1_coherence, relative_entropy_of_coherence
