"""Ambivalence quantification metrics."""
import torch
from typing import Tuple

from ...quantum.states.density_matrix import DensityMatrix


def ambivalence_score(rho: DensityMatrix) -> float:
    """
    Quantify ambivalence in emotional state.
    
    Based on coherence normalized by maximum possible.
    """
    n = rho.n_dims
    max_coherence = n - 1  # For maximally coherent pure state
    return min(rho.l1_coherence() / max_coherence, 1.0)


def ambivalence_vs_uncertainty(
    ambivalent_rho: DensityMatrix,
    uncertain_rho: DensityMatrix
) -> dict:
    """
    Compare ambivalent and uncertain states.
    
    Key validation: states with same diagonal should differ in coherence.
    """
    return {
        "ambivalent_coherence": ambivalent_rho.l1_coherence(),
        "uncertain_coherence": uncertain_rho.l1_coherence(),
        "coherence_ratio": (ambivalent_rho.l1_coherence() / 
                          (uncertain_rho.l1_coherence() + 1e-10)),
        "ambivalent_entropy": ambivalent_rho.von_neumann_entropy(),
        "uncertain_entropy": uncertain_rho.von_neumann_entropy(),
        "distinguishable": ambivalent_rho.l1_coherence() > uncertain_rho.l1_coherence() + 0.1,
    }
