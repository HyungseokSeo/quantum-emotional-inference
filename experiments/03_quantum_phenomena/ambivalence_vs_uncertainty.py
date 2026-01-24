"""
Experiment 03: Ambivalence vs Uncertainty.

Core theoretical validation: Can we empirically distinguish
coherent superposition (ambivalence) from classical mixture (uncertainty)?
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.quantum.states.emotional_basis import EmotionalBasis
from src.quantum.dynamics.lindblad import LindbladDynamics
from src.free_energy.quantum.quantum_variational import (
    QuantumVariationalFreeEnergy,
    free_energy_comparison
)


def compare_states():
    """Compare ambivalent and uncertain states."""
    basis = EmotionalBasis.ekman()
    
    # Same probabilities, different nature
    emotions = ["happy", "sad"]
    weights = [0.5, 0.5]
    
    # Ambivalent: |ψ⟩ = √0.5|happy⟩ + √0.5|sad⟩
    rho_ambivalent = basis.ambivalent(emotions, weights)
    
    # Uncertain: ρ = 0.5|happy⟩⟨happy| + 0.5|sad⟩⟨sad|
    rho_uncertain = basis.uncertain(emotions, weights)
    
    print("\n=== State Comparison ===")
    print(f"\nAmbivalent (coherent superposition):")
    print(f"  Matrix:\n{rho_ambivalent.matrix.real[:2, :2].numpy()}")
    print(f"  Purity: {rho_ambivalent.purity():.4f}")
    print(f"  Entropy: {rho_ambivalent.von_neumann_entropy():.4f}")
    print(f"  Coherence: {rho_ambivalent.l1_coherence():.4f}")
    
    print(f"\nUncertain (classical mixture):")
    print(f"  Matrix:\n{rho_uncertain.matrix.real[:2, :2].numpy()}")
    print(f"  Purity: {rho_uncertain.purity():.4f}")
    print(f"  Entropy: {rho_uncertain.von_neumann_entropy():.4f}")
    print(f"  Coherence: {rho_uncertain.l1_coherence():.4f}")
    
    return rho_ambivalent, rho_uncertain


def decoherence_trajectories(rho_ambivalent, rho_uncertain):
    """Show how ambivalent state decoheres to uncertain-like state."""
    n_emotions = rho_ambivalent.n_dims
    dynamics = LindbladDynamics.dephasing(n_emotions, rate=1.0)
    
    t_span = torch.linspace(0, 5, 50)
    
    trajectory_ambivalent = dynamics.evolve(rho_ambivalent, t_span)
    trajectory_uncertain = dynamics.evolve(rho_uncertain, t_span)
    
    coherence_ambivalent = [rho.l1_coherence() for rho in trajectory_ambivalent]
    coherence_uncertain = [rho.l1_coherence() for rho in trajectory_uncertain]
    
    print("\n=== Decoherence Trajectories ===")
    print(f"Initial coherence - Ambivalent: {coherence_ambivalent[0]:.4f}, "
          f"Uncertain: {coherence_uncertain[0]:.4f}")
    print(f"Final coherence - Ambivalent: {coherence_ambivalent[-1]:.4f}, "
          f"Uncertain: {coherence_uncertain[-1]:.4f}")
    print(f"\nAmbivalent state decoheres: coherence {coherence_ambivalent[0]:.4f} → "
          f"{coherence_ambivalent[-1]:.4f}")


def free_energy_analysis(rho_ambivalent, rho_uncertain):
    """Compare free energy of ambivalent vs uncertain states."""
    # Simple Hamiltonian (energy increases with emotion index)
    n = rho_ambivalent.n_dims
    H = torch.diag(torch.arange(n, dtype=torch.complex64))
    
    result = free_energy_comparison(rho_ambivalent, rho_uncertain, H, temperature=1.0)
    
    print("\n=== Free Energy Analysis ===")
    print(f"F(ambivalent): {result['F_ambivalent']:.4f}")
    print(f"F(uncertain):  {result['F_uncertain']:.4f}")
    print(f"Difference:    {result['difference']:.4f}")
    print(f"Coherence favored: {result['coherence_favored']}")


def main():
    print("=" * 60)
    print("Ambivalence vs Uncertainty: Core Theoretical Validation")
    print("=" * 60)
    
    rho_ambivalent, rho_uncertain = compare_states()
    decoherence_trajectories(rho_ambivalent, rho_uncertain)
    free_energy_analysis(rho_ambivalent, rho_uncertain)
    
    print("\n" + "=" * 60)
    print("Key findings:")
    print("1. Ambivalent and uncertain states have same diagonal (probabilities)")
    print("2. They differ in off-diagonal coherence")
    print("3. Decoherence transforms ambivalent → uncertain-like")
    print("4. Free energy captures this distinction")
    print("=" * 60)


if __name__ == "__main__":
    main()
