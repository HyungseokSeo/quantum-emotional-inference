"""
Experiment 01: Validate density matrix properties.

Tests:
1. Purity bounds
2. Von Neumann entropy properties
3. Coherence measures
4. Ambivalence vs uncertainty distinction
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig
from src.quantum.states.emotional_basis import EmotionalBasis


def test_purity_bounds():
    """Verify purity is bounded: 1/n ≤ Tr(ρ²) ≤ 1."""
    config = DensityMatrixConfig(n_dims=7)
    
    results = []
    
    # Pure state (purity = 1)
    psi = torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.complex64)
    rho_pure = DensityMatrix.from_pure_state(psi, config)
    results.append(("Pure", rho_pure.purity()))
    
    # Maximally mixed (purity = 1/n)
    rho_max_mixed = DensityMatrix.maximally_mixed(config)
    results.append(("Max Mixed", rho_max_mixed.purity()))
    
    # Superposition
    psi_super = torch.tensor([1, 1, 0, 0, 0, 0, 0], dtype=torch.complex64)
    rho_super = DensityMatrix.from_pure_state(psi_super, config)
    results.append(("Superposition", rho_super.purity()))
    
    # Mixed state
    probs = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05, 0.0, 0.0])
    rho_mixed = DensityMatrix.from_probabilities(probs, config)
    results.append(("Mixed", rho_mixed.purity()))
    
    print("\n=== Purity Bounds Test ===")
    print(f"Theoretical bounds: [{1/7:.4f}, 1.0]")
    for name, purity in results:
        assert 1/7 - 1e-6 <= purity <= 1 + 1e-6, f"Purity {purity} out of bounds"
        print(f"{name}: {purity:.4f}")
    print("✓ All within bounds")


def test_ambivalence_vs_uncertainty():
    """
    Core distinction: same probabilities, different coherence.
    
    This is the key empirical test for the quantum formalism.
    """
    basis = EmotionalBasis.ekman()
    
    # Create ambivalent state (coherent superposition)
    rho_ambivalent = basis.ambivalent(["happy", "sad"], [0.5, 0.5])
    
    # Create uncertain state (classical mixture)
    rho_uncertain = basis.uncertain(["happy", "sad"], [0.5, 0.5])
    
    print("\n=== Ambivalence vs Uncertainty Test ===")
    print(f"Ambivalent state:")
    print(f"  Probabilities: P(happy)={rho_ambivalent.probabilities()[0]:.3f}, "
          f"P(sad)={rho_ambivalent.probabilities()[1]:.3f}")
    print(f"  Coherence (l1): {rho_ambivalent.l1_coherence():.4f}")
    print(f"  Entropy: {rho_ambivalent.von_neumann_entropy():.4f}")
    
    print(f"\nUncertain state:")
    print(f"  Probabilities: P(happy)={rho_uncertain.probabilities()[0]:.3f}, "
          f"P(sad)={rho_uncertain.probabilities()[1]:.3f}")
    print(f"  Coherence (l1): {rho_uncertain.l1_coherence():.4f}")
    print(f"  Entropy: {rho_uncertain.von_neumann_entropy():.4f}")
    
    # Key assertion: same probabilities, different coherence
    assert torch.allclose(
        rho_ambivalent.probabilities()[:2], 
        rho_uncertain.probabilities()[:2],
        atol=1e-6
    ), "Probabilities should match"
    
    assert rho_ambivalent.l1_coherence() > rho_uncertain.l1_coherence() + 0.1, \
        "Ambivalent state should have more coherence"
    
    print("\n✓ Successfully distinguished ambivalence from uncertainty")
    print("  Same diagonal probabilities")
    print(f"  Coherence difference: {rho_ambivalent.l1_coherence() - rho_uncertain.l1_coherence():.4f}")


def main():
    print("=" * 60)
    print("Density Matrix Properties Validation")
    print("=" * 60)
    
    test_purity_bounds()
    test_ambivalence_vs_uncertainty()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
