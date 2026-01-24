"""Tests for coherence measures."""
import pytest
import torch

from src.quantum.information.coherence import l1_coherence, relative_entropy_of_coherence


class TestCoherenceMeasures:
    """Test coherence quantification."""
    
    def test_diagonal_state_zero_coherence(self):
        """Diagonal state should have zero coherence."""
        rho = torch.diag(torch.tensor([0.5, 0.5], dtype=torch.complex64))
        assert abs(l1_coherence(rho)) < 1e-6
    
    def test_maximally_coherent_state(self):
        """Pure superposition should have maximum coherence."""
        psi = torch.tensor([1.0, 1.0], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
        rho = torch.outer(psi, psi.conj())
        
        # l1 coherence for 2x2 maximally coherent: 2 * |off-diagonal| = 2 * 0.5 = 1.0
        assert abs(l1_coherence(rho) - 1.0) < 1e-6
