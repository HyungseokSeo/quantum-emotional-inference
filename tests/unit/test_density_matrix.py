"""Tests for density matrix implementation."""
import pytest
import torch
import numpy as np

from src.quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig
from src.core.types import StateType


@pytest.fixture
def config():
    return DensityMatrixConfig(n_dims=3)


class TestDensityMatrixConstruction:
    """Test density matrix construction methods."""
    
    def test_from_pure_state(self, config):
        """Pure state should have purity 1."""
        psi = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex64)
        rho = DensityMatrix.from_pure_state(psi, config)
        
        assert abs(rho.purity() - 1.0) < 1e-6
        assert rho.state_type() == StateType.PURE
    
    def test_from_probabilities(self, config):
        """Diagonal matrix should have zero coherence."""
        probs = torch.tensor([0.5, 0.3, 0.2])
        rho = DensityMatrix.from_probabilities(probs, config)
        
        assert abs(rho.l1_coherence()) < 1e-6
        assert rho.state_type() == StateType.MIXED
    
    def test_maximally_mixed(self, config):
        """Maximally mixed state should have minimal purity."""
        rho = DensityMatrix.maximally_mixed(config)
        
        expected_purity = 1.0 / config.n_dims
        assert abs(rho.purity() - expected_purity) < 1e-6
        assert rho.state_type() == StateType.MAXIMALLY_MIXED


class TestDensityMatrixProperties:
    """Test physical properties."""
    
    def test_hermiticity(self, config):
        """Density matrix should be Hermitian."""
        psi = torch.tensor([1.0, 1.0, 0.0], dtype=torch.complex64)
        rho = DensityMatrix.from_pure_state(psi, config)
        
        assert torch.allclose(rho.matrix, rho.matrix.conj().T)
    
    def test_unit_trace(self, config):
        """Trace should be 1."""
        psi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.complex64)
        rho = DensityMatrix.from_pure_state(psi, config)
        
        trace = torch.trace(rho.matrix).real
        assert abs(trace - 1.0) < 1e-6
    
    def test_positive_semidefinite(self, config):
        """Eigenvalues should be non-negative."""
        psi = torch.tensor([1.0, 2.0, 1.0], dtype=torch.complex64)
        rho = DensityMatrix.from_pure_state(psi, config)
        
        eigenvalues = torch.linalg.eigvalsh(rho.matrix)
        assert torch.all(eigenvalues >= -1e-6)


class TestCoherence:
    """Test coherence measures."""
    
    def test_pure_state_coherence(self, config):
        """Pure superposition should have coherence."""
        psi = torch.tensor([1.0, 1.0, 0.0], dtype=torch.complex64)
        rho = DensityMatrix.from_pure_state(psi, config)
        
        assert rho.l1_coherence() > 0
    
    def test_mixed_state_no_coherence(self, config):
        """Classical mixture should have zero coherence."""
        probs = torch.tensor([0.5, 0.5, 0.0])
        rho = DensityMatrix.from_probabilities(probs, config)
        
        assert abs(rho.l1_coherence()) < 1e-6
    
    def test_same_diagonal_different_coherence(self, config):
        """Ambivalent vs uncertain states with same probabilities."""
        # Pure superposition (ambivalent)
        psi = torch.tensor([1.0, 1.0, 0.0], dtype=torch.complex64)
        rho_ambivalent = DensityMatrix.from_pure_state(psi, config)
        
        # Classical mixture (uncertain)
        probs = rho_ambivalent.probabilities()
        rho_uncertain = DensityMatrix.from_probabilities(probs, config)
        
        # Same diagonal probabilities
        assert torch.allclose(
            rho_ambivalent.probabilities(), 
            rho_uncertain.probabilities(),
            atol=1e-6
        )
        
        # Different coherence
        assert rho_ambivalent.l1_coherence() > rho_uncertain.l1_coherence() + 0.1


class TestEntropy:
    """Test entropy measures."""
    
    def test_pure_state_zero_entropy(self, config):
        """Pure state should have zero von Neumann entropy."""
        psi = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex64)
        rho = DensityMatrix.from_pure_state(psi, config)
        
        assert abs(rho.von_neumann_entropy()) < 1e-6
    
    def test_maximally_mixed_max_entropy(self, config):
        """Maximally mixed state should have maximum entropy."""
        rho = DensityMatrix.maximally_mixed(config)
        
        max_entropy = np.log(config.n_dims)
        assert abs(rho.von_neumann_entropy() - max_entropy) < 1e-6
