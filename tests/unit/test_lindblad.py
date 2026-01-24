"""Tests for Lindblad dynamics."""
import pytest
import torch

from src.quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig
from src.quantum.dynamics.lindblad import LindbladDynamics


@pytest.fixture
def config():
    return DensityMatrixConfig(n_dims=2)


class TestLindbladDynamics:
    """Test Lindblad master equation."""
    
    def test_dephasing_destroys_coherence(self, config):
        """Dephasing should reduce off-diagonal elements."""
        dynamics = LindbladDynamics.dephasing(2, rate=1.0)
        
        # Start with superposition
        psi = torch.tensor([1.0, 1.0], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
        rho_init = DensityMatrix.from_pure_state(psi, config)
        
        # Evolve
        t_span = torch.linspace(0, 5, 50)
        trajectory = dynamics.evolve(rho_init, t_span)
        
        # Coherence should decrease
        assert trajectory[-1].l1_coherence() < trajectory[0].l1_coherence()
    
    def test_preserves_trace(self, config):
        """Evolution should preserve unit trace."""
        dynamics = LindbladDynamics.dephasing(2, rate=1.0)
        
        psi = torch.tensor([1.0, 1.0], dtype=torch.complex64)
        rho_init = DensityMatrix.from_pure_state(psi, config)
        
        t_span = torch.linspace(0, 2, 20)
        trajectory = dynamics.evolve(rho_init, t_span)
        
        for rho in trajectory:
            trace = torch.trace(rho.matrix).real.item()
            assert abs(trace - 1.0) < 1e-5
    
    def test_preserves_positivity(self, config):
        """Evolution should preserve positive semi-definiteness."""
        dynamics = LindbladDynamics.dephasing(2, rate=1.0)
        
        psi = torch.tensor([1.0, 1.0], dtype=torch.complex64)
        rho_init = DensityMatrix.from_pure_state(psi, config)
        
        t_span = torch.linspace(0, 2, 20)
        trajectory = dynamics.evolve(rho_init, t_span)
        
        for rho in trajectory:
            eigenvalues = torch.linalg.eigvalsh(rho.matrix)
            assert torch.all(eigenvalues >= -1e-6)
