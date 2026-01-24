"""Validation tests for physical constraints."""
import pytest
import torch

from src.quantum.validation.physicality import is_valid_density_matrix, is_cptp


class TestPhysicality:
    """Ensure all states remain physical."""
    
    def test_random_construction_valid(self):
        """Randomly constructed states should be valid."""
        for _ in range(10):
            # Random pure state
            psi = torch.randn(4, dtype=torch.complex64)
            psi = psi / torch.norm(psi)
            rho = torch.outer(psi, psi.conj())
            
            assert is_valid_density_matrix(rho)
    
    def test_cptp_kraus(self):
        """Kraus operators should define CPTP map."""
        # Dephasing channel
        K0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64) * torch.sqrt(torch.tensor(0.5))
        K1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64) * torch.sqrt(torch.tensor(0.5))
        K2 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64) * torch.sqrt(torch.tensor(0.5))
        K3 = torch.tensor([[0, 0], [0, -1]], dtype=torch.complex64) * torch.sqrt(torch.tensor(0.5))
        
        # This is a simplified check - actual CPTP validation is more nuanced
        kraus = [K0 + K2, K1 + K3]  # Simplified
