"""
Density matrix encoder: maps input features to valid density matrix.

Key architectural contribution: Learn to produce ρ directly,
not just probabilities. This preserves the distinction between
ambivalence (coherent) and uncertainty (mixed).
"""
import torch
import torch.nn as nn
from typing import Optional

from ....quantum.states.density_matrix import DensityMatrix, DensityMatrixConfig


class DensityEncoder(nn.Module):
    """
    Neural network that outputs valid density matrices.
    
    Architecture:
    1. Feature extraction (backbone)
    2. Map to Cholesky factor L
    3. Construct ρ = LL† / Tr(LL†)
    
    This guarantees ρ is positive semi-definite and unit trace.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_emotions: int,
        hidden_dim: int = 256,
        backbone: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.n_emotions = n_emotions
        self.config = DensityMatrixConfig(n_dims=n_emotions)
        
        # Backbone (feature extraction)
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        
        # Output Cholesky factor (lower triangular, real + imaginary parts)
        n_params = n_emotions * (n_emotions + 1)  # Real + imaginary for lower triangle
        self.cholesky_head = nn.Linear(hidden_dim, n_params)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input → density matrix.
        
        Returns:
            Batch of density matrices [B, n, n]
        """
        features = self.backbone(x)
        cholesky_params = self.cholesky_head(features)
        
        batch_size = x.shape[0]
        rho = self._params_to_density_matrix(cholesky_params, batch_size)
        
        return rho
    
    def _params_to_density_matrix(
        self,
        params: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Convert parameters to valid density matrix."""
        n = self.n_emotions
        
        # Split into real and imaginary parts
        n_lower = n * (n + 1) // 2
        real_params = params[:, :n_lower]
        imag_params = params[:, n_lower:]
        
        # Construct lower triangular Cholesky factor
        L = torch.zeros(batch_size, n, n, dtype=torch.complex64, device=params.device)
        
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: real and positive (use exp)
                    L[:, i, j] = torch.exp(real_params[:, idx])
                else:
                    # Off-diagonal: complex
                    L[:, i, j] = real_params[:, idx] + 1j * imag_params[:, idx - n]
                idx += 1
        
        # ρ = LL† / Tr(LL†)
        rho = L @ L.conj().transpose(-1, -2)
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        rho = rho / trace.real
        
        return rho
    
    def to_density_matrix_objects(
        self,
        rho_batch: torch.Tensor
    ) -> list:
        """Convert batch tensor to list of DensityMatrix objects."""
        return [
            DensityMatrix(rho_batch[i], self.config, validate=False)
            for i in range(rho_batch.shape[0])
        ]
