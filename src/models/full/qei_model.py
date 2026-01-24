"""Full Quantum Emotional Inference model."""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from ..encoders.quantum_inspired.density_encoder import DensityEncoder
from ..decoders.collapse_decoder import CollapseDecoder
from ...hierarchy.transitions.decoherence_cascade import DecoherenceCascade


class QEIModel(nn.Module):
    """
    Full Quantum Emotional Inference model.
    
    Pipeline:
    1. Encode input → density matrix
    2. Process through decoherence cascade
    3. Decode to emotion prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        n_emotions: int = 7,
        hidden_dim: int = 256,
        use_cascade: bool = True
    ):
        super().__init__()
        
        self.n_emotions = n_emotions
        self.use_cascade = use_cascade
        
        # Encoder: input → ρ
        self.encoder = DensityEncoder(
            input_dim=input_dim,
            n_emotions=n_emotions,
            hidden_dim=hidden_dim
        )
        
        # Decoherence cascade
        if use_cascade:
            self.cascade = DecoherenceCascade.default(n_emotions)
        
        # Decoder: ρ → prediction
        self.decoder = CollapseDecoder(n_emotions)
    
    def forward(
        self,
        x: torch.Tensor,
        return_density_matrix: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [B, input_dim]
            return_density_matrix: Whether to return intermediate ρ
            
        Returns:
            Dict with predictions and optionally density matrix
        """
        # Encode to density matrix
        rho = self.encoder(x)
        
        # Apply cascade (if enabled)
        # Note: batch processing of cascade would need vectorization
        # For now, return encoded ρ directly
        
        # Decode to probabilities
        probs = self.decoder(rho)
        
        output = {
            "probabilities": probs,
            "predicted": torch.argmax(probs, dim=-1),
        }
        
        if return_density_matrix:
            output["density_matrix"] = rho
            output["coherence"] = self._batch_coherence(rho)
        
        return output
    
    def _batch_coherence(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute l1 coherence for batch."""
        diag = torch.diagonal(rho, dim1=-2, dim2=-1)
        diag_matrix = torch.diag_embed(diag)
        off_diag = rho - diag_matrix
        return torch.sum(torch.abs(off_diag), dim=(-1, -2))
