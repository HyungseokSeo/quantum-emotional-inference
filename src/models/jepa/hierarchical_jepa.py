"""Hierarchical JEPA for emotional inference."""
import torch
import torch.nn as nn
from typing import Optional


class HierarchicalJEPA(nn.Module):
    """
    Hierarchical JEPA for multi-timescale emotional processing.
    
    Combines JEPA's predictive learning with the temporal hierarchy:
    - Fast encoder for reactive processing
    - Slow encoder for adaptive/reflective processing
    - Predictor learns temporal dynamics
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        n_emotions: int = 7,
        predictor_depth: int = 4,
        ema_decay: float = 0.996
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_emotions = n_emotions
        self.ema_decay = ema_decay
        
        # Encoder (context encoder)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # Target encoder (EMA of encoder)
        self.target_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # Predictor
        predictor_layers = []
        for _ in range(predictor_depth):
            predictor_layers.extend([
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            ])
        self.predictor = nn.Sequential(*predictor_layers)
        
        # Initialize target encoder
        self._init_target_encoder()
    
    def _init_target_encoder(self):
        """Initialize target encoder as copy of encoder."""
        for param_q, param_k in zip(self.encoder.parameters(), 
                                    self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder."""
        for param_q, param_k in zip(self.encoder.parameters(),
                                    self.target_encoder.parameters()):
            param_k.data = self.ema_decay * param_k.data + (1 - self.ema_decay) * param_q.data
    
    def forward(
        self,
        context: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            context: Context embedding [B, D]
            target: Target embedding [B, D] (for training)
            
        Returns:
            prediction, target_repr (if target provided)
        """
        # Encode context
        context_repr = self.encoder(context)
        
        # Predict target
        prediction = self.predictor(context_repr)
        
        if target is not None:
            # Get target representation (no gradient)
            with torch.no_grad():
                target_repr = self.target_encoder(target)
            return prediction, target_repr
        
        return prediction
