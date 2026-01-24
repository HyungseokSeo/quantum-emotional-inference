"""
Configuration management for the framework.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
import torch


@dataclass
class QuantumConfig:
    """Configuration for quantum formalism components."""
    n_emotions: int = 7  # Ekman basic emotions + neutral
    dtype: str = "complex64"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tolerance: float = 1e-6
    
    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)


@dataclass
class HierarchyConfig:
    """Configuration for hierarchical temporal processing."""
    # Timescales (in seconds)
    reactive_timescale: float = 0.1    # ~100ms
    adaptive_timescale: float = 1.0    # ~1s
    reflective_timescale: float = 10.0  # ~10s
    
    # Decoherence rates
    gamma_reactive_to_adaptive: float = 1.0
    gamma_adaptive_to_reflective: float = 0.5
    
    # Hidden dimensions
    reactive_dim: int = 256
    adaptive_dim: int = 128
    reflective_dim: int = 64


@dataclass
class FreeEnergyConfig:
    """Configuration for Free Energy dynamics."""
    temperature: float = 1.0
    use_quantum_entropy: bool = True
    
    # Valence/arousal computation
    history_length: int = 10
    smoothing_factor: float = 0.9


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    # Encoder
    encoder_type: str = "resnet18"
    encoder_pretrained: bool = True
    
    # JEPA
    jepa_embed_dim: int = 512
    jepa_predictor_depth: int = 4
    jepa_ema_decay: float = 0.996
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100


@dataclass
class Config:
    """Main configuration container."""
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    free_energy: FreeEnergyConfig = field(default_factory=FreeEnergyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    
    # Experiment
    experiment_name: str = "default"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        quantum = QuantumConfig(**config_dict.get('quantum', {}))
        hierarchy = HierarchyConfig(**config_dict.get('hierarchy', {}))
        free_energy = FreeEnergyConfig(**config_dict.get('free_energy', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        
        return cls(
            quantum=quantum,
            hierarchy=hierarchy,
            free_energy=free_energy,
            model=model,
            data_dir=Path(config_dict.get('data_dir', 'data')),
            output_dir=Path(config_dict.get('output_dir', 'outputs')),
            checkpoint_dir=Path(config_dict.get('checkpoint_dir', 'checkpoints')),
            experiment_name=config_dict.get('experiment_name', 'default'),
            seed=config_dict.get('seed', 42),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'quantum': self.quantum.__dict__,
            'hierarchy': self.hierarchy.__dict__,
            'free_energy': self.free_energy.__dict__,
            'model': self.model.__dict__,
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'checkpoint_dir': str(self.checkpoint_dir),
            'experiment_name': self.experiment_name,
            'seed': self.seed,
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
