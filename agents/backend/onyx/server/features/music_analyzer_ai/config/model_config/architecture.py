"""
Model Architecture Configuration Module

Architecture configuration dataclasses.
"""

from dataclasses import dataclass


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture"""
    model_type: str = "transformer"
    input_dim: int = 169
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    use_pre_norm: bool = True
    max_seq_len: int = 1000



