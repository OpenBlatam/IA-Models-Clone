"""
Configuration Schema
====================

Esquemas de configuración para validación.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfigSchema:
    """Esquema de configuración de modelo."""
    model_type: str = "mlp"  # mlp, gcn, gat, transformer
    input_dim: int = 20
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 128])
    output_dim: int = 4
    dropout: float = 0.2
    activation: str = "relu"
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    use_attention: bool = True
    device: Optional[str] = None


@dataclass
class TrainingConfigSchema:
    """Esquema de configuración de entrenamiento."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: Optional[str] = "reduce_on_plateau"
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10


@dataclass
class DataConfigSchema:
    """Esquema de configuración de datos."""
    data_dir: str = "./data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "scaler_type": "standard",
        "normalize_features": True,
        "normalize_targets": True
    })


