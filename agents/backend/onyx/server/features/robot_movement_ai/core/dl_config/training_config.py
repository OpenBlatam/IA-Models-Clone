"""
Training Configuration
=======================

Configuración de entrenamiento.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    use_mixed_precision: bool = False
    use_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "adam"
    scheduler_type: str = "plateau"
    checkpoint_frequency: int = 10
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "early_stopping_patience": self.early_stopping_patience,
            "validation_split": self.validation_split,
            "use_mixed_precision": self.use_mixed_precision,
            "use_gradient_accumulation": self.use_gradient_accumulation,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_type": self.optimizer_type,
            "scheduler_type": self.scheduler_type,
            "checkpoint_frequency": self.checkpoint_frequency,
            "device": self.device,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Crear desde diccionario."""
        return cls(**data)


