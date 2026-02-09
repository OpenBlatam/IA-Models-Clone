"""
Model Configuration
===================

Configuración de modelos.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ModelConfig:
    """Configuración de modelo."""
    model_type: str
    input_size: int
    output_size: int
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "relu"
    dropout: float = 0.1
    use_batch_norm: bool = False
    bidirectional: bool = False
    use_attention: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "bidirectional": self.bidirectional,
            "use_attention": self.use_attention,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Crear desde diccionario."""
        return cls(**data)


