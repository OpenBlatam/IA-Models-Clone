"""Config Loader"""

def generate_config_loader_code() -> str:
    return '''"""
Config Loader
=============

Cargador de configuración YAML.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuración del modelo."""
    type: str
    vocab_size: int = 50257
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    mixed_precision: bool = True


def load_config(path: str) -> Dict[str, Any]:
    """Carga configuración desde YAML."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return config


def save_config(config: Dict[str, Any], path: str):
    """Guarda configuración en YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to {path}")
'''

