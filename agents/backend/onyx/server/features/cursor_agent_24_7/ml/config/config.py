"""
ML Config - Configuración de ML
================================

Gestión de configuraciones YAML para modelos y entrenamiento.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MLConfig:
    """Configuración de Machine Learning"""
    # Model config
    model_name: str = "gpt2"
    model_type: str = "causal"  # causal, seq2seq
    max_length: int = 512
    
    # Training config
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 8
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    
    # Data config
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 0
    
    # Device config
    device: str = "auto"  # auto, cpu, cuda
    use_cuda: bool = True
    
    # Output config
    output_dir: str = "./checkpoints"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLConfig':
        """Crear desde diccionario"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return asdict(self)
    
    def save(self, path: str):
        """Guardar configuración"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MLConfig':
        """Cargar configuración"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(path: str) -> MLConfig:
    """Cargar configuración desde archivo YAML"""
    return MLConfig.load(path)


def save_config(config: MLConfig, path: str):
    """Guardar configuración a archivo YAML"""
    config.save(path)



