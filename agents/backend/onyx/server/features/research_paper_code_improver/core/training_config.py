"""
Training Configuration Manager - Gestor de configuración de entrenamiento
===========================================================================
"""

import logging
import yaml
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuración completa de entrenamiento"""
    # Model
    model_name: str = "default_model"
    model_type: str = "transformer"
    
    # Training
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler
    scheduler: str = "cosine"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 0
    
    # Loss
    loss_function: str = "cross_entropy"
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    num_workers: int = 4
    pin_memory: bool = True
    
    # Mixed Precision
    use_mixed_precision: bool = True
    mixed_precision_mode: str = "mixed"
    
    # Early Stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 1
    max_checkpoints: int = 5
    
    # Logging
    log_dir: str = "./logs"
    log_interval: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = ""
    
    # Device
    device: str = "cuda"
    num_gpus: int = 1
    
    # Seed
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Crea desde diccionario"""
        return cls(**config_dict)


class ConfigManager:
    """Gestor de configuraciones"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[TrainingConfig] = None
    
    def load_config(self, config_path: Optional[str] = None) -> TrainingConfig:
        """Carga configuración desde archivo"""
        path = config_path or self.config_path
        if not path:
            raise ValueError("No se proporcionó ruta de configuración")
        
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Formato de archivo no soportado: {path.suffix}")
        
        self.config = TrainingConfig.from_dict(config_dict)
        logger.info(f"Configuración cargada desde {path}")
        return self.config
    
    def save_config(
        self,
        config: TrainingConfig,
        save_path: str,
        format: str = "yaml"
    ):
        """Guarda configuración a archivo"""
        path = Path(save_path)
        config_dict = config.to_dict()
        
        if format == "yaml" or path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif format == "json" or path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Configuración guardada en {path}")
    
    def create_default_config(self) -> TrainingConfig:
        """Crea configuración por defecto"""
        return TrainingConfig()
    
    def update_config(
        self,
        config: TrainingConfig,
        updates: Dict[str, Any]
    ) -> TrainingConfig:
        """Actualiza configuración"""
        config_dict = config.to_dict()
        config_dict.update(updates)
        return TrainingConfig.from_dict(config_dict)
    
    def validate_config(self, config: TrainingConfig) -> List[str]:
        """Valida configuración"""
        errors = []
        
        if config.num_epochs <= 0:
            errors.append("num_epochs debe ser mayor que 0")
        
        if config.batch_size <= 0:
            errors.append("batch_size debe ser mayor que 0")
        
        if config.learning_rate <= 0:
            errors.append("learning_rate debe ser mayor que 0")
        
        if config.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps debe ser al menos 1")
        
        if not config.train_data_path:
            errors.append("train_data_path es requerido")
        
        return errors




