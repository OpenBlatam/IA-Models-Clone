"""
Pipeline Configuration Module
==============================

Configuraciones profesionales para pipelines de deep learning.
Soporta diferentes tipos de modelos y estrategias de entrenamiento.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"


class OptimizerType(str, Enum):
    """Tipos de optimizadores disponibles."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """Tipos de schedulers disponibles."""
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    WARMUP_COSINE = "warmup_cosine"


class LossType(str, Enum):
    """Tipos de funciones de pérdida."""
    MSE = "mse"
    MAE = "mae"
    CROSS_ENTROPY = "cross_entropy"
    HUBER = "huber"
    SMOOTH_L1 = "smooth_l1"


@dataclass
class OptimizerConfig:
    """Configuración de optimizador."""
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuración de scheduler."""
    scheduler_type: SchedulerType = SchedulerType.COSINE
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 100
    eta_min: float = 1e-6
    warmup_steps: int = 1000
    patience: int = 10
    factor: float = 0.5


@dataclass
class TrainingConfig:
    """Configuración completa de entrenamiento."""
    # Data
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Training
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed Precision
    mixed_precision: bool = True
    autocast_dtype: str = "float16"
    
    # Device
    device: str = field(default_factory=lambda: DEFAULT_DEVICE)
    multi_gpu: bool = False
    
    # Optimizer & Scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Loss
    loss_type: LossType = LossType.MSE
    loss_reduction: str = "mean"
    
    # Logging & Checkpointing
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    output_dir: str = "./models"
    
    # Early Stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6
    
    # Experiment Tracking
    use_wandb: bool = False
    wandb_project: str = "robot_movement_ai"
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = False
    tensorboard_log_dir: str = "./logs/tensorboard"
    
    # Validation
    validation_split: float = 0.2
    shuffle: bool = True
    seed: int = 42
    
    # Debugging & Anomaly Detection
    detect_anomaly: bool = False  # Enable autograd anomaly detection (slower but helpful for debugging)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario."""
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.optimizer.learning_rate,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "optimizer_type": self.optimizer.optimizer_type.value,
            "scheduler_type": self.scheduler.scheduler_type.value,
            "loss_type": self.loss_type.value,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Crear configuración desde diccionario."""
        optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType(config_dict.get("optimizer_type", "adamw")),
            learning_rate=config_dict.get("learning_rate", 1e-4),
            weight_decay=config_dict.get("weight_decay", 0.01)
        )
        
        scheduler_config = SchedulerConfig(
            scheduler_type=SchedulerType(config_dict.get("scheduler_type", "cosine")),
            T_max=config_dict.get("num_epochs", 100),
            warmup_steps=config_dict.get("warmup_steps", 1000)
        )
        
        return cls(
            batch_size=config_dict.get("batch_size", 32),
            num_epochs=config_dict.get("num_epochs", 100),
            device=config_dict.get("device", DEFAULT_DEVICE),
            mixed_precision=config_dict.get("mixed_precision", True),
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss_type=LossType(config_dict.get("loss_type", "mse"))
        )


@dataclass
class InferenceConfig:
    """Configuración para inferencia."""
    device: str = field(default_factory=lambda: DEFAULT_DEVICE)
    batch_size: int = 32
    mixed_precision: bool = True
    num_workers: int = 0  # No workers needed for inference
    pin_memory: bool = False
    compile_model: bool = False  # torch.compile for PyTorch 2.0+
    use_jit: bool = False  # TorchScript optimization

