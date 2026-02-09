"""
Model Configuration - Configuraciones de modelos
=================================================

Configuraciones centralizadas para modelos y entrenamiento.
Sigue mejores prácticas de deep learning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import torch


class OptimizerType(str, Enum):
    """Tipos de optimizadores"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """Tipos de schedulers"""
    COSINE = "cosine"
    STEP = "step"
    PLATEAU = "plateau"
    EXPONENTIAL = "exponential"


class InitializationMethod(str, Enum):
    """Métodos de inicialización"""
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    ORTHOGONAL = "orthogonal"
    NORMAL = "normal"


@dataclass
class OptimizerConfig:
    """Configuración de optimizador"""
    type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9  # Para SGD
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuración de scheduler"""
    type: SchedulerType = SchedulerType.COSINE
    T_max: int = 10  # Para cosine
    step_size: int = 10  # Para step
    gamma: float = 0.1  # Para step/exponential
    factor: float = 0.1  # Para plateau
    patience: int = 10  # Para plateau
    mode: str = "min"  # Para plateau


@dataclass
class TrainingConfig:
    """Configuración completa de entrenamiento"""
    # Model
    model_name: str = ""
    num_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # Scheduler
    scheduler: Optional[SchedulerConfig] = field(default_factory=SchedulerConfig)
    
    # Training settings
    use_mixed_precision: bool = True
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    
    # Device
    device: Optional[torch.device] = None
    use_cudnn_benchmark: bool = True
    
    # Validation
    validation_freq: int = 1  # Validate every N epochs
    save_best_only: bool = True
    
    # Logging
    log_freq: int = 10  # Log every N batches
    save_checkpoint_freq: int = 1  # Save checkpoint every N epochs


@dataclass
class ModelArchitectureConfig:
    """Configuración de arquitectura de modelo"""
    input_size: int
    output_size: int
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.1
    use_batch_norm: bool = True
    initialization: InitializationMethod = InitializationMethod.XAVIER_UNIFORM
    initialization_gain: float = 1.0


@dataclass
class DataConfig:
    """Configuración de datos"""
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle: bool = True
    random_seed: int = 42
    num_workers: int = 0  # DataLoader workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    persistent_workers: bool = False


def get_default_training_config() -> TrainingConfig:
    """Obtener configuración de entrenamiento por defecto"""
    return TrainingConfig()


def get_default_device() -> torch.device:
    """Obtener dispositivo por defecto"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




