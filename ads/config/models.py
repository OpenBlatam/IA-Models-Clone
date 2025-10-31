"""
Configuration Models for the ads feature.

This module consolidates all configuration dataclasses from config_manager.py,
providing a clean, structured approach to configuration management.
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConfigType(Enum):
    """Types of configuration files."""
    PROJECT = "project"
    MODEL = "model"
    TRAINING = "training"
    DATA = "data"
    EXPERIMENT = "experiment"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    type: str
    architecture: str
    input_size: Union[int, Tuple[int, ...]]
    output_size: Union[int, Tuple[int, ...]]
    hidden_sizes: List[int] = field(default_factory=list)
    dropout_rate: float = 0.1
    activation: str = "relu"
    batch_norm: bool = True
    pretrained: bool = False
    pretrained_path: Optional[str] = None
    freeze_backbone: bool = False
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    # Basic training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    
    # Optimizer settings
    optimizer: str = "adam"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    scheduler: str = "cosine"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Loss function settings
    loss_function: str = "cross_entropy"
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    
    # Advanced training settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    save_best_only: bool = True
    save_frequency: int = 1


@dataclass
class DataConfig:
    """Data processing configuration settings."""
    # Data paths
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    
    # Data processing
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Data augmentation
    augmentation_enabled: bool = True
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data validation
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Data caching
    cache_enabled: bool = False
    cache_path: str = "./cache"
    cache_ttl: int = 3600


@dataclass
class ExperimentConfig:
    """Experiment configuration settings."""
    experiment_name: str = ""
    project_name: str = ""
    description: str = ""
    
    # Tracking
    tracking_backend: str = "local"  # local, mlflow, wandb, tensorboard
    tracking_uri: Optional[str] = None
    tracking_params: Dict[str, Any] = field(default_factory=dict)
    
    # Logging
    log_frequency: int = 10
    checkpoint_frequency: int = 5
    save_artifacts: bool = True
    
    # Versioning
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationConfig:
    """Optimization configuration settings."""
    # Mixed precision
    enable_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"
    
    # Profiling
    enable_profiling: bool = False
    profiling_frequency: int = 100
    profiling_output_path: str = "./profiles"
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Memory optimization
    memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False
    
    # Distributed training
    distributed_enabled: bool = False
    distributed_backend: str = "nccl"
    distributed_world_size: int = 1
    
    # Custom optimizations
    custom_optimizations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    """Deployment configuration settings."""
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Scaling
    auto_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 0.7
    
    # Health checks
    health_check_enabled: bool = True
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    # Monitoring
    monitoring_enabled: bool = True
    metrics_endpoint: str = "/metrics"
    logging_level: str = "INFO"
    
    # Security
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limiting_enabled: bool = True
    rate_limit: int = 100  # requests per minute


@dataclass
class ProjectConfig:
    """Project configuration settings."""
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    license: str = "MIT"
    
    # Project structure
    project_root: str = "."
    config_dir: str = "./configs"
    output_dir: str = "./outputs"
    logs_dir: str = "./logs"
    
    # Dependencies
    python_version: str = "3.8"
    requirements_file: str = "requirements.txt"
    
    # Git integration
    git_enabled: bool = True
    git_remote: Optional[str] = None
    git_branch: str = "main"
    
    # Documentation
    docs_enabled: bool = True
    docs_dir: str = "./docs"
    docs_format: str = "markdown"
    
    # Testing
    testing_enabled: bool = True
    test_dir: str = "./tests"
    coverage_enabled: bool = True
    coverage_threshold: float = 80.0
