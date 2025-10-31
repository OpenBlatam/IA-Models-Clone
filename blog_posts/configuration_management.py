from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, TypeVar
from dataclasses import dataclass, asdict, field
from enum import Enum
import structlog
from pydantic import BaseModel, Field, validator, root_validator
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Management System
===============================

This module provides a comprehensive configuration management system using YAML files
for hyperparameters and model settings. It includes validation, inheritance,
environment-specific configurations, and best practices for production systems.

Key Features:
1. YAML-based configuration files
2. Configuration validation with Pydantic
3. Environment-specific configurations
4. Configuration inheritance and overrides
5. Type-safe configuration access
6. Configuration templates and examples
"""


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

T = TypeVar('T', bound='BaseConfig')


# =============================================================================
# CONFIGURATION ENUMS AND CONSTANTS
# =============================================================================

class ModelType(str, Enum):
    """Supported model types."""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    DIFFUSION = "diffusion"
    LSTM = "lstm"
    GRU = "gru"
    MLP = "mlp"
    CUSTOM = "custom"


class OptimizerType(str, Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSprop = "rmsprop"
    ADAGRAD = "adagrad"
    LION = "lion"


class SchedulerType(str, Enum):
    """Supported scheduler types."""
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    ONE_CYCLE = "one_cycle"
    WARMUP_COSINE = "warmup_cosine"


class LossType(str, Enum):
    """Supported loss function types."""
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    BCE = "bce"
    BCE_WITH_LOGITS = "bce_with_logits"
    KL_DIVERGENCE = "kl_divergence"
    L1 = "l1"
    SMOOTH_L1 = "smooth_l1"
    FOCAL = "focal"
    DICE = "dice"


class DeviceType(str, Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


# =============================================================================
# BASE CONFIGURATION CLASSES
# =============================================================================

class BaseConfig(BaseModel):
    """Base configuration class with common functionality."""
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"  # Prevent additional fields
        use_enum_values = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    def save(self, filepath: str):
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info("Configuration saved", filepath=str(filepath))
    
    @classmethod
    def load(cls: Type[T], filepath: str) -> T:
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = yaml.safe_load(f)
        
        config = cls(**data)
        logger.info("Configuration loaded", filepath=str(filepath))
        return config
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """Merge with another configuration, with other taking precedence."""
        merged_dict = {**self.to_dict(), **other.to_dict()}
        return self.__class__(**merged_dict)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelConfig(BaseConfig):
    """Configuration for model architecture and parameters."""
    
    # Basic model information
    model_type: ModelType = Field(..., description="Type of model architecture")
    model_name: str = Field(..., description="Name or identifier of the model")
    
    # Model architecture parameters
    num_classes: Optional[int] = Field(None, description="Number of output classes")
    input_dim: Optional[int] = Field(None, description="Input dimension")
    hidden_dim: Optional[int] = Field(None, description="Hidden dimension size")
    num_layers: Optional[int] = Field(None, description="Number of layers")
    num_heads: Optional[int] = Field(None, description="Number of attention heads")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")
    activation: str = Field("relu", description="Activation function")
    
    # Pretrained model settings
    pretrained: bool = Field(True, description="Use pretrained model")
    freeze_backbone: bool = Field(False, description="Freeze backbone layers")
    pretrained_path: Optional[str] = Field(None, description="Path to pretrained weights")
    
    # Model-specific parameters
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")
    
    # Validation
    @validator('model_type')
    def validate_model_type(cls, v) -> bool:
        """Validate model type."""
        if v not in ModelType:
            raise ValueError(f"Invalid model type: {v}")
        return v
    
    @validator('dropout')
    def validate_dropout(cls, v) -> bool:
        """Validate dropout rate."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Dropout must be between 0 and 1, got {v}")
        return v
    
    @root_validator
    def validate_model_params(cls, values) -> bool:
        """Validate model-specific parameters."""
        model_type = values.get('model_type')
        model_params = values.get('model_params', {})
        
        if model_type == ModelType.TRANSFORMER:
            if 'max_length' in model_params and model_params['max_length'] <= 0:
                raise ValueError("max_length must be positive for transformer models")
        
        elif model_type == ModelType.CNN:
            if 'kernel_size' in model_params and model_params['kernel_size'] <= 0:
                raise ValueError("kernel_size must be positive for CNN models")
        
        return values


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_path: str = Field(..., description="Path to the dataset")
    train_data_path: Optional[str] = Field(None, description="Path to training data")
    val_data_path: Optional[str] = Field(None, description="Path to validation data")
    test_data_path: Optional[str] = Field(None, description="Path to test data")
    
    # Data loading parameters
    batch_size: int = Field(32, gt=0, description="Batch size for training")
    val_batch_size: Optional[int] = Field(None, description="Batch size for validation")
    test_batch_size: Optional[int] = Field(None, description="Batch size for testing")
    num_workers: int = Field(4, ge=0, description="Number of data loading workers")
    pin_memory: bool = Field(True, description="Pin memory for faster GPU transfer")
    shuffle: bool = Field(True, description="Shuffle training data")
    
    # Data splits
    train_split: float = Field(0.8, gt=0.0, lt=1.0, description="Training data split ratio")
    val_split: float = Field(0.1, gt=0.0, lt=1.0, description="Validation data split ratio")
    test_split: float = Field(0.1, gt=0.0, lt=1.0, description="Test data split ratio")
    
    # Column specifications
    target_column: Optional[str] = Field(None, description="Target column name")
    feature_columns: Optional[List[str]] = Field(None, description="Feature column names")
    text_column: Optional[str] = Field(None, description="Text column for NLP tasks")
    image_column: Optional[str] = Field(None, description="Image column for CV tasks")
    
    # Data preprocessing
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="Preprocessing configuration")
    augmentations: Dict[str, Any] = Field(default_factory=dict, description="Data augmentation configuration")
    
    # Validation
    @validator('val_batch_size', 'test_batch_size', pre=True, always=True)
    def set_default_batch_sizes(cls, v, values) -> Any:
        """Set default batch sizes if not specified."""
        if v is None:
            return values.get('batch_size', 32)
        return v
    
    @root_validator
    def validate_data_splits(cls, values) -> bool:
        """Validate data split ratios sum to 1."""
        train_split = values.get('train_split', 0.8)
        val_split = values.get('val_split', 0.1)
        test_split = values.get('test_split', 0.1)
        
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total}")
        
        return values
    
    @validator('data_path', 'train_data_path', 'val_data_path', 'test_data_path')
    def validate_data_paths(cls, v) -> bool:
        """Validate data paths exist if provided."""
        if v and not Path(v).exists():
            logger.warning(f"Data path does not exist: {v}")
        return v


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class TrainingConfig(BaseConfig):
    """Configuration for training process."""
    
    # Training parameters
    epochs: int = Field(100, gt=0, description="Number of training epochs")
    learning_rate: float = Field(0.001, gt=0.0, description="Learning rate")
    weight_decay: float = Field(1e-5, ge=0.0, description="Weight decay")
    
    # Optimizer configuration
    optimizer: OptimizerType = Field(OptimizerType.ADAM, description="Optimizer type")
    optimizer_params: Dict[str, Any] = Field(default_factory=dict, description="Optimizer-specific parameters")
    
    # Scheduler configuration
    scheduler: SchedulerType = Field(SchedulerType.COSINE, description="Learning rate scheduler")
    scheduler_params: Dict[str, Any] = Field(default_factory=dict, description="Scheduler-specific parameters")
    
    # Loss function configuration
    loss_function: LossType = Field(LossType.CROSS_ENTROPY, description="Loss function type")
    loss_params: Dict[str, Any] = Field(default_factory=dict, description="Loss function parameters")
    
    # Device and performance
    device: DeviceType = Field(DeviceType.AUTO, description="Device to use for training")
    mixed_precision: bool = Field(True, description="Use mixed precision training")
    gradient_clipping: float = Field(1.0, ge=0.0, description="Gradient clipping norm")
    
    # Training control
    early_stopping_patience: int = Field(10, ge=0, description="Early stopping patience")
    save_best_model: bool = Field(True, description="Save best model during training")
    save_checkpoints: bool = Field(True, description="Save training checkpoints")
    checkpoint_frequency: int = Field(5, gt=0, description="Checkpoint save frequency")
    
    # Output configuration
    output_dir: str = Field("outputs", description="Output directory for results")
    checkpoint_dir: str = Field("checkpoints", description="Checkpoint directory")
    log_dir: str = Field("logs", description="Logging directory")
    
    # Experiment tracking
    experiment_name: Optional[str] = Field(None, description="Experiment name for tracking")
    use_wandb: bool = Field(False, description="Use Weights & Biases for tracking")
    use_tensorboard: bool = Field(True, description="Use TensorBoard for tracking")
    
    # Validation
    @validator('device', pre=True)
    def validate_device(cls, v) -> bool:
        """Validate and set device."""
        if v == DeviceType.AUTO:
            if torch.cuda.is_available():
                return DeviceType.CUDA
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return DeviceType.MPS
            else:
                return DeviceType.CPU
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v) -> bool:
        """Validate learning rate."""
        if v <= 0:
            raise ValueError(f"Learning rate must be positive, got {v}")
        return v
    
    @root_validator
    def validate_training_params(cls, values) -> bool:
        """Validate training-specific parameters."""
        scheduler = values.get('scheduler')
        scheduler_params = values.get('scheduler_params', {})
        
        if scheduler == SchedulerType.ONE_CYCLE:
            if 'max_lr' not in scheduler_params:
                raise ValueError("max_lr is required for OneCycleLR scheduler")
        
        return values


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

class EvaluationConfig(BaseConfig):
    """Configuration for model evaluation."""
    
    # Evaluation metrics
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"], description="Metrics to calculate")
    save_predictions: bool = Field(True, description="Save model predictions")
    save_plots: bool = Field(True, description="Save evaluation plots")
    
    # Output configuration
    output_dir: str = Field("evaluation_results", description="Evaluation output directory")
    results_file: str = Field("results.json", description="Results file name")
    
    # Threshold configuration
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    thresholds: Optional[List[float]] = Field(None, description="Multiple thresholds for evaluation")
    
    # Visualization configuration
    plot_config: Dict[str, Any] = Field(default_factory=dict, description="Plot configuration")
    
    # Validation
    @validator('metrics')
    def validate_metrics(cls, v) -> bool:
        """Validate evaluation metrics."""
        valid_metrics = {
            "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc",
            "mse", "rmse", "mae", "r2_score", "log_loss", "confusion_matrix"
        }
        
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics: {valid_metrics}")
        
        return v
    
    @validator('threshold')
    def validate_threshold(cls, v) -> bool:
        """Validate threshold value."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {v}")
        return v


# =============================================================================
# COMPLETE EXPERIMENT CONFIGURATION
# =============================================================================

class ExperimentConfig(BaseConfig):
    """Complete experiment configuration combining all components."""
    
    # Experiment metadata
    experiment_name: str = Field(..., description="Name of the experiment")
    experiment_description: Optional[str] = Field(None, description="Experiment description")
    version: str = Field("1.0.0", description="Experiment version")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")
    
    # Component configurations
    model: ModelConfig = Field(..., description="Model configuration")
    data: DataConfig = Field(..., description="Data configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration")
    
    # Environment configuration
    environment: str = Field("development", description="Environment (development, staging, production)")
    seed: int = Field(42, description="Random seed for reproducibility")
    debug: bool = Field(False, description="Enable debug mode")
    
    # Validation
    @validator('experiment_name')
    def validate_experiment_name(cls, v) -> bool:
        """Validate experiment name."""
        if not v or not v.strip():
            raise ValueError("Experiment name cannot be empty")
        return v.strip()
    
    @validator('environment')
    def validate_environment(cls, v) -> bool:
        """Validate environment."""
        valid_environments = ["development", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Invalid environment: {v}. Valid environments: {valid_environments}")
        return v
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get all output paths for the experiment."""
        base_path = Path(self.training.output_dir) / self.experiment_name
        
        return {
            'base': base_path,
            'checkpoints': base_path / self.training.checkpoint_dir,
            'logs': base_path / self.training.log_dir,
            'evaluation': base_path / self.evaluation.output_dir,
            'models': base_path / "models",
            'plots': base_path / "plots",
            'configs': base_path / "configs"
        }
    
    def create_output_directories(self) -> Any:
        """Create all output directories."""
        paths = self.get_output_paths()
        
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Output directories created", paths=list(paths.keys()))


# =============================================================================
# CONFIGURATION TEMPLATES
# =============================================================================

class ConfigurationTemplates:
    """Templates for common configuration scenarios."""
    
    @staticmethod
    def get_transformer_classification_template() -> ExperimentConfig:
        """Get template for transformer-based classification."""
        return ExperimentConfig(
            experiment_name="transformer_classification",
            experiment_description="Transformer-based text classification",
            model=ModelConfig(
                model_type=ModelType.TRANSFORMER,
                model_name="bert-base-uncased",
                num_classes=10,
                hidden_dim=512,
                dropout=0.1,
                model_params={"max_length": 512}
            ),
            data=DataConfig(
                data_path="data/dataset.csv",
                batch_size=16,
                target_column="label",
                text_column="text"
            ),
            training=TrainingConfig(
                epochs=10,
                learning_rate=2e-5,
                optimizer=OptimizerType.ADAMW,
                scheduler=SchedulerType.WARMUP_COSINE,
                loss_function=LossType.CROSS_ENTROPY
            ),
            evaluation=EvaluationConfig(
                metrics=["accuracy", "precision", "recall", "f1_score"]
            )
        )
    
    @staticmethod
    def get_cnn_image_classification_template() -> ExperimentConfig:
        """Get template for CNN-based image classification."""
        return ExperimentConfig(
            experiment_name="cnn_image_classification",
            experiment_description="CNN-based image classification",
            model=ModelConfig(
                model_type=ModelType.CNN,
                model_name="resnet50",
                num_classes=1000,
                hidden_dim=512,
                dropout=0.2,
                pretrained=True,
                model_params={"input_channels": 3}
            ),
            data=DataConfig(
                data_path="data/images/",
                batch_size=32,
                target_column="class",
                augmentations={
                    "horizontal_flip": True,
                    "random_rotation": 10,
                    "color_jitter": {"brightness": 0.2, "contrast": 0.2}
                }
            ),
            training=TrainingConfig(
                epochs=50,
                learning_rate=0.001,
                optimizer=OptimizerType.ADAM,
                scheduler=SchedulerType.COSINE,
                loss_function=LossType.CROSS_ENTROPY
            ),
            evaluation=EvaluationConfig(
                metrics=["accuracy", "precision", "recall", "f1_score"]
            )
        )
    
    @staticmethod
    def get_diffusion_generation_template() -> ExperimentConfig:
        """Get template for diffusion model generation."""
        return ExperimentConfig(
            experiment_name="diffusion_generation",
            experiment_description="Diffusion model for image generation",
            model=ModelConfig(
                model_type=ModelType.DIFFUSION,
                model_name="stable-diffusion-v1-5",
                model_params={
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "image_size": 512
                }
            ),
            data=DataConfig(
                data_path="data/images/",
                batch_size=1,
                target_column=None
            ),
            training=TrainingConfig(
                epochs=100,
                learning_rate=1e-4,
                optimizer=OptimizerType.ADAMW,
                scheduler=SchedulerType.COSINE,
                loss_function=LossType.MSE
            ),
            evaluation=EvaluationConfig(
                metrics=["fid", "lpips", "ssim"],
                save_predictions=True,
                save_plots=True
            )
        )


# =============================================================================
# CONFIGURATION MANAGER
# =============================================================================

class ConfigurationManager:
    """Manager for handling configuration files and operations."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger(__name__)
    
    def save_config(self, config: ExperimentConfig, filename: Optional[str] = None):
        """Save configuration to file."""
        if filename is None:
            filename = f"{config.experiment_name}.yaml"
        
        filepath = self.config_dir / filename
        config.save(str(filepath))
        
        self.logger.info("Configuration saved", filepath=str(filepath))
        return filepath
    
    def load_config(self, filename: str) -> ExperimentConfig:
        """Load configuration from file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        config = ExperimentConfig.load(str(filepath))
        self.logger.info("Configuration loaded", filepath=str(filepath))
        return config
    
    def list_configs(self) -> List[str]:
        """List all available configuration files."""
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.name for f in config_files]
    
    def create_template(self, template_name: str, experiment_name: str) -> ExperimentConfig:
        """Create configuration from template."""
        templates = {
            "transformer_classification": ConfigurationTemplates.get_transformer_classification_template,
            "cnn_image_classification": ConfigurationTemplates.get_cnn_image_classification_template,
            "diffusion_generation": ConfigurationTemplates.get_diffusion_generation_template
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
        
        config = templates[template_name]()
        config.experiment_name = experiment_name
        
        # Save template
        filename = f"{experiment_name}.yaml"
        self.save_config(config, filename)
        
        self.logger.info("Template created", template=template_name, experiment=experiment_name)
        return config
    
    def validate_config(self, config: ExperimentConfig) -> bool:
        """Validate configuration and return True if valid."""
        try:
            # This will raise validation errors if invalid
            config_dict = config.to_dict()
            ExperimentConfig(**config_dict)
            self.logger.info("Configuration validation passed")
            return True
        except Exception as e:
            self.logger.error("Configuration validation failed", error=str(e))
            return False
    
    def merge_configs(self, base_config: ExperimentConfig, override_config: ExperimentConfig) -> ExperimentConfig:
        """Merge two configurations with override taking precedence."""
        merged_config = base_config.merge(override_config)
        self.logger.info("Configurations merged", 
                        base=base_config.experiment_name,
                        override=override_config.experiment_name)
        return merged_config


# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

class EnvironmentConfigManager:
    """Manager for environment-specific configurations."""
    
    def __init__(self, base_config_dir: str = "configs"):
        
    """__init__ function."""
self.base_config_dir = Path(base_config_dir)
        self.logger = structlog.get_logger(__name__)
    
    def get_environment_config(self, base_config: ExperimentConfig, environment: str) -> ExperimentConfig:
        """Get environment-specific configuration."""
        env_config_path = self.base_config_dir / f"{base_config.experiment_name}_{environment}.yaml"
        
        if env_config_path.exists():
            env_config = ExperimentConfig.load(str(env_config_path))
            # Merge with base config
            return base_config.merge(env_config)
        else:
            # Return base config with environment override
            base_config.environment = environment
            return base_config
    
    def create_environment_config(self, base_config: ExperimentConfig, environment: str, 
                                overrides: Dict[str, Any]) -> ExperimentConfig:
        """Create environment-specific configuration."""
        # Create environment-specific config
        env_config = ExperimentConfig(
            experiment_name=f"{base_config.experiment_name}_{environment}",
            environment=environment,
            model=base_config.model,
            data=base_config.data,
            training=base_config.training,
            evaluation=base_config.evaluation
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(env_config, key):
                setattr(env_config, key, value)
            elif hasattr(env_config.training, key):
                setattr(env_config.training, key, value)
            elif hasattr(env_config.data, key):
                setattr(env_config.data, key, value)
            elif hasattr(env_config.model, key):
                setattr(env_config.model, key, value)
            elif hasattr(env_config.evaluation, key):
                setattr(env_config.evaluation, key, value)
        
        # Save environment config
        env_config_path = self.base_config_dir / f"{base_config.experiment_name}_{environment}.yaml"
        env_config.save(str(env_config_path))
        
        self.logger.info("Environment configuration created", 
                        environment=environment,
                        filepath=str(env_config_path))
        
        return env_config


# =============================================================================
# CONFIGURATION VALIDATION UTILITIES
# =============================================================================

class ConfigurationValidator:
    """Utilities for validating configurations."""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """Validate model configuration and return list of issues."""
        issues = []
        
        # Check required fields based on model type
        if config.model_type == ModelType.TRANSFORMER:
            if not config.num_classes and not config.model_params.get("regression", False):
                issues.append("num_classes is required for transformer classification")
        
        elif config.model_type == ModelType.CNN:
            if not config.num_classes:
                issues.append("num_classes is required for CNN models")
        
        # Check parameter ranges
        if config.dropout < 0 or config.dropout > 1:
            issues.append("dropout must be between 0 and 1")
        
        return issues
    
    @staticmethod
    def validate_training_config(config: TrainingConfig) -> List[str]:
        """Validate training configuration and return list of issues."""
        issues = []
        
        # Check learning rate
        if config.learning_rate <= 0:
            issues.append("learning_rate must be positive")
        
        # Check epochs
        if config.epochs <= 0:
            issues.append("epochs must be positive")
        
        # Check device availability
        if config.device == DeviceType.CUDA and not torch.cuda.is_available():
            issues.append("CUDA requested but not available")
        
        return issues
    
    @staticmethod
    def validate_data_config(config: DataConfig) -> List[str]:
        """Validate data configuration and return list of issues."""
        issues = []
        
        # Check data path exists
        if not Path(config.data_path).exists():
            issues.append(f"Data path does not exist: {config.data_path}")
        
        # Check batch size
        if config.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        # Check split ratios
        total_split = config.train_split + config.val_split + config.test_split
        if abs(total_split - 1.0) > 1e-6:
            issues.append(f"Data splits must sum to 1.0, got {total_split}")
        
        return issues


# =============================================================================
# EXAMPLE USAGE AND MAIN FUNCTION
# =============================================================================

def create_example_configurations():
    """Create example configuration files."""
    config_manager = ConfigurationManager()
    
    # Create templates
    templates = [
        ("transformer_classification", "text_classification_experiment"),
        ("cnn_image_classification", "image_classification_experiment"),
        ("diffusion_generation", "image_generation_experiment")
    ]
    
    for template_name, experiment_name in templates:
        try:
            config = config_manager.create_template(template_name, experiment_name)
            print(f"Created {template_name} template: {experiment_name}")
        except Exception as e:
            print(f"Failed to create {template_name} template: {e}")


def main():
    """Example usage of the configuration management system."""
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Create a custom configuration
    config = ExperimentConfig(
        experiment_name="custom_experiment",
        experiment_description="Custom experiment with YAML configuration",
        model=ModelConfig(
            model_type=ModelType.TRANSFORMER,
            model_name="bert-base-uncased",
            num_classes=5,
            hidden_dim=256,
            dropout=0.1
        ),
        data=DataConfig(
            data_path="data/custom_dataset.csv",
            batch_size=16,
            target_column="label",
            text_column="text"
        ),
        training=TrainingConfig(
            epochs=20,
            learning_rate=2e-5,
            optimizer=OptimizerType.ADAMW,
            scheduler=SchedulerType.COSINE,
            loss_function=LossType.CROSS_ENTROPY
        ),
        evaluation=EvaluationConfig(
            metrics=["accuracy", "f1_score", "precision", "recall"]
        )
    )
    
    # Save configuration
    config_path = config_manager.save_config(config)
    print(f"Configuration saved to: {config_path}")
    
    # Load configuration
    loaded_config = config_manager.load_config("custom_experiment.yaml")
    print(f"Configuration loaded: {loaded_config.experiment_name}")
    
    # Validate configuration
    is_valid = config_manager.validate_config(loaded_config)
    print(f"Configuration is valid: {is_valid}")
    
    # Create environment-specific configuration
    env_manager = EnvironmentConfigManager()
    prod_config = env_manager.create_environment_config(
        loaded_config, 
        "production", 
        {"training.epochs": 50, "training.learning_rate": 1e-5}
    )
    print(f"Production configuration created: {prod_config.experiment_name}")


match __name__:
    case "__main__":
    main() 