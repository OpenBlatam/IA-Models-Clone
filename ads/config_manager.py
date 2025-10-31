from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, Any, List, Optional, Union, Tuple
import yaml
import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime
from enum import Enum
import copy
from contextlib import contextmanager
from onyx.utils.logger import setup_logger
from typing import Any, List, Dict, Optional
import asyncio
"""
Configuration Manager for Onyx Ads Backend

This module provides comprehensive configuration management using YAML files for:
- Hyperparameters and model settings
- Training configurations
- Data processing settings
- Experiment configurations
- Environment-specific settings
"""


logger = setup_logger()

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
    cache_dir: str = "./cache"
    
    # Data processing
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Data augmentation
    augmentation: bool = False
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data validation
    validate_data: bool = True
    max_samples: Optional[int] = None
    shuffle: bool = True
    
    # Data types
    input_dtype: str = "float32"
    target_dtype: str = "long"

@dataclass
class ExperimentConfig:
    """Experiment tracking configuration settings."""
    # Experiment identification
    experiment_name: str = ""
    experiment_id: str = ""
    project_name: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Tracking settings
    track_experiments: bool = True
    tracking_backend: str = "wandb"  # wandb, mlflow, tensorboard, local
    tracking_params: Dict[str, Any] = field(default_factory=dict)
    
    # Logging settings
    log_level: str = "INFO"
    log_frequency: int = 100
    log_metrics: List[str] = field(default_factory=list)
    log_gradients: bool = False
    log_hyperparameters: bool = True
    
    # Checkpointing settings
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 1
    max_checkpoints: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True

@dataclass
class OptimizationConfig:
    """Optimization configuration settings."""
    # Performance optimization
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_model_compilation: bool = True
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    
    # Memory optimization
    memory_fraction: float = 0.8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Multi-GPU settings
    distributed_training: bool = False
    num_gpus: int = 1
    backend: str = "nccl"
    
    # Profiling settings
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_performance: bool = True

@dataclass
class DeploymentConfig:
    """Deployment configuration settings."""
    # Model serving
    model_format: str = "torchscript"  # torchscript, onnx, tensorrt
    optimization_level: str = "O1"
    quantization: bool = False
    quantization_params: Dict[str, Any] = field(default_factory=dict)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_timeout: int = 30
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_endpoint: str = "/metrics"
    health_check_endpoint: str = "/health"

class ConfigManager:
    """Comprehensive configuration manager for ML projects."""
    
    def __init__(self, config_dir: str = "./configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.configs = {}
        
    def create_default_configs(self, project_name: str) -> Dict[str, str]:
        """Create default configuration files for a project."""
        config_files = {}
        
        # Model configuration
        model_config = ModelConfig(
            name="default_model",
            type="transformer",
            architecture="bert-base-uncased",
            input_size=768,
            output_size=10,
            hidden_sizes=[512, 256],
            dropout_rate=0.1,
            activation="gelu",
            batch_norm=True,
            pretrained=True
        )
        config_files['model'] = self.save_config(
            model_config, f"{project_name}_model_config.yaml", ConfigType.MODEL
        )
        
        # Training configuration
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=2e-5,
            epochs=10,
            optimizer="adamw",
            optimizer_params={"weight_decay": 0.01},
            scheduler="cosine",
            loss_function="cross_entropy",
            mixed_precision=True,
            gradient_accumulation_steps=4
        )
        config_files['training'] = self.save_config(
            training_config, f"{project_name}_training_config.yaml", ConfigType.TRAINING
        )
        
        # Data configuration
        data_config = DataConfig(
            train_data_path="./data/train",
            val_data_path="./data/val",
            test_data_path="./data/test",
            num_workers=4,
            pin_memory=True,
            augmentation=True,
            augmentation_params={
                "rotation": 10,
                "horizontal_flip": True,
                "color_jitter": 0.1
            }
        )
        config_files['data'] = self.save_config(
            data_config, f"{project_name}_data_config.yaml", ConfigType.DATA
        )
        
        # Experiment configuration
        experiment_config = ExperimentConfig(
            experiment_name=f"{project_name}_experiment",
            project_name=project_name,
            track_experiments=True,
            tracking_backend="wandb",
            save_checkpoints=True,
            checkpoint_dir=f"./checkpoints/{project_name}",
            log_metrics=["loss", "accuracy", "f1_score"]
        )
        config_files['experiment'] = self.save_config(
            experiment_config, f"{project_name}_experiment_config.yaml", ConfigType.EXPERIMENT
        )
        
        # Optimization configuration
        optimization_config = OptimizationConfig(
            enable_mixed_precision=True,
            enable_gradient_checkpointing=True,
            enable_model_compilation=True,
            distributed_training=False,
            num_gpus=1
        )
        config_files['optimization'] = self.save_config(
            optimization_config, f"{project_name}_optimization_config.yaml", ConfigType.OPTIMIZATION
        )
        
        # Deployment configuration
        deployment_config = DeploymentConfig(
            model_format="torchscript",
            optimization_level="O1",
            quantization=False,
            enable_monitoring=True
        )
        config_files['deployment'] = self.save_config(
            deployment_config, f"{project_name}_deployment_config.yaml", ConfigType.DEPLOYMENT
        )
        
        return config_files
    
    def save_config(self, config: Any, filename: str, config_type: ConfigType) -> str:
        """Save configuration to YAML file."""
        config_path = self.config_dir / filename
        
        # Convert dataclass to dict
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        else:
            config_dict = config
        
        # Add metadata
        config_dict['_metadata'] = {
            'config_type': config_type.value,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        # Save to YAML
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_path}")
        return str(config_path)
    
    def load_config(self, config_path: str, config_class: Optional[type] = None) -> Any:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = yaml.safe_load(f)
        
        # Remove metadata
        metadata = config_dict.pop('_metadata', {})
        
        # Convert to dataclass if specified
        if config_class:
            config = config_class(**config_dict)
        else:
            config = config_dict
        
        self.logger.info(f"Configuration loaded from: {config_path}")
        return config
    
    def load_all_configs(self, project_name: str) -> Dict[str, Any]:
        """Load all configuration files for a project."""
        configs = {}
        
        config_patterns = {
            'model': f"{project_name}_model_config.yaml",
            'training': f"{project_name}_training_config.yaml",
            'data': f"{project_name}_data_config.yaml",
            'experiment': f"{project_name}_experiment_config.yaml",
            'optimization': f"{project_name}_optimization_config.yaml",
            'deployment': f"{project_name}_deployment_config.yaml"
        }
        
        config_classes = {
            'model': ModelConfig,
            'training': TrainingConfig,
            'data': DataConfig,
            'experiment': ExperimentConfig,
            'optimization': OptimizationConfig,
            'deployment': DeploymentConfig
        }
        
        for config_type, filename in config_patterns.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                configs[config_type] = self.load_config(
                    str(config_path), config_classes[config_type]
                )
        
        return configs
    
    def update_config(self, config_path: str, updates: Dict[str, Any]) -> str:
        """Update existing configuration with new values."""
        # Load existing config
        config_dict = self.load_config(config_path)
        
        # Apply updates
        self._update_nested_dict(config_dict, updates)
        
        # Save updated config
        config_path = Path(config_path)
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration updated: {config_path}")
        return str(config_path)
    
    def _update_nested_dict(self, base_dict: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self, config: Any, config_type: ConfigType) -> bool:
        """Validate configuration settings."""
        try:
            if config_type == ConfigType.MODEL:
                return self._validate_model_config(config)
            elif config_type == ConfigType.TRAINING:
                return self._validate_training_config(config)
            elif config_type == ConfigType.DATA:
                return self._validate_data_config(config)
            elif config_type == ConfigType.EXPERIMENT:
                return self._validate_experiment_config(config)
            elif config_type == ConfigType.OPTIMIZATION:
                return self._validate_optimization_config(config)
            elif config_type == ConfigType.DEPLOYMENT:
                return self._validate_deployment_config(config)
            else:
                return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_model_config(self, config: ModelConfig) -> bool:
        """Validate model configuration."""
        if not config.name or not config.architecture:
            return False
        if config.dropout_rate < 0 or config.dropout_rate > 1:
            return False
        return True
    
    def _validate_training_config(self, config: TrainingConfig) -> bool:
        """Validate training configuration."""
        if config.batch_size <= 0 or config.learning_rate <= 0:
            return False
        if config.validation_split < 0 or config.validation_split > 1:
            return False
        if config.test_split < 0 or config.test_split > 1:
            return False
        return True
    
    def _validate_data_config(self, config: DataConfig) -> bool:
        """Validate data configuration."""
        if config.num_workers < 0:
            return False
        if config.prefetch_factor < 1:
            return False
        return True
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration."""
        if config.checkpoint_frequency < 1:
            return False
        if config.max_checkpoints < 1:
            return False
        return True
    
    def _validate_optimization_config(self, config: OptimizationConfig) -> bool:
        """Validate optimization configuration."""
        if config.memory_fraction <= 0 or config.memory_fraction > 1:
            return False
        if config.num_gpus < 1:
            return False
        return True
    
    def _validate_deployment_config(self, config: DeploymentConfig) -> bool:
        """Validate deployment configuration."""
        if config.api_port < 1 or config.api_port > 65535:
            return False
        if config.api_workers < 1:
            return False
        return True
    
    def get_config_summary(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all configurations."""
        summary = {
            'project_info': {},
            'model_info': {},
            'training_info': {},
            'data_info': {},
            'experiment_info': {},
            'optimization_info': {},
            'deployment_info': {}
        }
        
        for config_type, config in configs.items():
            if hasattr(config, '__dataclass_fields__'):
                config_dict = asdict(config)
            else:
                config_dict = config
            
            summary[f'{config_type}_info'] = {
                'type': config_type,
                'key_parameters': self._extract_key_parameters(config_dict, config_type)
            }
        
        return summary
    
    def _extract_key_parameters(self, config_dict: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """Extract key parameters for summary."""
        key_params = {}
        
        if config_type == 'model':
            key_params = {
                'name': config_dict.get('name'),
                'architecture': config_dict.get('architecture'),
                'input_size': config_dict.get('input_size'),
                'output_size': config_dict.get('output_size')
            }
        elif config_type == 'training':
            key_params = {
                'batch_size': config_dict.get('batch_size'),
                'learning_rate': config_dict.get('learning_rate'),
                'epochs': config_dict.get('epochs'),
                'optimizer': config_dict.get('optimizer')
            }
        elif config_type == 'data':
            key_params = {
                'num_workers': config_dict.get('num_workers'),
                'augmentation': config_dict.get('augmentation'),
                'shuffle': config_dict.get('shuffle')
            }
        elif config_type == 'experiment':
            key_params = {
                'experiment_name': config_dict.get('experiment_name'),
                'tracking_backend': config_dict.get('tracking_backend'),
                'save_checkpoints': config_dict.get('save_checkpoints')
            }
        elif config_type == 'optimization':
            key_params = {
                'mixed_precision': config_dict.get('enable_mixed_precision'),
                'distributed_training': config_dict.get('distributed_training'),
                'num_gpus': config_dict.get('num_gpus')
            }
        elif config_type == 'deployment':
            key_params = {
                'model_format': config_dict.get('model_format'),
                'api_port': config_dict.get('api_port'),
                'enable_monitoring': config_dict.get('enable_monitoring')
            }
        
        return key_params

# Utility functions
def create_config_from_dict(config_dict: Dict[str, Any], config_class: type) -> Any:
    """Create configuration object from dictionary."""
    return config_class(**config_dict)

def merge_configs(base_config: Any, override_config: Dict[str, Any]) -> Any:
    """Merge configuration with overrides."""
    if hasattr(base_config, '__dataclass_fields__'):
        base_dict = asdict(base_config)
    else:
        base_dict = base_config
    
    # Apply overrides
    for key, value in override_config.items():
        if key in base_dict:
            base_dict[key] = value
    
    # Return new config object
    config_class = type(base_config)
    return config_class(**base_dict)

@contextmanager
def config_context(config_manager: ConfigManager, config_path: str):
    """Context manager for configuration handling."""
    try:
        config = config_manager.load_config(config_path)
        yield config
    finally:
        pass

# Example usage
if __name__ == "__main__":
    # Create config manager
    config_manager = ConfigManager("./configs")
    
    # Create default configs for a project
    config_files = config_manager.create_default_configs("ad_classification")
    
    # Load all configs
    configs = config_manager.load_all_configs("ad_classification")
    
    # Get config summary
    summary = config_manager.get_config_summary(configs)
    print("Configuration Summary:")
    print(json.dumps(summary, indent=2)) 