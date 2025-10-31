from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, TypeVar
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import asyncio
"""
Configuration Management System
==============================

This module provides a comprehensive configuration management system using YAML files
for hyperparameters and model settings in the AI video generation system.

Features:
- YAML-based configuration files
- Configuration validation and type checking
- Environment-specific configurations
- Configuration inheritance and merging
- Default configurations and overrides
- Configuration templates and presets
"""


# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic config classes
T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig(ABC):
    """Base configuration class with common functionality."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save config to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(self.to_dict(), f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls: Type[T], filepath: Union[str, Path]) -> T:
        """Load config from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate configuration. Override in subclasses."""
        return True
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors. Override in subclasses."""
        return []


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for AI video models."""
    
    # Model identification
    model_type: str = "diffusion"  # "diffusion", "gan", "transformer"
    model_name: str = "default_model"
    version: str = "1.0.0"
    
    # Model architecture
    input_channels: int = 3
    output_channels: int = 3
    latent_dim: int = 512
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_layers: int = 4
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    activation: str = "relu"  # "relu", "leaky_relu", "gelu", "swish"
    
    # Video-specific parameters
    frame_size: List[int] = field(default_factory=lambda: [256, 256])
    num_frames: int = 16
    temporal_stride: int = 1
    fps: int = 30
    
    # Device and performance
    device: str = "cuda"
    dtype: str = "float16"  # "float32", "float16", "bfloat16"
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    
    # Model-specific parameters
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"  # "linear", "cosine", "sigmoid"
    classifier_free_guidance: bool = True
    guidance_scale: float = 7.5
    
    def validate(self) -> bool:
        """Validate model configuration."""
        errors = self.get_validation_errors()
        return len(errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors for model configuration."""
        errors = []
        
        # Validate model type
        valid_model_types = ["diffusion", "gan", "transformer"]
        if self.model_type not in valid_model_types:
            errors.append(f"Invalid model_type: {self.model_type}. Must be one of {valid_model_types}")
        
        # Validate frame size
        if len(self.frame_size) != 2:
            errors.append(f"frame_size must have exactly 2 elements, got {len(self.frame_size)}")
        elif any(size <= 0 for size in self.frame_size):
            errors.append("frame_size elements must be positive")
        
        # Validate dimensions
        if self.input_channels <= 0:
            errors.append("input_channels must be positive")
        if self.output_channels <= 0:
            errors.append("output_channels must be positive")
        if self.latent_dim <= 0:
            errors.append("latent_dim must be positive")
        if self.num_frames <= 0:
            errors.append("num_frames must be positive")
        
        # Validate dropout rate
        if not 0 <= self.dropout_rate <= 1:
            errors.append("dropout_rate must be between 0 and 1")
        
        # Validate activation
        valid_activations = ["relu", "leaky_relu", "gelu", "swish", "tanh", "sigmoid"]
        if self.activation not in valid_activations:
            errors.append(f"Invalid activation: {self.activation}. Must be one of {valid_activations}")
        
        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            errors.append(f"Invalid device: {self.device}. Must be one of {valid_devices}")
        
        # Validate dtype
        valid_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtype not in valid_dtypes:
            errors.append(f"Invalid dtype: {self.dtype}. Must be one of {valid_dtypes}")
        
        return errors


@dataclass
class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_dir: str = "data/videos"
    metadata_file: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Video parameters
    frame_size: List[int] = field(default_factory=lambda: [256, 256])
    num_frames: int = 16
    fps: int = 30
    channels: int = 3
    
    # Data loading parameters
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    
    # Preprocessing parameters
    normalize: bool = True
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    augment: bool = True
    cache_data: bool = False
    max_cache_size: int = 1000
    
    # Augmentation parameters
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    rotation_prob: float = 0.2
    rotation_degrees: float = 15.0
    crop_prob: float = 0.3
    crop_ratio: float = 0.8
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1
    saturation_jitter: float = 0.1
    hue_jitter: float = 0.05
    
    # Split parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    def validate(self) -> bool:
        """Validate data configuration."""
        errors = self.get_validation_errors()
        return len(errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors for data configuration."""
        errors = []
        
        # Validate data directory
        if not Path(self.data_dir).exists():
            errors.append(f"Data directory does not exist: {self.data_dir}")
        
        # Validate frame size
        if len(self.frame_size) != 2:
            errors.append(f"frame_size must have exactly 2 elements, got {len(self.frame_size)}")
        elif any(size <= 0 for size in self.frame_size):
            errors.append("frame_size elements must be positive")
        
        # Validate dimensions
        if self.channels <= 0:
            errors.append("channels must be positive")
        if self.num_frames <= 0:
            errors.append("num_frames must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.num_workers < 0:
            errors.append("num_workers must be non-negative")
        
        # Validate split ratios
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            errors.append(f"Split ratios must sum to 1.0, got {total_split}")
        
        # Validate probabilities
        probabilities = [
            self.horizontal_flip_prob, self.vertical_flip_prob, self.rotation_prob,
            self.crop_prob, self.brightness_jitter, self.contrast_jitter,
            self.saturation_jitter, self.hue_jitter
        ]
        for prob in probabilities:
            if not 0 <= prob <= 1:
                errors.append(f"Probability must be between 0 and 1, got {prob}")
        
        # Validate normalization parameters
        if len(self.normalize_mean) != 3:
            errors.append(f"normalize_mean must have exactly 3 elements, got {len(self.normalize_mean)}")
        if len(self.normalize_std) != 3:
            errors.append(f"normalize_std must have exactly 3 elements, got {len(self.normalize_std)}")
        
        return errors


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for model training."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    max_grad_norm: float = 1.0
    
    # Optimizer parameters
    optimizer: str = "adam"  # "adam", "adamw", "sgd", "rmsprop"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    scheduler: str = "cosine"  # "step", "cosine", "plateau", "exponential"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Loss parameters
    loss_type: str = "mse"  # "mse", "l1", "perceptual", "adversarial", "combined"
    loss_weights: Dict[str, float] = field(default_factory=dict)
    
    # Monitoring parameters
    save_frequency: int = 10
    eval_frequency: int = 5
    log_frequency: int = 100
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    max_checkpoints: int = 5
    resume_from_checkpoint: Optional[str] = None
    
    # Experiment tracking
    use_wandb: bool = False
    use_tensorboard: bool = True
    experiment_name: str = "ai_video_training"
    project_name: str = "ai_video"
    
    # Advanced training parameters
    use_amp: bool = True  # Automatic mixed precision
    use_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 1
    use_ddp: bool = False  # Distributed data parallel
    local_rank: int = -1
    
    def validate(self) -> bool:
        """Validate training configuration."""
        errors = self.get_validation_errors()
        return len(errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors for training configuration."""
        errors = []
        
        # Validate basic parameters
        if self.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.weight_decay < 0:
            errors.append("weight_decay must be non-negative")
        if self.gradient_clip < 0:
            errors.append("gradient_clip must be non-negative")
        
        # Validate optimizer
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop"]
        if self.optimizer not in valid_optimizers:
            errors.append(f"Invalid optimizer: {self.optimizer}. Must be one of {valid_optimizers}")
        
        # Validate scheduler
        valid_schedulers = ["step", "cosine", "plateau", "exponential", "none"]
        if self.scheduler not in valid_schedulers:
            errors.append(f"Invalid scheduler: {self.scheduler}. Must be one of {valid_schedulers}")
        
        # Validate loss type
        valid_losses = ["mse", "l1", "perceptual", "adversarial", "combined"]
        if self.loss_type not in valid_losses:
            errors.append(f"Invalid loss_type: {self.loss_type}. Must be one of {valid_losses}")
        
        # Validate frequencies
        if self.save_frequency <= 0:
            errors.append("save_frequency must be positive")
        if self.eval_frequency <= 0:
            errors.append("eval_frequency must be positive")
        if self.log_frequency <= 0:
            errors.append("log_frequency must be positive")
        
        # Validate gradient accumulation
        if self.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")
        
        return errors


@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for model evaluation."""
    
    # Evaluation parameters
    batch_size: int = 8
    num_samples: Optional[int] = None  # None for all samples
    device: str = "cuda"
    
    # Metrics to compute
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_lpips: bool = True
    compute_fid: bool = False
    compute_inception_score: bool = False
    
    # Output parameters
    save_results: bool = True
    save_videos: bool = False
    output_dir: str = "evaluation_results"
    save_format: str = "mp4"  # "mp4", "gif", "frames"
    
    # Visualization parameters
    create_plots: bool = True
    plot_samples: int = 5
    plot_style: str = "seaborn"  # "default", "seaborn", "ggplot"
    
    # Advanced evaluation parameters
    use_ensemble: bool = False
    ensemble_size: int = 3
    compute_uncertainty: bool = False
    num_mc_samples: int = 10
    
    def validate(self) -> bool:
        """Validate evaluation configuration."""
        errors = self.get_validation_errors()
        return len(errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors for evaluation configuration."""
        errors = []
        
        # Validate basic parameters
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.num_samples is not None and self.num_samples <= 0:
            errors.append("num_samples must be positive if specified")
        
        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            errors.append(f"Invalid device: {self.device}. Must be one of {valid_devices}")
        
        # Validate save format
        valid_formats = ["mp4", "gif", "frames"]
        if self.save_format not in valid_formats:
            errors.append(f"Invalid save_format: {self.save_format}. Must be one of {valid_formats}")
        
        # Validate plot style
        valid_styles = ["default", "seaborn", "ggplot"]
        if self.plot_style not in valid_styles:
            errors.append(f"Invalid plot_style: {self.plot_style}. Must be one of {valid_styles}")
        
        # Validate ensemble parameters
        if self.ensemble_size <= 0:
            errors.append("ensemble_size must be positive")
        if self.num_mc_samples <= 0:
            errors.append("num_mc_samples must be positive")
        
        return errors


@dataclass
class SystemConfig(BaseConfig):
    """Configuration for system settings."""
    
    # Environment
    environment: str = "development"  # "development", "staging", "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # Paths
    base_dir: str = "."
    config_dir: str = "configs"
    output_dir: str = "outputs"
    logs_dir: str = "logs"
    
    # Performance
    num_threads: int = 4
    memory_limit: Optional[int] = None  # MB
    gpu_memory_fraction: float = 0.9
    
    # Security
    enable_security: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov"])
    
    def validate(self) -> bool:
        """Validate system configuration."""
        errors = self.get_validation_errors()
        return len(errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors for system configuration."""
        errors = []
        
        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            errors.append(f"Invalid environment: {self.environment}. Must be one of {valid_environments}")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")
        
        # Validate performance parameters
        if self.num_threads <= 0:
            errors.append("num_threads must be positive")
        if not 0 < self.gpu_memory_fraction <= 1:
            errors.append("gpu_memory_fraction must be between 0 and 1")
        
        return errors


@dataclass
class CompleteConfig(BaseConfig):
    """Complete configuration combining all components."""
    
    # Component configurations
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Metadata
    name: str = "default_config"
    description: str = "Default configuration for AI video generation"
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def validate(self) -> bool:
        """Validate complete configuration."""
        errors = self.get_validation_errors()
        return len(errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors for complete configuration."""
        errors = []
        
        # Validate each component
        component_configs = [
            ("system", self.system),
            ("model", self.model),
            ("data", self.data),
            ("training", self.training),
            ("evaluation", self.evaluation)
        ]
        
        for name, config in component_configs:
            component_errors = config.get_validation_errors()
            for error in component_errors:
                errors.append(f"{name}: {error}")
        
        return errors


class ConfigManager:
    """Manager for handling configuration files and operations."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configurations
        self.default_configs = self._load_default_configs()
        
        logger.info(f"ConfigManager initialized with config directory: {self.config_dir}")
    
    def _load_default_configs(self) -> Dict[str, CompleteConfig]:
        """Load default configurations."""
        defaults = {}
        
        # Diffusion model config
        diffusion_config = CompleteConfig(
            name="diffusion_default",
            description="Default configuration for diffusion-based video generation",
            model=ModelConfig(
                model_type="diffusion",
                model_name="diffusion_model",
                frame_size=[256, 256],
                num_frames=16,
                diffusion_steps=1000
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=1e-4,
                loss_type="mse"
            )
        )
        defaults["diffusion"] = diffusion_config
        
        # GAN model config
        gan_config = CompleteConfig(
            name="gan_default",
            description="Default configuration for GAN-based video generation",
            model=ModelConfig(
                model_type="gan",
                model_name="gan_model",
                frame_size=[256, 256],
                num_frames=16
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=2e-4,
                loss_type="adversarial"
            )
        )
        defaults["gan"] = gan_config
        
        # Transformer model config
        transformer_config = CompleteConfig(
            name="transformer_default",
            description="Default configuration for transformer-based video generation",
            model=ModelConfig(
                model_type="transformer",
                model_name="transformer_model",
                frame_size=[256, 256],
                num_frames=16,
                num_layers=6
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=1e-4,
                loss_type="mse"
            )
        )
        defaults["transformer"] = transformer_config
        
        return defaults
    
    def create_config(self, config_name: str, model_type: str = "diffusion") -> CompleteConfig:
        """Create a new configuration based on model type."""
        if model_type not in self.default_configs:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.default_configs.keys())}")
        
        # Copy default config and customize
        config = copy.deepcopy(self.default_configs[model_type])
        config.name = config_name
        config.description = f"Configuration for {config_name}"
        config.created_at = datetime.now().isoformat()
        
        return config
    
    def save_config(self, config: CompleteConfig, filename: Optional[str] = None) -> Path:
        """Save configuration to file."""
        if filename is None:
            filename = f"{config.name}_{config.model.model_type}.yaml"
        
        filepath = self.config_dir / filename
        
        # Validate before saving
        if not config.validate():
            errors = config.get_validation_errors()
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
        
        config.save(filepath)
        return filepath
    
    def load_config(self, filename: str) -> CompleteConfig:
        """Load configuration from file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        config = CompleteConfig.load(filepath)
        
        # Validate loaded config
        if not config.validate():
            errors = config.get_validation_errors()
            logger.warning(f"Configuration validation warnings:\n" + "\n".join(errors))
        
        return config
    
    def list_configs(self) -> List[str]:
        """List all available configuration files."""
        config_files = []
        for filepath in self.config_dir.glob("*.yaml"):
            config_files.append(filepath.name)
        for filepath in self.config_dir.glob("*.yml"):
            config_files.append(filepath.name)
        for filepath in self.config_dir.glob("*.json"):
            config_files.append(filepath.name)
        
        return sorted(config_files)
    
    def merge_configs(self, base_config: CompleteConfig, override_config: Dict[str, Any]) -> CompleteConfig:
        """Merge configuration with overrides."""
        merged_config = copy.deepcopy(base_config)
        
        def merge_dict(target: Dict[str, Any], source: Dict[str, Any]):
            
    """merge_dict function."""
for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dict(target[key], value)
                else:
                    target[key] = value
        
        # Convert config to dict, merge, then convert back
        config_dict = merged_config.to_dict()
        merge_dict(config_dict, override_config)
        
        return CompleteConfig.from_dict(config_dict)
    
    def create_environment_config(self, base_config: CompleteConfig, environment: str) -> CompleteConfig:
        """Create environment-specific configuration."""
        env_config = copy.deepcopy(base_config)
        env_config.system.environment = environment
        
        # Environment-specific overrides
        if environment == "production":
            env_config.system.debug = False
            env_config.system.log_level = "WARNING"
            env_config.training.use_wandb = True
            env_config.training.save_best_only = True
        elif environment == "development":
            env_config.system.debug = True
            env_config.system.log_level = "DEBUG"
            env_config.training.num_epochs = 5
            env_config.training.use_wandb = False
        elif environment == "staging":
            env_config.system.debug = False
            env_config.system.log_level = "INFO"
            env_config.training.num_epochs = 20
        
        return env_config
    
    def validate_config(self, config: CompleteConfig) -> Dict[str, List[str]]:
        """Validate configuration and return detailed errors."""
        validation_results = {}
        
        # Validate each component
        components = [
            ("system", config.system),
            ("model", config.model),
            ("data", config.data),
            ("training", config.training),
            ("evaluation", config.evaluation)
        ]
        
        for name, component_config in components:
            errors = component_config.get_validation_errors()
            if errors:
                validation_results[name] = errors
        
        return validation_results


# Convenience functions
def create_config_manager(config_dir: str = "configs") -> ConfigManager:
    """Create a configuration manager instance."""
    return ConfigManager(config_dir)


def load_config_from_file(filepath: Union[str, Path]) -> CompleteConfig:
    """Load configuration from file."""
    return CompleteConfig.load(filepath)


def save_config_to_file(config: CompleteConfig, filepath: Union[str, Path]) -> None:
    """Save configuration to file."""
    config.save(filepath)


def create_default_config(model_type: str = "diffusion") -> CompleteConfig:
    """Create a default configuration for the specified model type."""
    manager = ConfigManager()
    return manager.create_config(f"{model_type}_default", model_type)


if __name__ == "__main__":
    # Example usage
    print("🔧 Configuration Management System")
    print("=" * 40)
    
    # Create config manager
    manager = create_config_manager()
    
    # Create and save a configuration
    config = manager.create_config("my_experiment", "diffusion")
    
    # Customize the configuration
    config.model.frame_size = [512, 512]
    config.training.num_epochs = 50
    config.training.learning_rate = 5e-5
    
    # Save configuration
    filepath = manager.save_config(config)
    print(f"✅ Configuration saved to: {filepath}")
    
    # Load configuration
    loaded_config = manager.load_config(filepath.name)
    print(f"✅ Configuration loaded: {loaded_config.name}")
    
    # Validate configuration
    validation_results = manager.validate_config(loaded_config)
    if validation_results:
        print("⚠️ Validation warnings:")
        for component, errors in validation_results.items():
            for error in errors:
                print(f"   {component}: {error}")
    else:
        print("✅ Configuration is valid")
    
    # List available configs
    configs = manager.list_configs()
    print(f"📁 Available configurations: {configs}") 