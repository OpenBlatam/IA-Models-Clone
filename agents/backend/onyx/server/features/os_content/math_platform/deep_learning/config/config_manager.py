from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import yaml
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import copy
import argparse
from datetime import datetime
import jinja2
import re
        import itertools
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Configuration Manager
Comprehensive configuration management system using YAML files for hyperparameters and model settings.
"""


logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    """Base configuration class with common functionality."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    def save(self, filepath: str):
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
        else:
            with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Config saved to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'BaseConfig':
        """Create config from YAML string."""
        config_dict = yaml.safe_load(yaml_content)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'BaseConfig':
        """Load config from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def merge(self, other_config: 'BaseConfig') -> 'BaseConfig':
        """Merge with another config."""
        merged_dict = self.to_dict()
        other_dict = other_config.to_dict()
        
        # Deep merge
        for key, value in other_dict.items():
            if key in merged_dict and isinstance(merged_dict[key], dict) and isinstance(value, dict):
                merged_dict[key].update(value)
            else:
                merged_dict[key] = value
        
        return self.__class__.from_dict(merged_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration. Returns list of validation errors."""
        errors = []
        # Override in subclasses for specific validation
        return errors


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model architecture and parameters."""
    # Model identification
    model_name: str = "default_model"
    model_type: str = "mlp"  # mlp, cnn, rnn, transformer, diffusion
    model_version: str = "1.0.0"
    
    # Architecture parameters
    input_size: Union[int, Tuple[int, ...]] = 784
    output_size: int = 10
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_layers: int = 3
    
    # Layer-specific parameters
    activation: str = "relu"  # relu, tanh, sigmoid, gelu, swish
    dropout_rate: float = 0.1
    batch_norm: bool = True
    layer_norm: bool = False
    
    # Initialization parameters
    weight_init: str = "xavier"  # xavier, kaiming, normal, uniform
    bias_init: str = "zeros"
    
    # Model-specific parameters
    num_heads: int = 8  # For transformers
    embedding_dim: int = 512  # For transformers
    max_seq_length: int = 512  # For transformers
    vocab_size: int = 30000  # For transformers
    
    # CNN-specific parameters
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    
    # RNN-specific parameters
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    bidirectional: bool = False
    
    # Diffusion-specific parameters
    noise_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    def validate(self) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        if self.model_name.strip() == "":
            errors.append("Model name cannot be empty")
        
        if self.input_size <= 0:
            errors.append("Input size must be positive")
        
        if self.output_size <= 0:
            errors.append("Output size must be positive")
        
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            errors.append("Dropout rate must be between 0 and 1")
        
        if self.model_type == "transformer":
            if self.num_heads <= 0:
                errors.append("Number of heads must be positive")
            if self.embedding_dim <= 0:
                errors.append("Embedding dimension must be positive")
        
        if self.model_type == "cnn":
            if len(self.conv_channels) != len(self.kernel_sizes):
                errors.append("Number of conv channels must match number of kernel sizes")
        
        return errors


@dataclass
class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing."""
    # Dataset identification
    dataset_name: str = "default_dataset"
    dataset_type: str = "classification"  # classification, regression, segmentation
    data_path: str = "./data"
    
    # Data parameters
    input_size: Tuple[int, ...] = (224, 224)
    num_classes: int = 10
    num_channels: int = 3
    num_samples: int = -1  # -1 for all samples
    
    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    
    # Preprocessing
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Augmentation
    augmentation: bool = False
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    scale: Tuple[float, float] = (0.8, 1.2)
    crop_size: Tuple[int, int] = (224, 224)
    
    # Data format
    data_format: str = "image"  # image, text, audio, tabular
    file_extensions: List[str] = field(default_factory=lambda: [".jpg", ".png", ".jpeg"])
    
    # Caching
    cache_data: bool = False
    cache_dir: str = "./cache"
    
    def validate(self) -> List[str]:
        """Validate data configuration."""
        errors = []
        
        if self.dataset_name.strip() == "":
            errors.append("Dataset name cannot be empty")
        
        if not (0 < self.train_split < 1):
            errors.append("Train split must be between 0 and 1")
        
        if not (0 < self.val_split < 1):
            errors.append("Validation split must be between 0 and 1")
        
        if not (0 < self.test_split < 1):
            errors.append("Test split must be between 0 and 1")
        
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            errors.append(f"Split ratios must sum to 1.0, got {total_split}")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.num_workers < 0:
            errors.append("Number of workers cannot be negative")
        
        return errors


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training process."""
    # Training identification
    experiment_name: str = "default_experiment"
    run_name: str = "run_1"
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adam"  # adam, sgd, adamw, rmsprop
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    scheduler: str = "cosine"  # cosine, step, exponential, plateau, none
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Loss function
    loss_function: str = "cross_entropy"  # cross_entropy, mse, bce, focal
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Device and performance
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and monitoring
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 1
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "deep_learning_project"
    wandb_entity: str = ""
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    save_best_only: bool = True
    save_last: bool = True
    save_top_k: int = 3
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    mode: str = "min"  # min, max
    
    # Validation
    validation_split: float = 0.2
    validation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    
    # Advanced training
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    max_steps: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        if self.experiment_name.strip() == "":
            errors.append("Experiment name cannot be empty")
        
        if self.epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        if self.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.weight_decay < 0:
            errors.append("Weight decay cannot be negative")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.patience <= 0:
            errors.append("Patience must be positive")
        
        if self.gradient_accumulation_steps < 1:
            errors.append("Gradient accumulation steps must be at least 1")
        
        return errors


@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for evaluation process."""
    # Evaluation identification
    evaluation_name: str = "default_evaluation"
    
    # Evaluation parameters
    batch_size: int = 32
    num_workers: int = 4
    device: str = "auto"
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    average: str = "weighted"  # micro, macro, weighted, binary
    zero_division: int = 0
    
    # Output
    save_predictions: bool = True
    save_plots: bool = True
    save_report: bool = True
    output_dir: str = "./evaluation_results"
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall: bool = True
    plot_learning_curves: bool = True
    plot_prediction_distribution: bool = True
    
    # Analysis
    error_analysis: bool = True
    feature_importance: bool = False
    model_interpretation: bool = False
    confidence_analysis: bool = True
    
    # Export formats
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "html"])
    
    def validate(self) -> List[str]:
        """Validate evaluation configuration."""
        errors = []
        
        if self.evaluation_name.strip() == "":
            errors.append("Evaluation name cannot be empty")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.num_workers < 0:
            errors.append("Number of workers cannot be negative")
        
        if self.average not in ["micro", "macro", "weighted", "binary"]:
            errors.append("Average must be one of: micro, macro, weighted, binary")
        
        return errors


@dataclass
class SystemConfig(BaseConfig):
    """Configuration for system settings."""
    # Project settings
    project_name: str = "deep_learning_project"
    project_root: str = "./"
    experiment_dir: str = "./experiments"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Random seeds
    random_seed: int = 42
    deterministic: bool = False
    
    # Performance
    num_threads: int = -1  # -1 for auto
    memory_fraction: float = 0.8
    
    # Security
    enable_encryption: bool = False
    encryption_key: str = ""
    
    def validate(self) -> List[str]:
        """Validate system configuration."""
        errors = []
        
        if self.project_name.strip() == "":
            errors.append("Project name cannot be empty")
        
        if not (0 < self.memory_fraction <= 1):
            errors.append("Memory fraction must be between 0 and 1")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append("Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        
        return errors


@dataclass
class CompleteConfig(BaseConfig):
    """Complete configuration combining all configs."""
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def validate(self) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        # Validate individual configs
        errors.extend(self.system.validate())
        errors.extend(self.model.validate())
        errors.extend(self.data.validate())
        errors.extend(self.training.validate())
        errors.extend(self.evaluation.validate())
        
        # Cross-config validation
        if self.data.batch_size != self.training.batch_size:
            logger.warning("Data batch size and training batch size are different")
        
        if self.data.num_classes != self.model.output_size:
            errors.append("Data number of classes must match model output size")
        
        return errors


class ConfigManager:
    """Manager for handling configuration files and settings."""
    
    def __init__(self, config_dir: str = "./configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.config_dir)),
            autoescape=True
        )
    
    def create_default_configs(self) -> Dict[str, str]:
        """Create default configuration files."""
        configs = {
            'system.yaml': SystemConfig().to_yaml(),
            'model.yaml': ModelConfig().to_yaml(),
            'data.yaml': DataConfig().to_yaml(),
            'training.yaml': TrainingConfig().to_yaml(),
            'evaluation.yaml': EvaluationConfig().to_yaml(),
            'complete.yaml': CompleteConfig().to_yaml()
        }
        
        created_files = {}
        for filename, content in configs.items():
            filepath = self.config_dir / filename
            with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            created_files[filename] = str(filepath)
            logger.info(f"Created default config: {filepath}")
        
        return created_files
    
    def load_config(self, config_type: str, filepath: Optional[str] = None) -> BaseConfig:
        """Load configuration by type."""
        config_classes = {
            'system': SystemConfig,
            'model': ModelConfig,
            'data': DataConfig,
            'training': TrainingConfig,
            'evaluation': EvaluationConfig,
            'complete': CompleteConfig
        }
        
        if config_type not in config_classes:
            raise ValueError(f"Unknown config type: {config_type}")
        
        config_class = config_classes[config_type]
        
        if filepath is None:
            filepath = self.config_dir / f"{config_type}.yaml"
        
        return config_class.from_file(filepath)
    
    def load_complete_config(self, filepath: Optional[str] = None) -> CompleteConfig:
        """Load complete configuration."""
        if filepath is None:
            filepath = self.config_dir / "complete.yaml"
        
        return CompleteConfig.from_file(filepath)
    
    def save_config(self, config: BaseConfig, filepath: str):
        """Save configuration to file."""
        config.save(filepath)
    
    def merge_configs(self, base_config: BaseConfig, override_config: BaseConfig) -> BaseConfig:
        """Merge two configurations."""
        return base_config.merge(override_config)
    
    def validate_config(self, config: BaseConfig) -> Tuple[bool, List[str]]:
        """Validate configuration."""
        errors = config.validate()
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
        
        return is_valid, errors
    
    def create_experiment_config(self, experiment_name: str, 
                               base_config: CompleteConfig,
                               overrides: Dict[str, Any] = None) -> CompleteConfig:
        """Create experiment-specific configuration."""
        # Create experiment directory
        experiment_dir = self.config_dir / "experiments" / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply overrides
        if overrides:
            override_config = self._dict_to_config(overrides)
            experiment_config = base_config.merge(override_config)
        else:
            experiment_config = copy.deepcopy(base_config)
        
        # Update experiment name
        experiment_config.training.experiment_name = experiment_name
        
        # Save experiment config
        config_file = experiment_dir / "config.yaml"
        experiment_config.save(str(config_file))
        
        logger.info(f"Created experiment config: {config_file}")
        return experiment_config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CompleteConfig:
        """Convert dictionary to CompleteConfig."""
        # This is a simplified version - in practice, you'd want more robust conversion
        return CompleteConfig.from_dict(config_dict)
    
    def load_experiment_config(self, experiment_name: str) -> CompleteConfig:
        """Load experiment configuration."""
        config_file = self.config_dir / "experiments" / experiment_name / "config.yaml"
        return CompleteConfig.from_file(str(config_file))
    
    def list_experiments(self) -> List[str]:
        """List all available experiments."""
        experiments_dir = self.config_dir / "experiments"
        if not experiments_dir.exists():
            return []
        
        return [d.name for d in experiments_dir.iterdir() if d.is_dir()]
    
    def create_config_template(self, template_name: str, template_content: str):
        """Create a configuration template."""
        template_file = self.config_dir / f"{template_name}.j2"
        with open(template_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(template_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Created config template: {template_file}")
    
    def render_config_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Render a configuration template with variables."""
        template = self.template_env.get_template(f"{template_name}.j2")
        return template.render(**variables)
    
    def create_hyperparameter_sweep_config(self, base_config: CompleteConfig,
                                         sweep_params: Dict[str, List[Any]]) -> List[CompleteConfig]:
        """Create configurations for hyperparameter sweep."""
        
        # Generate all combinations
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        combinations = list(itertools.product(*param_values))
        
        sweep_configs = []
        for i, combination in enumerate(combinations):
            # Create config with current combination
            sweep_config = copy.deepcopy(base_config)
            
            # Apply parameter values
            for param_name, param_value in zip(param_names, combination):
                self._set_nested_config_value(sweep_config, param_name, param_value)
            
            # Update run name
            sweep_config.training.run_name = f"sweep_run_{i+1}"
            
            sweep_configs.append(sweep_config)
        
        return sweep_configs
    
    def _set_nested_config_value(self, config: CompleteConfig, param_path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        parts = param_path.split('.')
        current = config
        
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise ValueError(f"Invalid parameter path: {param_path}")
        
        if hasattr(current, parts[-1]):
            setattr(current, parts[-1], value)
        else:
            raise ValueError(f"Invalid parameter path: {param_path}")


class ConfigCLI:
    """Command-line interface for configuration management."""
    
    def __init__(self, config_manager: ConfigManager):
        
    """__init__ function."""
self.config_manager = config_manager
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(description="Configuration Management CLI")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Create default configs
        create_parser = subparsers.add_parser('create-defaults', help='Create default configuration files')
        
        # Validate config
        validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
        validate_parser.add_argument('config_file', help='Path to configuration file')
        
        # Create experiment
        experiment_parser = subparsers.add_parser('create-experiment', help='Create experiment configuration')
        experiment_parser.add_argument('experiment_name', help='Name of the experiment')
        experiment_parser.add_argument('--base-config', help='Base configuration file')
        experiment_parser.add_argument('--overrides', help='JSON file with overrides')
        
        # List experiments
        list_parser = subparsers.add_parser('list-experiments', help='List all experiments')
        
        # Hyperparameter sweep
        sweep_parser = subparsers.add_parser('create-sweep', help='Create hyperparameter sweep configurations')
        sweep_parser.add_argument('sweep_file', help='JSON file with sweep parameters')
        sweep_parser.add_argument('--output-dir', help='Output directory for sweep configs')
        
        return parser
    
    def run(self, args: List[str] = None):
        """Run the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if parsed_args.command == 'create-defaults':
            self._create_defaults()
        elif parsed_args.command == 'validate':
            self._validate_config(parsed_args.config_file)
        elif parsed_args.command == 'create-experiment':
            self._create_experiment(parsed_args.experiment_name, 
                                  parsed_args.base_config, 
                                  parsed_args.overrides)
        elif parsed_args.command == 'list-experiments':
            self._list_experiments()
        elif parsed_args.command == 'create-sweep':
            self._create_sweep(parsed_args.sweep_file, parsed_args.output_dir)
        else:
            parser.print_help()
    
    def _create_defaults(self) -> Any:
        """Create default configuration files."""
        created_files = self.config_manager.create_default_configs()
        print("Created default configuration files:")
        for filename, filepath in created_files.items():
            print(f"  {filename}: {filepath}")
    
    def _validate_config(self, config_file: str):
        """Validate configuration file."""
        try:
            config = CompleteConfig.from_file(config_file)
            is_valid, errors = self.config_manager.validate_config(config)
            
            if is_valid:
                print("✅ Configuration is valid!")
            else:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
    
    def _create_experiment(self, experiment_name: str, base_config: str, overrides: str):
        """Create experiment configuration."""
        try:
            # Load base config
            if base_config:
                base = CompleteConfig.from_file(base_config)
            else:
                base = CompleteConfig()
            
            # Load overrides
            override_dict = {}
            if overrides:
                with open(overrides, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    override_dict = json.load(f)
            
            # Create experiment config
            experiment_config = self.config_manager.create_experiment_config(
                experiment_name, base, override_dict
            )
            
            print(f"✅ Created experiment configuration: {experiment_name}")
        except Exception as e:
            print(f"❌ Error creating experiment: {e}")
    
    def _list_experiments(self) -> List[Any]:
        """List all experiments."""
        experiments = self.config_manager.list_experiments()
        if experiments:
            print("Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print("No experiments found.")
    
    def _create_sweep(self, sweep_file: str, output_dir: str):
        """Create hyperparameter sweep configurations."""
        try:
            # Load sweep parameters
            with open(sweep_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                sweep_params = json.load(f)
            
            # Create sweep configs
            base_config = CompleteConfig()
            sweep_configs = self.config_manager.create_hyperparameter_sweep_config(
                base_config, sweep_params
            )
            
            # Save sweep configs
            output_path = Path(output_dir or "./sweep_configs")
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, config in enumerate(sweep_configs):
                config_file = output_path / f"sweep_config_{i+1}.yaml"
                config.save(str(config_file))
            
            print(f"✅ Created {len(sweep_configs)} sweep configurations in {output_path}")
        except Exception as e:
            print(f"❌ Error creating sweep: {e}")


# Example usage
if __name__ == "__main__":
    # Create config manager
    config_manager = ConfigManager("./configs")
    
    # Create CLI
    cli = ConfigCLI(config_manager)
    
    # Run CLI
    cli.run() 