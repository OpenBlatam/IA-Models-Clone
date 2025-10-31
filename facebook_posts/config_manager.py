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
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
import copy
import argparse
from datetime import datetime
import shutil
import hashlib
from typing import Any, List, Dict, Optional
import asyncio
"""
⚙️ Configuration Management System for Facebook Posts AI
=======================================================
Comprehensive configuration management using YAML files for
hyperparameters, model settings, and system configuration.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration settings."""
    # Model architecture
    model_type: str = "transformer"
    input_dim: int = 768
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    
    # Model size
    vocab_size: int = 50000
    max_seq_length: int = 512
    embedding_dim: int = 768
    
    # Advanced settings
    layer_norm_eps: float = 1e-12
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    initializer_range: float = 0.02
    
    # Model variants
    use_bias: bool = True
    use_relative_position: bool = False
    use_absolute_position: bool = True
    position_embedding_type: str = "learned"

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    # Basic training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"
    lr_warmup_ratio: float = 0.1
    lr_decay_ratio: float = 0.1
    min_lr: float = 1e-6
    
    # Training strategies
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    fp16: bool = True
    bf16: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4

@dataclass
class DataConfig:
    """Data configuration settings."""
    # Dataset settings
    dataset_name: str = "facebook_posts"
    dataset_size: int = 10000
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Data processing
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"
    
    # Data augmentation
    use_augmentation: bool = False
    augmentation_prob: float = 0.5
    augmentation_methods: List[str] = field(default_factory=lambda: ["random_mask", "synonym_replacement"])
    
    # Data loading
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    
    # Caching
    cache_dir: str = "cache"
    use_cache: bool = True
    cache_size: int = 1000

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    # Mixed precision
    use_mixed_precision: bool = True
    dtype: str = "float16"
    scaler_enabled: bool = True
    
    # Memory optimization
    memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False
    memory_pool: bool = True
    
    # Data loading optimization
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Model optimization
    compile_model: bool = True
    use_torch_compile: bool = True
    optimize_for_inference: bool = False
    
    # Parallel processing
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/training.log"
    log_interval: int = 10
    
    # Experiment tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    tensorboard_dir: str = "runs"
    wandb_project: str = "facebook-posts-ai"
    wandb_entity: str = ""
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_interval: int = 100
    save_best_only: bool = True
    save_last: bool = True
    max_checkpoints: int = 5
    
    # Metrics
    eval_interval: int = 100
    log_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy", "learning_rate"])

@dataclass
class SystemConfig:
    """System and environment configuration."""
    # Device settings
    device: str = "auto"  # auto, cpu, cuda, mps
    seed: int = 42
    deterministic: bool = False
    
    # Paths
    base_dir: str = "."
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    
    # Environment
    num_threads: int = 4
    use_multiprocessing: bool = True
    max_workers: int = 4
    
    # Debugging
    debug: bool = False
    verbose: bool = False
    profile: bool = False

@dataclass
class Config:
    """Main configuration class combining all settings."""
    # Configuration metadata
    name: str = "facebook_posts_ai"
    version: str = "1.0.0"
    description: str = "Facebook Posts AI Configuration"
    created_at: str = ""
    updated_at: str = ""
    
    # Configuration sections
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Initialize timestamps and validate configuration."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def get_hash(self) -> str:
        """Get configuration hash for versioning."""
        config_str = self.to_yaml()
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class ConfigManager:
    """Configuration manager for loading, saving, and managing configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.current_config: Optional[Config] = None
    
    def create_default_config(self) -> Config:
        """Create default configuration."""
        logger.info("Creating default configuration")
        return Config()
    
    def load_config(self, config_path: Union[str, Path]) -> Config:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        # Convert dictionary to Config object
        config = self._dict_to_config(config_dict)
        self.current_config = config
        
        logger.info(f"Configuration loaded successfully: {config.name}")
        return config
    
    def save_config(self, config: Config, config_path: Union[str, Path], 
                   format: str = "yaml") -> Path:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving configuration to {config_path}")
        
        # Update timestamp
        config.updated_at = datetime.now().isoformat()
        
        with open(config_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if format.lower() == "yaml":
                f.write(config.to_yaml())
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            elif format.lower() == "json":
                f.write(config.to_json())
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved successfully")
        return config_path
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        # Handle nested dataclasses
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'performance' in config_dict:
            config_dict['performance'] = PerformanceConfig(**config_dict['performance'])
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        if 'system' in config_dict:
            config_dict['system'] = SystemConfig(**config_dict['system'])
        
        return Config(**config_dict)
    
    def create_config_template(self, template_name: str = "template") -> Path:
        """Create a configuration template file."""
        config = self.create_default_config()
        config.name = template_name
        config.description = f"Configuration template for {template_name}"
        
        template_path = self.config_dir / f"{template_name}.yaml"
        return self.save_config(config, template_path, "yaml")
    
    def list_configs(self) -> List[Path]:
        """List all available configuration files."""
        config_files = []
        for ext in ['*.yaml', '*.yml', '*.json']:
            config_files.extend(self.config_dir.glob(ext))
        return sorted(config_files)
    
    def validate_config(self, config: Config) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate model configuration
        if config.model.input_dim <= 0:
            issues.append("Model input_dim must be positive")
        if config.model.hidden_dim <= 0:
            issues.append("Model hidden_dim must be positive")
        if config.model.num_layers <= 0:
            issues.append("Model num_layers must be positive")
        if config.model.num_heads <= 0:
            issues.append("Model num_heads must be positive")
        if not 0 <= config.model.dropout <= 1:
            issues.append("Model dropout must be between 0 and 1")
        
        # Validate training configuration
        if config.training.batch_size <= 0:
            issues.append("Training batch_size must be positive")
        if config.training.learning_rate <= 0:
            issues.append("Training learning_rate must be positive")
        if config.training.num_epochs <= 0:
            issues.append("Training num_epochs must be positive")
        
        # Validate data configuration
        if not 0 < config.data.train_split < 1:
            issues.append("Data train_split must be between 0 and 1")
        if not 0 < config.data.val_split < 1:
            issues.append("Data val_split must be between 0 and 1")
        if not 0 < config.data.test_split < 1:
            issues.append("Data test_split must be between 0 and 1")
        
        # Validate splits sum to 1
        total_split = config.data.train_split + config.data.val_split + config.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            issues.append(f"Data splits must sum to 1.0, got {total_split}")
        
        return issues
    
    def merge_configs(self, base_config: Config, override_config: Config) -> Config:
        """Merge two configurations, with override_config taking precedence."""
        logger.info("Merging configurations")
        
        # Convert to dictionaries
        base_dict = base_config.to_dict()
        override_dict = override_config.to_dict()
        
        # Deep merge
        merged_dict = self._deep_merge(base_dict, override_dict)
        
        # Convert back to Config
        merged_config = self._dict_to_config(merged_dict)
        merged_config.updated_at = datetime.now().isoformat()
        
        logger.info("Configurations merged successfully")
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_experiment_config(self, base_config: Config, 
                               experiment_name: str,
                               overrides: Dict[str, Any]) -> Config:
        """Create experiment configuration with overrides."""
        logger.info(f"Creating experiment configuration: {experiment_name}")
        
        # Create override config
        override_config = Config()
        override_config.name = experiment_name
        override_config.description = f"Experiment: {experiment_name}"
        
        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                # Nested key (e.g., "training.learning_rate")
                keys = key.split('.')
                current = override_config
                for k in keys[:-1]:
                    if not hasattr(current, k):
                        setattr(current, k, {})
                    current = getattr(current, k)
                current[keys[-1]] = value
            else:
                setattr(override_config, key, value)
        
        # Merge with base config
        experiment_config = self.merge_configs(base_config, override_config)
        
        # Save experiment config
        experiment_path = self.config_dir / f"{experiment_name}.yaml"
        self.save_config(experiment_config, experiment_path)
        
        logger.info(f"Experiment configuration saved to {experiment_path}")
        return experiment_config
    
    def backup_config(self, config: Config, backup_name: str = None) -> Path:
        """Create a backup of the current configuration."""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.config_dir / f"{backup_name}.yaml"
        return self.save_config(config, backup_path)
    
    def diff_configs(self, config1: Config, config2: Config) -> Dict[str, Any]:
        """Compare two configurations and return differences."""
        logger.info("Comparing configurations")
        
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        differences = {}
        
        def compare_dicts(d1: Dict, d2: Dict, path: str = ""):
            
    """compare_dicts function."""
for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {"type": "added", "value": d2[key]}
                elif key not in d2:
                    differences[current_path] = {"type": "removed", "value": d1[key]}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {
                        "type": "modified",
                        "old_value": d1[key],
                        "new_value": d2[key]
                    }
        
        compare_dicts(dict1, dict2)
        
        logger.info(f"Found {len(differences)} differences")
        return differences

def create_preset_configs() -> Dict[str, Config]:
    """Create preset configurations for common use cases."""
    presets = {}
    
    # Small model for quick experiments
    small_config = Config()
    small_config.name = "small_model"
    small_config.description = "Small model for quick experiments"
    small_config.model.input_dim = 256
    small_config.model.hidden_dim = 128
    small_config.model.num_layers = 2
    small_config.model.num_heads = 4
    small_config.training.batch_size = 16
    small_config.training.num_epochs = 10
    small_config.data.dataset_size = 1000
    presets["small"] = small_config
    
    # Medium model for standard training
    medium_config = Config()
    medium_config.name = "medium_model"
    medium_config.description = "Medium model for standard training"
    medium_config.model.input_dim = 512
    medium_config.model.hidden_dim = 256
    medium_config.model.num_layers = 4
    medium_config.model.num_heads = 8
    medium_config.training.batch_size = 32
    medium_config.training.num_epochs = 50
    medium_config.data.dataset_size = 5000
    presets["medium"] = medium_config
    
    # Large model for production
    large_config = Config()
    large_config.name = "large_model"
    large_config.description = "Large model for production use"
    large_config.model.input_dim = 768
    large_config.model.hidden_dim = 512
    large_config.model.num_layers = 6
    large_config.model.num_heads = 8
    large_config.training.batch_size = 64
    large_config.training.num_epochs = 100
    large_config.data.dataset_size = 10000
    large_config.performance.use_mixed_precision = True
    large_config.performance.gradient_checkpointing = True
    presets["large"] = large_config
    
    # High performance configuration
    high_perf_config = Config()
    high_perf_config.name = "high_performance"
    high_perf_config.description = "High performance configuration"
    high_perf_config.model.input_dim = 768
    high_perf_config.model.hidden_dim = 512
    high_perf_config.model.num_layers = 6
    high_perf_config.training.batch_size = 128
    high_perf_config.training.num_epochs = 100
    high_perf_config.performance.use_mixed_precision = True
    high_perf_config.performance.gradient_checkpointing = True
    high_perf_config.performance.compile_model = True
    high_perf_config.performance.memory_efficient_attention = True
    high_perf_config.data.dataset_size = 20000
    presets["high_performance"] = high_perf_config
    
    return presets

def main():
    """Main function to demonstrate configuration management."""
    # Create config manager
    config_manager = ConfigManager("configs")
    
    # Create default configuration
    default_config = config_manager.create_default_config()
    
    # Save default configuration
    config_path = config_manager.save_config(default_config, "configs/default.yaml")
    logger.info(f"Default configuration saved to {config_path}")
    
    # Create preset configurations
    presets = create_preset_configs()
    for name, config in presets.items():
        preset_path = config_manager.save_config(config, f"configs/{name}.yaml")
        logger.info(f"Preset configuration '{name}' saved to {preset_path}")
    
    # Create configuration template
    template_path = config_manager.create_config_template("template")
    logger.info(f"Configuration template saved to {template_path}")
    
    # List all configurations
    configs = config_manager.list_configs()
    logger.info(f"Available configurations: {[c.name for c in configs]}")
    
    # Validate configuration
    issues = config_manager.validate_config(default_config)
    if issues:
        logger.warning(f"Configuration validation issues: {issues}")
    else:
        logger.info("Configuration validation passed")
    
    # Create experiment configuration
    experiment_overrides = {
        "training.learning_rate": 2e-4,
        "training.batch_size": 64,
        "model.hidden_dim": 256,
        "custom.experiment_id": "exp_001"
    }
    
    experiment_config = config_manager.create_experiment_config(
        default_config, "experiment_001", experiment_overrides
    )
    
    # Compare configurations
    differences = config_manager.diff_configs(default_config, experiment_config)
    logger.info(f"Configuration differences: {len(differences)} changes")
    
    # Print configuration summary
    logger.info("Configuration Management System Ready!")
    logger.info(f"Config directory: {config_manager.config_dir}")
    logger.info(f"Available presets: {list(presets.keys())}")

match __name__:
    case "__main__":
    main() 