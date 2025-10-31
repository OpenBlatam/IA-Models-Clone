"""
Configuration Management System
Follows key convention: Use configuration files (e.g., YAML) for hyperparameters and model settings
"""

import yaml
import json
import toml
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
import logging
import os
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION BASE CLASSES
# ============================================================================

@dataclass
class BaseConfig:
    """Base configuration class with common methods"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to YAML: {filepath}")
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to JSON: {filepath}")
    
    def to_toml(self, filepath: str):
        """Save configuration to TOML file"""
        with open(filepath, 'w') as f:
            toml.dump(self.to_dict(), f)
        logger.info(f"Configuration saved to TOML: {filepath}")
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'BaseConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'BaseConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_toml(cls, filepath: str) -> 'BaseConfig':
        """Load configuration from TOML file"""
        with open(filepath, 'r') as f:
            config_dict = toml.load(filepath)
        return cls(**config_dict)

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

@dataclass
class ModelConfig(BaseConfig):
    """Base model configuration"""
    model_type: str
    model_name: str
    version: str = "1.0.0"
    description: str = ""
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.model_type:
            raise ValueError("model_type is required")
        if not self.model_name:
            raise ValueError("model_name is required")
        return True

@dataclass
class TransformerConfig(ModelConfig):
    """Transformer model configuration"""
    model_type: str = "transformer"
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 2048
    max_length: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-12
    output_size: int = 1000
    use_positional_encoding: bool = True
    use_relative_position: bool = False
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    
    def validate(self) -> bool:
        """Validate transformer configuration"""
        super().validate()
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        
        return True

@dataclass
class DiffusionConfig(ModelConfig):
    """Diffusion model configuration"""
    model_type: str = "diffusion"
    input_dim: int = 768
    hidden_dim: int = 1024
    time_dim: int = 128
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    dropout: float = 0.1
    noise_schedule: str = "linear"  # linear, cosine, quadratic
    use_learned_variance: bool = False
    use_class_conditioning: bool = False
    num_classes: Optional[int] = None
    use_attention: bool = True
    attention_heads: int = 8
    attention_layers: int = 4
    
    def validate(self) -> bool:
        """Validate diffusion configuration"""
        super().validate()
        
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be less than beta_end")
        
        if self.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        
        return True

@dataclass
class UNetConfig(ModelConfig):
    """UNet model configuration"""
    model_type: str = "unet"
    input_dim: int = 768
    hidden_dim: int = 1024
    output_dim: int = 768
    num_encoder_blocks: int = 4
    num_decoder_blocks: int = 4
    dropout: float = 0.1
    use_skip_connections: bool = True
    use_batch_norm: bool = True
    use_residual: bool = True
    activation: str = "relu"
    kernel_size: int = 3
    padding: int = 1
    use_attention: bool = False
    attention_heads: int = 8
    
    def validate(self) -> bool:
        """Validate UNet configuration"""
        super().validate()
        
        if self.num_encoder_blocks <= 0 or self.num_decoder_blocks <= 0:
            raise ValueError("Number of blocks must be positive")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        
        return True

# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

@dataclass
class TrainingConfig(BaseConfig):
    """Base training configuration"""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    save_checkpoint_every: int = 5
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1
    use_mixed_precision: bool = True
    use_grad_scaler: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    
    def validate(self) -> bool:
        """Validate training configuration"""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        return True

@dataclass
class OptimizerConfig(BaseConfig):
    """Optimizer configuration"""
    optimizer_type: str = "adamw"  # adam, adamw, sgd, rmsprop
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.9  # for SGD
    nesterov: bool = False  # for SGD
    centered: bool = False  # for RMSprop
    alpha: float = 0.99  # for RMSprop
    
    def validate(self) -> bool:
        """Validate optimizer configuration"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        if self.beta1 < 0 or self.beta1 > 1:
            raise ValueError("beta1 must be between 0 and 1")
        
        if self.beta2 < 0 or self.beta2 > 1:
            raise ValueError("beta2 must be between 0 and 1")
        
        return True

@dataclass
class SchedulerConfig(BaseConfig):
    """Learning rate scheduler configuration"""
    scheduler_type: str = "cosine"  # step, cosine, exponential, reduce_on_plateau
    learning_rate: float = 1e-4
    step_size: int = 30  # for StepLR
    gamma: float = 0.1  # for StepLR
    T_max: int = 100  # for CosineAnnealingLR
    eta_min: float = 0.0  # for CosineAnnealingLR
    factor: float = 0.1  # for ReduceLROnPlateau
    patience: int = 10  # for ReduceLROnPlateau
    min_lr: float = 1e-7  # for ReduceLROnPlateau
    warmup_steps: int = 0  # for warmup
    warmup_factor: float = 0.1  # for warmup
    
    def validate(self) -> bool:
        """Validate scheduler configuration"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        
        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError("gamma must be between 0 and 1")
        
        if self.T_max <= 0:
            raise ValueError("T_max must be positive")
        
        return True

@dataclass
class LossConfig(BaseConfig):
    """Loss function configuration"""
    loss_type: str = "cross_entropy"  # cross_entropy, mse, l1, focal, dice
    reduction: str = "mean"  # mean, sum, none
    label_smoothing: float = 0.0  # for CrossEntropyLoss
    focal_alpha: float = 1.0  # for FocalLoss
    focal_gamma: float = 2.0  # for FocalLoss
    dice_smooth: float = 1e-6  # for DiceLoss
    class_weights: Optional[List[float]] = None
    
    def validate(self) -> bool:
        """Validate loss configuration"""
        if self.label_smoothing < 0 or self.label_smoothing > 1:
            raise ValueError("label_smoothing must be between 0 and 1")
        
        if self.focal_alpha < 0:
            raise ValueError("focal_alpha must be non-negative")
        
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        
        return True

# ============================================================================
# DATA CONFIGURATIONS
# ============================================================================

@dataclass
class DataConfig(BaseConfig):
    """Data configuration"""
    data_dir: str = "./data"
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"
    max_length: int = 512
    text_column: str = "text"
    target_column: str = "label"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    use_cache: bool = True
    cache_dir: str = "./cache"
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate data configuration"""
        if not self.data_dir:
            raise ValueError("data_dir is required")
        
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        return True

@dataclass
class TokenizerConfig(BaseConfig):
    """Tokenizer configuration"""
    tokenizer_type: str = "wordpiece"  # wordpiece, bpe, sentencepiece, custom
    vocab_size: int = 30000
    max_length: int = 512
    padding: str = "max_length"  # max_length, longest, do_not_pad
    truncation: bool = True
    return_tensors: str = "pt"  # pt, tf, np
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    do_lower_case: bool = True
    strip_accents: bool = True
    unk_token: str = "[UNK]"
    sep_token: str = "[SEP]"
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    mask_token: str = "[MASK]"
    
    def validate(self) -> bool:
        """Validate tokenizer configuration"""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        return True

# ============================================================================
# EVALUATION CONFIGURATIONS
# ============================================================================

@dataclass
class EvaluationConfig(BaseConfig):
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    eval_batch_size: int = 64
    num_eval_samples: Optional[int] = None
    save_predictions: bool = True
    predictions_file: str = "predictions.json"
    save_confusion_matrix: bool = True
    confusion_matrix_file: str = "confusion_matrix.png"
    use_tqdm: bool = True
    verbose: bool = True
    
    def validate(self) -> bool:
        """Validate evaluation configuration"""
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        
        if not self.metrics:
            raise ValueError("At least one metric must be specified")
        
        return True

# ============================================================================
# SYSTEM CONFIGURATIONS
# ============================================================================

@dataclass
class SystemConfig(BaseConfig):
    """System configuration"""
    device: str = "auto"  # auto, cpu, cuda, mps
    num_gpus: int = 1
    use_mixed_precision: bool = True
    use_grad_scaler: bool = True
    use_data_parallel: bool = False
    use_distributed: bool = False
    rank: int = 0
    world_size: int = 1
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    memory_efficient: bool = False
    compile_model: bool = False  # torch.compile
    
    def validate(self) -> bool:
        """Validate system configuration"""
        if self.num_gpus < 0:
            raise ValueError("num_gpus must be non-negative")
        
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        
        return True

@dataclass
class LoggingConfig(BaseConfig):
    """Logging configuration"""
    log_level: str = "INFO"
    log_file: str = "./logs/training.log"
    log_dir: str = "./logs"
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    use_wandb: bool = False
    wandb_project: str = "nlp_project"
    wandb_entity: Optional[str] = None
    save_logs: bool = True
    log_every_n_steps: int = 10
    log_every_n_epochs: int = 1
    
    def validate(self) -> bool:
        """Validate logging configuration"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        
        return True

# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

@dataclass
class MainConfig(BaseConfig):
    """Main configuration class that combines all configurations"""
    model: Union[TransformerConfig, DiffusionConfig, UNetConfig]
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    evaluation: EvaluationConfig
    system: SystemConfig
    logging: LoggingConfig
    
    # Metadata
    project_name: str = "nlp_project"
    experiment_name: str = "experiment_1"
    description: str = ""
    author: str = ""
    created_at: str = ""
    version: str = "1.0.0"
    
    def validate(self) -> bool:
        """Validate all configurations"""
        try:
            self.model.validate()
            self.training.validate()
            self.optimizer.validate()
            self.scheduler.validate()
            self.loss.validate()
            self.data.validate()
            self.tokenizer.validate()
            self.evaluation.validate()
            self.system.validate()
            self.logging.validate()
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        return self.model
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration"""
        return self.training
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """Get optimizer configuration"""
        return self.optimizer
    
    def get_scheduler_config(self) -> SchedulerConfig:
        """Get scheduler configuration"""
        return self.scheduler
    
    def get_loss_config(self) -> LossConfig:
        """Get loss configuration"""
        return self.loss
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration"""
        return self.data
    
    def get_tokenizer_config(self) -> TokenizerConfig:
        """Get tokenizer configuration"""
        return self.tokenizer
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration"""
        return self.evaluation
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.system
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return self.logging

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigurationManager:
    """Manager for handling configuration files"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs = {}
    
    def create_default_config(self, model_type: str = "transformer") -> MainConfig:
        """Create default configuration for specified model type"""
        
        # Create model config based on type
        if model_type == "transformer":
            model_config = TransformerConfig()
        elif model_type == "diffusion":
            model_config = DiffusionConfig()
        elif model_type == "unet":
            model_config = UNetConfig()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create other configs
        training_config = TrainingConfig()
        optimizer_config = OptimizerConfig()
        scheduler_config = SchedulerConfig()
        loss_config = LossConfig()
        data_config = DataConfig()
        tokenizer_config = TokenizerConfig()
        evaluation_config = EvaluationConfig()
        system_config = SystemConfig()
        logging_config = LoggingConfig()
        
        # Create main config
        main_config = MainConfig(
            model=model_config,
            training=training_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss=loss_config,
            data=data_config,
            tokenizer=tokenizer_config,
            evaluation=evaluation_config,
            system=system_config,
            logging=logging_config
        )
        
        return main_config
    
    def save_config(self, config: MainConfig, filename: str, format: str = "yaml"):
        """Save configuration to file"""
        filepath = self.config_dir / filename
        
        if format.lower() == "yaml":
            config.to_yaml(str(filepath))
        elif format.lower() == "json":
            config.to_json(str(filepath))
        elif format.lower() == "toml":
            config.to_toml(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {filepath}")
        return filepath
    
    def load_config(self, filepath: str) -> MainConfig:
        """Load configuration from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        # Determine format from extension
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() == '.toml':
            config_dict = toml.load(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Create configuration object
        config = self._dict_to_config(config_dict)
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Configuration validation failed")
        
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> MainConfig:
        """Convert dictionary to configuration object"""
        
        # Extract model config
        model_dict = config_dict.get('model', {})
        model_type = model_dict.get('model_type', 'transformer')
        
        if model_type == 'transformer':
            model_config = TransformerConfig(**model_dict)
        elif model_type == 'diffusion':
            model_config = DiffusionConfig(**model_dict)
        elif model_type == 'unet':
            model_config = UNetConfig(**model_dict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create other configs
        training_config = TrainingConfig(**config_dict.get('training', {}))
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        scheduler_config = SchedulerConfig(**config_dict.get('scheduler', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        tokenizer_config = TokenizerConfig(**config_dict.get('tokenizer', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Create main config
        main_config = MainConfig(
            model=model_config,
            training=training_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss=loss_config,
            data=data_config,
            tokenizer=tokenizer_config,
            evaluation=evaluation_config,
            system=system_config,
            logging=logging_config,
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'training', 'optimizer', 'scheduler', 
                           'loss', 'data', 'tokenizer', 'evaluation', 
                           'system', 'logging']}
        )
        
        return main_config
    
    def create_config_template(self, model_type: str = "transformer", 
                             filename: str = None) -> str:
        """Create configuration template file"""
        
        config = self.create_default_config(model_type)
        
        if filename is None:
            filename = f"{model_type}_config.yaml"
        
        filepath = self.save_config(config, filename, "yaml")
        return str(filepath)
    
    def validate_config_file(self, filepath: str) -> bool:
        """Validate configuration file"""
        try:
            config = self.load_config(filepath)
            return config.validate()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def merge_configs(self, base_config: MainConfig, 
                     override_config: Dict[str, Any]) -> MainConfig:
        """Merge base configuration with override values"""
        
        # Convert base config to dict
        base_dict = base_config.to_dict()
        
        # Recursively merge
        merged_dict = self._recursive_merge(base_dict, override_config)
        
        # Convert back to config object
        merged_config = self._dict_to_config(merged_dict)
        
        return merged_config
    
    def _recursive_merge(self, base: Dict[str, Any], 
                        override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._recursive_merge(result[key], value)
            else:
                result[key] = value
        
        return result

# ============================================================================
# CONFIGURATION VALIDATOR
# ============================================================================

class ConfigurationValidator:
    """Validator for configuration files"""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """Validate model configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Model validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_training_config(config: TrainingConfig) -> List[str]:
        """Validate training configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Training validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_optimizer_config(config: OptimizerConfig) -> List[str]:
        """Validate optimizer configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Optimizer validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_scheduler_config(config: SchedulerConfig) -> List[str]:
        """Validate scheduler configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Scheduler validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_loss_config(config: LossConfig) -> List[str]:
        """Validate loss configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Loss validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_data_config(config: DataConfig) -> List[str]:
        """Validate data configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Data validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_tokenizer_config(config: TokenizerConfig) -> List[str]:
        """Validate tokenizer configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Tokenizer validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_evaluation_config(config: EvaluationConfig) -> List[str]:
        """Validate evaluation configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Evaluation validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_system_config(config: SystemConfig) -> List[str]:
        """Validate system configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"System validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_logging_config(config: LoggingConfig) -> List[str]:
        """Validate logging configuration"""
        errors = []
        
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Logging validation error: {e}")
        
        return errors
    
    @staticmethod
    def validate_main_config(config: MainConfig) -> List[str]:
        """Validate main configuration"""
        errors = []
        
        # Validate individual components
        errors.extend(ConfigurationValidator.validate_model_config(config.model))
        errors.extend(ConfigurationValidator.validate_training_config(config.training))
        errors.extend(ConfigurationValidator.validate_optimizer_config(config.optimizer))
        errors.extend(ConfigurationValidator.validate_scheduler_config(config.scheduler))
        errors.extend(ConfigurationValidator.validate_loss_config(config.loss))
        errors.extend(ConfigurationValidator.validate_data_config(config.data))
        errors.extend(ConfigurationValidator.validate_tokenizer_config(config.tokenizer))
        errors.extend(ConfigurationValidator.validate_evaluation_config(config.evaluation))
        errors.extend(ConfigurationValidator.validate_system_config(config.system))
        errors.extend(ConfigurationValidator.validate_logging_config(config.logging))
        
        # Validate main config
        try:
            config.validate()
        except Exception as e:
            errors.append(f"Main configuration validation error: {e}")
        
        return errors

# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================

class ConfigurationTemplates:
    """Templates for common configuration scenarios"""
    
    @staticmethod
    def get_transformer_classification_template() -> Dict[str, Any]:
        """Get transformer classification template"""
        return {
            "model": {
                "model_type": "transformer",
                "model_name": "transformer_classifier",
                "vocab_size": 10000,
                "hidden_size": 512,
                "num_layers": 6,
                "num_heads": 8,
                "ff_dim": 2048,
                "max_length": 512,
                "dropout": 0.1,
                "output_size": 10
            },
            "training": {
                "num_epochs": 100,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 0.01
            },
            "optimizer": {
                "optimizer_type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01
            },
            "scheduler": {
                "scheduler_type": "cosine",
                "learning_rate": 1e-4,
                "T_max": 100
            },
            "loss": {
                "loss_type": "cross_entropy",
                "label_smoothing": 0.1
            },
            "data": {
                "data_dir": "./data",
                "max_length": 512,
                "batch_size": 32
            },
            "evaluation": {
                "metrics": ["accuracy", "f1", "precision", "recall"]
            }
        }
    
    @staticmethod
    def get_diffusion_generation_template() -> Dict[str, Any]:
        """Get diffusion generation template"""
        return {
            "model": {
                "model_type": "diffusion",
                "model_name": "diffusion_generator",
                "input_dim": 768,
                "hidden_dim": 1024,
                "time_dim": 128,
                "num_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02
            },
            "training": {
                "num_epochs": 200,
                "batch_size": 64,
                "learning_rate": 1e-4
            },
            "optimizer": {
                "optimizer_type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01
            },
            "scheduler": {
                "scheduler_type": "cosine",
                "learning_rate": 1e-4,
                "T_max": 200
            },
            "loss": {
                "loss_type": "mse"
            },
            "data": {
                "data_dir": "./data",
                "batch_size": 64
            },
            "evaluation": {
                "metrics": ["reconstruction_error", "sample_quality"]
            }
        }
    
    @staticmethod
    def get_unet_segmentation_template() -> Dict[str, Any]:
        """Get UNet segmentation template"""
        return {
            "model": {
                "model_type": "unet",
                "model_name": "unet_segmentation",
                "input_dim": 3,
                "hidden_dim": 64,
                "output_dim": 1,
                "num_encoder_blocks": 4,
                "num_decoder_blocks": 4,
                "use_skip_connections": True
            },
            "training": {
                "num_epochs": 150,
                "batch_size": 16,
                "learning_rate": 1e-4
            },
            "optimizer": {
                "optimizer_type": "adam",
                "learning_rate": 1e-4
            },
            "scheduler": {
                "scheduler_type": "reduce_on_plateau",
                "learning_rate": 1e-4,
                "factor": 0.1,
                "patience": 15
            },
            "loss": {
                "loss_type": "dice",
                "dice_smooth": 1e-6
            },
            "data": {
                "data_dir": "./data",
                "batch_size": 16
            },
            "evaluation": {
                "metrics": ["dice", "iou", "precision", "recall"]
            }
        }

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def main():
    """Example usage of the configuration management system"""
    
    # Create configuration manager
    config_manager = ConfigurationManager("./configs")
    
    # Create default transformer configuration
    print("Creating default transformer configuration...")
    transformer_config = config_manager.create_default_config("transformer")
    
    # Save configuration
    config_file = config_manager.save_config(transformer_config, "transformer_config.yaml")
    print(f"Configuration saved to: {config_file}")
    
    # Load configuration
    print("Loading configuration...")
    loaded_config = config_manager.load_config(str(config_file))
    
    # Validate configuration
    print("Validating configuration...")
    validator = ConfigurationValidator()
    errors = validator.validate_main_config(loaded_config)
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed!")
    
    # Create configuration template
    print("Creating configuration template...")
    template_file = config_manager.create_config_template("diffusion", "diffusion_template.yaml")
    print(f"Template created: {template_file}")
    
    # Show configuration structure
    print("\nConfiguration structure:")
    print(f"  Model: {loaded_config.model.model_type}")
    print(f"  Training epochs: {loaded_config.training.num_epochs}")
    print(f"  Learning rate: {loaded_config.optimizer.learning_rate}")
    print(f"  Batch size: {loaded_config.data.batch_size}")
    
    print("\nConfiguration management system ready!")

if __name__ == "__main__":
    main()


