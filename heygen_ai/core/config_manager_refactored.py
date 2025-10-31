"""
Refactored Configuration Manager for HeyGen AI

This module provides a clean, type-safe configuration management system
following deep learning best practices with proper validation and error handling.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import yaml
from pydantic import BaseModel, Field, validator
import torch

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Pydantic model for model configuration validation."""
    
    # Transformer Models
    gpt2: Dict[str, Any] = Field(default_factory=dict)
    bert: Dict[str, Any] = Field(default_factory=dict)
    t5: Dict[str, Any] = Field(default_factory=dict)
    
    # Diffusion Models
    stable_diffusion: Dict[str, Any] = Field(default_factory=dict)
    stable_diffusion_xl: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"


class TrainingConfig(BaseModel):
    """Pydantic model for training configuration validation."""
    
    # General Settings
    seed: int = Field(default=42, ge=0)
    num_epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=8, gt=0)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    warmup_steps: int = Field(default=100, ge=0)
    save_steps: int = Field(default=500, gt=0)
    eval_steps: int = Field(default=500, gt=0)
    logging_steps: int = Field(default=100, gt=0)
    
    # Learning Rate
    initial_lr: float = Field(default=5e-5, gt=0)
    min_lr: float = Field(default=1e-6, gt=0)
    scheduler_type: str = Field(default="cosine", regex="^(cosine|linear|constant)$")
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0)
    
    # Mixed Precision
    mixed_precision_enabled: bool = Field(default=True)
    dtype: str = Field(default="fp16", regex="^(fp16|bf16|fp32)$")
    autocast: bool = Field(default=True)
    scaler: bool = Field(default=True)
    
    # Early Stopping
    early_stopping_enabled: bool = Field(default=True)
    patience: int = Field(default=3, gt=0)
    min_delta: float = Field(default=0.001, gt=0)
    monitor: str = Field(default="val_loss")
    mode: str = Field(default="min", regex="^(min|max)$")
    
    @validator('min_lr')
    def validate_min_lr(cls, v, values):
        if 'initial_lr' in values and v >= values['initial_lr']:
            raise ValueError('min_lr must be less than initial_lr')
        return v


class OptimizationConfig(BaseModel):
    """Pydantic model for optimization configuration validation."""
    
    # LoRA Settings
    lora_enabled: bool = Field(default=False)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.1, ge=0, le=1)
    lora_target_modules: List[str] = Field(default_factory=list)
    lora_bias: str = Field(default="none", regex="^(none|all|lora_only)$")
    lora_task_type: str = Field(default="CAUSAL_LM")
    
    # Quantization
    quantization_enabled: bool = Field(default=False)
    quantization_dtype: str = Field(default="int8", regex="^(int8|fp16|bf16)$")
    quantization_backend: str = Field(default="auto", regex="^(auto|x86|arm|cuda)$")
    
    # Gradient Checkpointing
    gradient_checkpointing_enabled: bool = Field(default=True)
    memory_efficient: bool = Field(default=True)


class DataConfig(BaseModel):
    """Pydantic model for data configuration validation."""
    
    # Dataset Settings
    train_file: str = Field(default="data/train.json")
    validation_file: str = Field(default="data/validation.json")
    test_file: str = Field(default="data/test.json")
    max_length: int = Field(default=512, gt=0)
    truncation: bool = Field(default=True)
    padding: str = Field(default="max_length", regex="^(max_length|longest|do_not_pad)$")
    return_tensors: str = Field(default="pt", regex="^(pt|tf|np)$")
    
    # DataLoader Settings
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = Field(default=True)
    shuffle: bool = Field(default=True)
    drop_last: bool = Field(default=False)
    persistent_workers: bool = Field(default=True)
    
    @validator('train_file', 'validation_file', 'test_file')
    def validate_file_paths(cls, v):
        if v and not os.path.exists(v):
            logger.warning(f"File path does not exist: {v}")
        return v


class HardwareConfig(BaseModel):
    """Pydantic model for hardware configuration validation."""
    
    # Device Settings
    device_type: str = Field(default="auto", regex="^(auto|cuda|cpu|mps)$")
    cuda_device: int = Field(default=0, ge=0)
    mixed_precision: bool = Field(default=True)
    
    # Memory Management
    max_memory_usage: float = Field(default=0.9, gt=0, le=1)
    enable_memory_efficient_attention: bool = Field(default=True)
    enable_attention_slicing: bool = Field(default=True)
    enable_vae_slicing: bool = Field(default=True)
    
    # Multi-GPU
    distributed_enabled: bool = Field(default=False)
    backend: str = Field(default="nccl", regex="^(nccl|gloo)$")
    world_size: int = Field(default=1, gt=0)
    rank: int = Field(default=0, ge=0)
    
    @validator('device_type')
    def validate_device_type(cls, v):
        if v == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif v == "mps" and not hasattr(torch.backends, 'mps'):
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return v


class MonitoringConfig(BaseModel):
    """Pydantic model for monitoring configuration validation."""
    
    # Experiment Tracking
    experiment_tracking_enabled: bool = Field(default=True)
    backend: str = Field(default="wandb", regex="^(wandb|tensorboard|mlflow)$")
    project_name: str = Field(default="heygen-ai")
    run_name: str = Field(default="experiment-001")
    log_interval: int = Field(default=100, gt=0)
    
    # Logging
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file: str = Field(default="logs/training.log")
    console_logging: bool = Field(default=True)
    
    # Metrics
    save_metrics: bool = Field(default=True)
    metrics_file: str = Field(default="metrics/training_metrics.json")
    plot_metrics: bool = Field(default=True)


class PerformanceConfig(BaseModel):
    """Pydantic model for performance configuration validation."""
    
    # Torch Compile
    torch_compile_enabled: bool = Field(default=True)
    compile_mode: str = Field(default="max-autotune", regex="^(default|reduce-overhead|max-autotune)$")
    dynamic: bool = Field(default=True)
    fullgraph: bool = Field(default=True)
    backend: str = Field(default="inductor", regex="^(inductor|aot_eager|aot_ts|debug)$")
    
    # Attention Optimization
    enable_flash_attention: bool = Field(default=True)
    enable_xformers: bool = Field(default=True)
    enable_memory_efficient_attention: bool = Field(default=True)
    attention_backend: str = Field(default="auto", regex="^(auto|flash|xformers|triton|standard)$")
    
    # Memory Optimization
    enable_gradient_checkpointing: bool = Field(default=True)
    enable_activation_checkpointing: bool = Field(default=True)
    enable_selective_checkpointing: bool = Field(default=True)
    memory_pool_optimization: bool = Field(default=True)
    
    # Batch Optimization
    enable_dynamic_batching: bool = Field(default=True)
    max_batch_size: int = Field(default=32, gt=0)
    min_batch_size: int = Field(default=1, gt=0)
    optimization_interval: int = Field(default=100, gt=0)


class HeyGenAIConfig(BaseModel):
    """Main configuration class for HeyGen AI."""
    
    models: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    class Config:
        extra = "allow"


@dataclass
class ConfigManager:
    """Refactored configuration manager with best practices."""
    
    config_path: Path = field(default_factory=lambda: Path("config/model_config.yaml"))
    config: Optional[HeyGenAIConfig] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    
    def __post_init__(self):
        """Initialize configuration after dataclass creation."""
        self._load_config()
        self._setup_logging()
        self._validate_environment()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file with error handling."""
        try:
            if not self.config_path.exists():
                self.logger.warning(f"Config file not found: {self.config_path}")
                self.logger.info("Using default configuration")
                self.config = HeyGenAIConfig()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Validate configuration
            self.config = HeyGenAIConfig(**config_data)
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if not self.config:
            return
        
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config.monitoring.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.monitoring.log_level),
            format=self.config.monitoring.log_format,
            handlers=[
                logging.FileHandler(self.config.monitoring.log_file),
                logging.StreamHandler() if self.config.monitoring.console_logging else logging.NullHandler()
            ]
        )
        
        self.logger.info("Logging configured successfully")
    
    def _validate_environment(self) -> None:
        """Validate environment and hardware capabilities."""
        if not self.config:
            return
        
        # Check CUDA availability
        if self.config.hardware.device_type == "cuda":
            if not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available")
                self.config.hardware.device_type = "cpu"
            else:
                cuda_count = torch.cuda.device_count()
                self.logger.info(f"CUDA available with {cuda_count} device(s)")
                
                if self.config.hardware.cuda_device >= cuda_count:
                    self.logger.warning(f"CUDA device {self.config.hardware.cuda_device} not available")
                    self.config.hardware.cuda_device = 0
        
        # Check MPS availability
        if self.config.hardware.device_type == "mps":
            if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
                self.logger.warning("MPS requested but not available")
                self.config.hardware.device_type = "cpu"
        
        # Validate memory settings
        if self.config.hardware.device_type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = int(total_memory * self.config.hardware.max_memory_usage)
            self.logger.info(f"GPU memory: {total_memory / 1024**3:.1f}GB, "
                           f"Max usage: {max_memory / 1024**3:.1f}GB")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if not self.config:
            return {}
        
        model_configs = {
            'gpt2': self.config.models.gpt2,
            'bert': self.config.models.bert,
            't5': self.config.models.t5,
            'stable_diffusion': self.config.models.stable_diffusion,
            'stable_diffusion_xl': self.config.models.stable_diffusion_xl
        }
        
        if model_name not in model_configs:
            self.logger.warning(f"Model configuration not found: {model_name}")
            return {}
        
        return model_configs[model_name]
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        if not self.config:
            return TrainingConfig()
        return self.config.training
    
    def get_optimization_config(self) -> OptimizationConfig:
        """Get optimization configuration."""
        if not self.config:
            return OptimizationConfig()
        return self.config.optimization
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        if not self.config:
            return DataConfig()
        return self.config.data
    
    def get_hardware_config(self) -> HardwareConfig:
        """Get hardware configuration."""
        if not self.config:
            return HardwareConfig()
        return self.config.hardware
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        if not self.config:
            return MonitoringConfig()
        return self.config.monitoring
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        if not self.config:
            return PerformanceConfig()
        return self.config.performance
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        if not self.config:
            return
        
        try:
            # Update configuration
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    self.logger.warning(f"Unknown configuration key: {key}")
            
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        if not self.config:
            return
        
        try:
            save_path = path or self.config_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict and save
            config_dict = self.config.dict()
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        if not self.config:
            return False
        
        try:
            # Validate using Pydantic
            self.config.validate(self.config.dict())
            self.logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_device(self) -> torch.device:
        """Get the appropriate device based on configuration."""
        if not self.config:
            return torch.device("cpu")
        
        device_type = self.config.hardware.device_type
        
        if device_type == "auto":
            if torch.cuda.is_available():
                device_type = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_type = "mps"
            else:
                device_type = "cpu"
        
        if device_type == "cuda":
            return torch.device(f"cuda:{self.config.hardware.cuda_device}")
        elif device_type == "mps":
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_mixed_precision_dtype(self) -> torch.dtype:
        """Get the appropriate mixed precision dtype."""
        if not self.config:
            return torch.float32
        
        dtype_str = self.config.training.dtype
        
        if dtype_str == "fp16":
            return torch.float16
        elif dtype_str == "bf16":
            return torch.bfloat16
        else:
            return torch.float32


# Factory function for easy usage
def create_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Create a configuration manager instance."""
    if config_path:
        config_path = Path(config_path)
    
    return ConfigManager(config_path=config_path)


# Example usage
if __name__ == "__main__":
    # Create configuration manager
    config_manager = create_config_manager()
    
    # Get specific configurations
    training_config = config_manager.get_training_config()
    hardware_config = config_manager.get_hardware_config()
    
    print(f"Training epochs: {training_config.num_epochs}")
    print(f"Device type: {hardware_config.device_type}")
    print(f"Mixed precision: {training_config.mixed_precision_enabled}")
    
    # Validate configuration
    if config_manager.validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed")

