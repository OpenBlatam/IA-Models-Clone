#!/usr/bin/env python3
"""
Enhanced Configuration Management System for Frontier Model Training
Provides dynamic configuration loading, validation, and environment-specific settings.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import jsonschema
from jsonschema import validate, ValidationError

console = Console()

class Environment(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "deepseek-ai/deepseek-r1"
    use_deepspeed: bool = False
    fp16: bool = True
    bf16: bool = False
    torch_dtype: str = "auto"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True
    model_revision: Optional[str] = None

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    max_steps: int = 1000
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    num_cycles: int = 1
    seed: int = 42
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

@dataclass
class OptimizationConfig:
    """Optimization configuration parameters."""
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_8bit_optimizer: bool = False
    use_cpu_offload: bool = False
    use_activation_checkpointing: bool = True
    use_attention_slicing: bool = True
    use_sequence_parallelism: bool = False
    use_cudnn_benchmark: bool = True
    use_tf32: bool = True
    use_channels_last: bool = True
    use_compile: bool = True

@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration parameters."""
    use_deepspeed: bool = False
    zero_stage: int = 2
    offload_optimizer: bool = True
    offload_param: bool = False
    gradient_clipping: float = 1.0
    fp16: bool = True
    bf16: bool = False
    config_file: Optional[str] = None

@dataclass
class DeepSeekConfig:
    """DeepSeek-specific configuration parameters."""
    model_type: str = "deepseek"
    max_position_embeddings: int = 8192
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    use_rotary_embeddings: bool = True
    use_alibi: bool = False
    use_flash_attention_2: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 4096

@dataclass
class ParallelConfig:
    """Parallel processing configuration."""
    attention: bool = True
    mlp: bool = True
    layernorm: bool = True
    embedding: bool = True
    output: bool = True
    residual: bool = True
    ffn: bool = True
    attention_output: bool = True
    mlp_output: bool = True
    layernorm_output: bool = True
    embedding_output: bool = True
    residual_output: bool = True
    ffn_output: bool = True
    attention_input: bool = True
    mlp_input: bool = True
    layernorm_input: bool = True
    embedding_input: bool = True
    residual_input: bool = True
    ffn_input: bool = True

@dataclass
class KalmanConfig:
    """Kalman filter configuration."""
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    memory_size: int = 1000

@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    backend: str = "nccl"
    world_size: int = -1
    rank: int = -1
    master_addr: str = "localhost"
    master_port: str = "29500"

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "your_dataset"
    config: str = "your_config"
    train_split: str = "train"
    test_split: str = "test"
    cache_dir: Optional[str] = None
    streaming: bool = False

@dataclass
class RewardConfig:
    """Reward functions configuration."""
    functions: List[str] = field(default_factory=lambda: ["accuracy", "format", "tag_count"])
    cosine_min_value_wrong: float = 0.0
    cosine_max_value_wrong: float = -0.5
    cosine_min_value_correct: float = 0.5
    cosine_max_value_correct: float = 1.0
    cosine_max_len: int = 1000
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0
    code_language: str = "python"

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    use_wandb: bool = False
    use_mlflow: bool = False
    use_tensorboard: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    metrics_log_interval: int = 10
    checkpoint_interval: int = 500

@dataclass
class FrontierConfig:
    """Main configuration class for Frontier Model Training."""
    environment: Environment = Environment.DEVELOPMENT
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Metadata
    version: str = "1.0.0"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    description: Optional[str] = None

class ConfigManager:
    """Enhanced configuration manager with validation and environment support."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[FrontierConfig] = None
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for configuration validation."""
        schema = {
            "type": "object",
            "properties": {
                "environment": {"type": "string", "enum": [e.value for e in Environment]},
                "model": {"type": "object"},
                "training": {"type": "object"},
                "optimization": {"type": "object"},
                "deepspeed": {"type": "object"},
                "deepseek": {"type": "object"},
                "parallel": {"type": "object"},
                "kalman": {"type": "object"},
                "distributed": {"type": "object"},
                "dataset": {"type": "object"},
                "rewards": {"type": "object"},
                "monitoring": {"type": "object"}
            },
            "required": ["environment", "model", "training"]
        }
        return schema
    
    def load_config(self, config_path: Optional[str] = None) -> FrontierConfig:
        """Load configuration from file."""
        if config_path:
            self.config_path = config_path
            
        if not self.config_path:
            raise ValueError("No configuration file path provided")
            
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        # Load configuration based on file extension
        if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
            
        # Validate configuration
        try:
            validate(instance=config_data, schema=self.schema)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
            
        # Convert to FrontierConfig
        self.config = self._dict_to_config(config_data)
        return self.config
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> FrontierConfig:
        """Convert dictionary to FrontierConfig object."""
        # Handle environment
        env = Environment(config_data.get('environment', 'development'))
        
        # Create config sections
        model = ModelConfig(**config_data.get('model', {}))
        training = TrainingConfig(**config_data.get('training', {}))
        optimization = OptimizationConfig(**config_data.get('optimization', {}))
        deepspeed = DeepSpeedConfig(**config_data.get('deepspeed', {}))
        deepseek = DeepSeekConfig(**config_data.get('deepseek', {}))
        parallel = ParallelConfig(**config_data.get('parallel', {}))
        kalman = KalmanConfig(**config_data.get('kalman', {}))
        distributed = DistributedConfig(**config_data.get('distributed', {}))
        dataset = DatasetConfig(**config_data.get('dataset', {}))
        rewards = RewardConfig(**config_data.get('rewards', {}))
        monitoring = MonitoringConfig(**config_data.get('monitoring', {}))
        
        return FrontierConfig(
            environment=env,
            model=model,
            training=training,
            optimization=optimization,
            deepspeed=deepspeed,
            deepseek=deepseek,
            parallel=parallel,
            kalman=kalman,
            distributed=distributed,
            dataset=dataset,
            rewards=rewards,
            monitoring=monitoring,
            version=config_data.get('version', '1.0.0'),
            created_at=config_data.get('created_at'),
            updated_at=config_data.get('updated_at'),
            description=config_data.get('description')
        )
    
    def save_config(self, config: FrontierConfig, output_path: str, format: str = 'yaml') -> None:
        """Save configuration to file."""
        config_dict = asdict(config)
        
        # Convert enum to string
        config_dict['environment'] = config.environment.value
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'yaml':
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {format}")
            
        console.print(f"[green]Configuration saved to: {output_path}[/green]")
    
    def create_default_config(self, output_path: str, environment: Environment = Environment.DEVELOPMENT) -> None:
        """Create a default configuration file."""
        config = FrontierConfig(environment=environment)
        self.save_config(config, output_path)
        console.print(f"[green]Default configuration created at: {output_path}[/green]")
    
    def validate_config(self, config: FrontierConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate model configuration
        if not config.model.name:
            issues.append("Model name is required")
            
        # Validate training configuration
        if config.training.batch_size <= 0:
            issues.append("Batch size must be positive")
            
        if config.training.learning_rate <= 0:
            issues.append("Learning rate must be positive")
            
        # Validate optimization configuration
        if config.optimization.use_amp and config.optimization.use_8bit_optimizer:
            issues.append("Cannot use both AMP and 8-bit optimizer simultaneously")
            
        # Validate DeepSpeed configuration
        if config.deepspeed.use_deepspeed and not config.deepspeed.config_file:
            issues.append("DeepSpeed config file is required when using DeepSpeed")
            
        return issues
    
    def display_config(self, config: FrontierConfig) -> None:
        """Display configuration in a formatted table."""
        table = Table(title="Frontier Model Configuration")
        table.add_column("Section", style="cyan")
        table.add_column("Parameter", style="magenta")
        table.add_column("Value", style="green")
        
        # Environment
        table.add_row("Environment", "Type", config.environment.value)
        
        # Model
        table.add_row("Model", "Name", config.model.name)
        table.add_row("Model", "Use DeepSpeed", str(config.model.use_deepspeed))
        table.add_row("Model", "FP16", str(config.model.fp16))
        
        # Training
        table.add_row("Training", "Batch Size", str(config.training.batch_size))
        table.add_row("Training", "Learning Rate", str(config.training.learning_rate))
        table.add_row("Training", "Max Steps", str(config.training.max_steps))
        
        # Optimization
        table.add_row("Optimization", "Use AMP", str(config.optimization.use_amp))
        table.add_row("Optimization", "Gradient Checkpointing", str(config.optimization.use_gradient_checkpointing))
        table.add_row("Optimization", "Flash Attention", str(config.optimization.use_flash_attention))
        
        console.print(table)
    
    def get_environment_config(self, environment: Environment) -> FrontierConfig:
        """Get environment-specific configuration."""
        if not self.config:
            raise ValueError("No configuration loaded")
            
        # Create environment-specific config
        env_config = FrontierConfig(
            environment=environment,
            model=self.config.model,
            training=self.config.training,
            optimization=self.config.optimization,
            deepspeed=self.config.deepspeed,
            deepseek=self.config.deepseek,
            parallel=self.config.parallel,
            kalman=self.config.kalman,
            distributed=self.config.distributed,
            dataset=self.config.dataset,
            rewards=self.config.rewards,
            monitoring=self.config.monitoring
        )
        
        # Apply environment-specific overrides
        if environment == Environment.PRODUCTION:
            env_config.training.batch_size = min(env_config.training.batch_size, 4)
            env_config.optimization.use_cpu_offload = True
            env_config.monitoring.log_level = "WARNING"
        elif environment == Environment.DEVELOPMENT:
            env_config.training.max_steps = min(env_config.training.max_steps, 100)
            env_config.monitoring.log_level = "DEBUG"
        elif environment == Environment.TESTING:
            env_config.training.max_steps = 10
            env_config.training.batch_size = 1
            
        return env_config

def main():
    """Main function for configuration management CLI."""
    parser = argparse.ArgumentParser(description="Frontier Model Configuration Manager")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create-default", type=str, help="Create default configuration file")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--display", action="store_true", help="Display configuration")
    parser.add_argument("--environment", type=str, choices=[e.value for e in Environment], 
                       help="Environment type")
    
    args = parser.parse_args()
    
    manager = ConfigManager()
    
    if args.create_default:
        env = Environment(args.environment) if args.environment else Environment.DEVELOPMENT
        manager.create_default_config(args.create_default, env)
        return
        
    if args.config:
        try:
            config = manager.load_config(args.config)
            
            if args.validate:
                issues = manager.validate_config(config)
                if issues:
                    console.print("[red]Configuration validation failed:[/red]")
                    for issue in issues:
                        console.print(f"  - {issue}")
                else:
                    console.print("[green]Configuration validation passed[/green]")
                    
            if args.display:
                manager.display_config(config)
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    else:
        console.print("[yellow]No configuration file specified. Use --help for usage information.[/yellow]")

if __name__ == "__main__":
    main()
