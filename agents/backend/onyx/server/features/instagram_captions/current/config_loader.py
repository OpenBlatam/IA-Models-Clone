"""
Configuration Loader for Optimized NLP System v15.0
Follows best practices for configuration management
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import logging

class ModelConfig(BaseModel):
    """Model configuration with validation"""
    name: str = Field(default="gpt2", description="Model name")
    max_length: int = Field(default=512, ge=1, le=2048, description="Maximum sequence length")
    vocab_size: int = Field(default=50257, description="Vocabulary size")
    hidden_size: int = Field(default=768, description="Hidden layer size")
    num_layers: int = Field(default=12, description="Number of layers")
    num_attention_heads: int = Field(default=12, description="Number of attention heads")
    intermediate_size: int = Field(default=3072, description="Intermediate size")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")
    activation_function: str = Field(default="gelu", description="Activation function")
    layer_norm_eps: float = Field(default=1e-5, description="Layer normalization epsilon")
    initializer_range: float = Field(default=0.02, description="Initializer range")

class TrainingConfig(BaseModel):
    """Training configuration with validation"""
    batch_size: int = Field(default=16, ge=1, description="Batch size")
    learning_rate: float = Field(default=2e-5, gt=0, description="Learning rate")
    num_epochs: int = Field(default=3, ge=1, description="Number of epochs")
    warmup_steps: int = Field(default=500, ge=0, description="Warmup steps")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Maximum gradient norm")
    lr_scheduler: str = Field(default="cosine", description="Learning rate scheduler")
    optimizer: str = Field(default="adamw", description="Optimizer")
    betas: list = Field(default=[0.9, 0.999], description="Adam betas")
    eps: float = Field(default=1e-8, gt=0, description="Adam epsilon")

class OptimizationConfig(BaseModel):
    """Optimization configuration with validation"""
    fp16: bool = Field(default=True, description="Enable FP16")
    mixed_precision: bool = Field(default=True, description="Enable mixed precision")
    gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    dataloader_pin_memory: bool = Field(default=True, description="Pin memory for dataloader")
    dataloader_num_workers: int = Field(default=4, ge=0, description="Number of dataloader workers")
    prefetch_factor: int = Field(default=2, ge=1, description="Prefetch factor")
    persistent_workers: bool = Field(default=True, description="Persistent workers")

class HardwareConfig(BaseModel):
    """Hardware configuration with validation"""
    device: str = Field(default="auto", description="Device to use")
    cuda_visible_devices: str = Field(default="0", description="CUDA visible devices")
    memory_efficient_attention: bool = Field(default=True, description="Memory efficient attention")
    use_flash_attention: bool = Field(default=False, description="Use flash attention")
    compile_model: bool = Field(default=False, description="Compile model")

class GenerationConfig(BaseModel):
    """Generation configuration with validation"""
    max_length: int = Field(default=100, ge=1, description="Maximum generation length")
    min_length: int = Field(default=10, ge=1, description="Minimum generation length")
    temperature: float = Field(default=0.7, gt=0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, gt=0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=50, ge=1, description="Top-k sampling")
    do_sample: bool = Field(default=True, description="Enable sampling")
    num_beams: int = Field(default=1, ge=1, description="Number of beams")
    repetition_penalty: float = Field(default=1.1, gt=0, description="Repetition penalty")
    length_penalty: float = Field(default=1.0, description="Length penalty")
    no_repeat_ngram_size: int = Field(default=3, ge=0, description="No repeat n-gram size")
    early_stopping: bool = Field(default=True, description="Early stopping")
    pad_token_id: int = Field(default=50256, description="Pad token ID")
    eos_token_id: int = Field(default=50256, description="EOS token ID")

class APIConfig(BaseModel):
    """API configuration with validation"""
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8150, ge=1024, le=65535, description="API port")
    workers: int = Field(default=1, ge=1, description="Number of workers")
    reload: bool = Field(default=False, description="Enable auto-reload")
    log_level: str = Field(default="info", description="Log level")
    cors_origins: list = Field(default=["*"], description="CORS origins")
    rate_limit: int = Field(default=1000, ge=1, description="Rate limit")
    timeout: int = Field(default=60, ge=1, description="Timeout")

class DemoConfig(BaseModel):
    """Demo configuration with validation"""
    host: str = Field(default="0.0.0.0", description="Demo host")
    port: int = Field(default=8151, ge=1024, le=65535, description="Demo port")
    share: bool = Field(default=False, description="Enable sharing")
    debug: bool = Field(default=True, description="Enable debug mode")
    theme: str = Field(default="soft", description="Gradio theme")
    show_error: bool = Field(default=True, description="Show errors")

class LoggingConfig(BaseModel):
    """Logging configuration with validation"""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file: str = Field(default="logs/nlp_system.log", description="Log file")
    max_bytes: int = Field(default=10485760, description="Max bytes per log file")
    backup_count: int = Field(default=5, ge=0, description="Number of backup files")
    console: bool = Field(default=True, description="Console logging")
    file_handler: bool = Field(default=True, description="File logging")

class MonitoringConfig(BaseModel):
    """Monitoring configuration with validation"""
    enable_metrics: bool = Field(default=True, description="Enable metrics")
    metrics_port: int = Field(default=8152, ge=1024, le=65535, description="Metrics port")
    health_check_interval: int = Field(default=30, ge=1, description="Health check interval")
    memory_monitoring: bool = Field(default=True, description="Memory monitoring")
    gpu_monitoring: bool = Field(default=True, description="GPU monitoring")
    performance_tracking: bool = Field(default=True, description="Performance tracking")

class CheckpointingConfig(BaseModel):
    """Checkpointing configuration with validation"""
    save_dir: str = Field(default="checkpoints", description="Save directory")
    save_steps: int = Field(default=1000, ge=1, description="Save steps")
    save_total_limit: int = Field(default=2, ge=1, description="Total save limit")
    save_strategy: str = Field(default="steps", description="Save strategy")
    evaluation_strategy: str = Field(default="steps", description="Evaluation strategy")
    eval_steps: int = Field(default=500, ge=1, description="Evaluation steps")
    load_best_model_at_end: bool = Field(default=True, description="Load best model at end")
    metric_for_best_model: str = Field(default="eval_loss", description="Metric for best model")
    greater_is_better: bool = Field(default=False, description="Greater is better")

class ExperimentTrackingConfig(BaseModel):
    """Experiment tracking configuration with validation"""
    enable_wandb: bool = Field(default=False, description="Enable Weights & Biases")
    wandb_project: str = Field(default="nlp-system-v15", description="WandB project")
    wandb_entity: Optional[str] = Field(default=None, description="WandB entity")
    enable_tensorboard: bool = Field(default=True, description="Enable TensorBoard")
    tensorboard_dir: str = Field(default="runs", description="TensorBoard directory")
    log_metrics: bool = Field(default=True, description="Log metrics")
    log_artifacts: bool = Field(default=True, description="Log artifacts")

class SecurityConfig(BaseModel):
    """Security configuration with validation"""
    api_key_required: bool = Field(default=True, description="API key required")
    api_key_header: str = Field(default="X-API-Key", description="API key header")
    rate_limiting: bool = Field(default=True, description="Rate limiting")
    max_requests_per_minute: int = Field(default=100, ge=1, description="Max requests per minute")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: list = Field(default=["*"], description="Allowed origins")

class PerformanceConfig(BaseModel):
    """Performance configuration with validation"""
    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_size: int = Field(default=1000, ge=1, description="Cache size")
    cache_ttl: int = Field(default=3600, ge=1, description="Cache TTL")
    enable_compression: bool = Field(default=True, description="Enable compression")
    compression_level: int = Field(default=6, ge=1, le=9, description="Compression level")
    enable_profiling: bool = Field(default=False, description="Enable profiling")
    profile_dir: str = Field(default="profiles", description="Profile directory")

class ErrorHandlingConfig(BaseModel):
    """Error handling configuration with validation"""
    max_retries: int = Field(default=3, ge=0, description="Max retries")
    retry_delay: int = Field(default=1, ge=0, description="Retry delay")
    timeout: int = Field(default=30, ge=1, description="Timeout")
    graceful_shutdown: bool = Field(default=True, description="Graceful shutdown")
    error_logging: bool = Field(default=True, description="Error logging")
    fallback_responses: bool = Field(default=True, description="Fallback responses")

class DevelopmentConfig(BaseModel):
    """Development configuration with validation"""
    debug_mode: bool = Field(default=False, description="Debug mode")
    verbose_logging: bool = Field(default=False, description="Verbose logging")
    enable_tests: bool = Field(default=True, description="Enable tests")
    test_coverage: bool = Field(default=True, description="Test coverage")
    linting: bool = Field(default=True, description="Linting")
    type_checking: bool = Field(default=True, description="Type checking")

class ProductionConfig(BaseModel):
    """Production configuration with validation"""
    environment: str = Field(default="production", description="Environment")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    enable_alerting: bool = Field(default=True, description="Enable alerting")
    backup_enabled: bool = Field(default=True, description="Backup enabled")
    auto_scaling: bool = Field(default=False, description="Auto scaling")
    load_balancing: bool = Field(default=False, description="Load balancing")
    ssl_enabled: bool = Field(default=False, description="SSL enabled")

class ModelVariantsConfig(BaseModel):
    """Model variants configuration with validation"""
    gpt2: Dict[str, Any] = Field(default_factory=lambda: {"name": "gpt2", "max_length": 512, "batch_size": 16})
    gpt2_medium: Dict[str, Any] = Field(default_factory=lambda: {"name": "gpt2-medium", "max_length": 512, "batch_size": 8})
    gpt2_large: Dict[str, Any] = Field(default_factory=lambda: {"name": "gpt2-large", "max_length": 512, "batch_size": 4})
    bert_base: Dict[str, Any] = Field(default_factory=lambda: {"name": "bert-base-uncased", "max_length": 512, "batch_size": 16})
    t5_base: Dict[str, Any] = Field(default_factory=lambda: {"name": "t5-base", "max_length": 512, "batch_size": 8})

class EnvironmentConfig(BaseModel):
    """Environment configuration with validation"""
    CUDA_VISIBLE_DEVICES: str = Field(default="0", description="CUDA visible devices")
    TOKENIZERS_PARALLELISM: str = Field(default="false", description="Tokenizers parallelism")
    TRANSFORMERS_CACHE: str = Field(default="./cache", description="Transformers cache")
    HF_HOME: str = Field(default="./cache", description="Hugging Face home")
    TORCH_HOME: str = Field(default="./cache", description="PyTorch home")

@dataclass
class NLPSystemConfig:
    """Complete NLP system configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    experiment_tracking: ExperimentTrackingConfig = field(default_factory=ExperimentTrackingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    model_variants: ModelVariantsConfig = field(default_factory=ModelVariantsConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

class ConfigLoader:
    """Configuration loader with validation and best practices"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/nlp_config.yaml"
        self.config = None
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> NLPSystemConfig:
        """Load configuration from YAML file with validation"""
        try:
            config_path = Path(self.config_path)
            
            if not config_path.exists():
                self.logger.warning(f"Config file {config_path} not found, using defaults")
                return NLPSystemConfig()
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            # Validate and create configuration objects
            config = NLPSystemConfig(
                model=ModelConfig(**config_data.get('model', {})),
                training=TrainingConfig(**config_data.get('training', {})),
                optimization=OptimizationConfig(**config_data.get('optimization', {})),
                hardware=HardwareConfig(**config_data.get('hardware', {})),
                generation=GenerationConfig(**config_data.get('generation', {})),
                api=APIConfig(**config_data.get('api', {})),
                demo=DemoConfig(**config_data.get('demo', {})),
                logging=LoggingConfig(**config_data.get('logging', {})),
                monitoring=MonitoringConfig(**config_data.get('monitoring', {})),
                checkpointing=CheckpointingConfig(**config_data.get('checkpointing', {})),
                experiment_tracking=ExperimentTrackingConfig(**config_data.get('experiment_tracking', {})),
                security=SecurityConfig(**config_data.get('security', {})),
                performance=PerformanceConfig(**config_data.get('performance', {})),
                error_handling=ErrorHandlingConfig(**config_data.get('error_handling', {})),
                development=DevelopmentConfig(**config_data.get('development', {})),
                production=ProductionConfig(**config_data.get('production', {})),
                model_variants=ModelVariantsConfig(**config_data.get('model_variants', {})),
                environment=EnvironmentConfig(**config_data.get('environment', {}))
            )
            
            self.config = config
            self.logger.info(f"Configuration loaded successfully from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
            return NLPSystemConfig()
    
    def save_config(self, config: NLPSystemConfig, output_path: Optional[str] = None) -> None:
        """Save configuration to YAML file"""
        try:
            output_path = output_path or self.config_path
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            config_dict = {
                'model': config.model.dict(),
                'training': config.training.dict(),
                'optimization': config.optimization.dict(),
                'hardware': config.hardware.dict(),
                'generation': config.generation.dict(),
                'api': config.api.dict(),
                'demo': config.demo.dict(),
                'logging': config.logging.dict(),
                'monitoring': config.monitoring.dict(),
                'checkpointing': config.checkpointing.dict(),
                'experiment_tracking': config.experiment_tracking.dict(),
                'security': config.security.dict(),
                'performance': config.performance.dict(),
                'error_handling': config.error_handling.dict(),
                'development': config.development.dict(),
                'production': config.production.dict(),
                'model_variants': config.model_variants.dict(),
                'environment': config.environment.dict()
            }
            
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2, allow_unicode=True)
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get specific model configuration"""
        if not self.config:
            self.load_config()
        
        model_variants = self.config.model_variants.dict()
        return model_variants.get(model_name, model_variants['gpt2'])
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        if not self.config:
            self.load_config()
        
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.logger.info("Configuration updated")
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        if not self.config:
            self.load_config()
        
        try:
            # Validate all configuration objects
            self.config.model.validate()
            self.config.training.validate()
            self.config.optimization.validate()
            self.config.hardware.validate()
            self.config.generation.validate()
            self.config.api.validate()
            self.config.demo.validate()
            self.config.logging.validate()
            self.config.monitoring.validate()
            self.config.checkpointing.validate()
            self.config.experiment_tracking.validate()
            self.config.security.validate()
            self.config.performance.validate()
            self.config.error_handling.validate()
            self.config.development.validate()
            self.config.production.validate()
            self.config.model_variants.validate()
            self.config.environment.validate()
            
            self.logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
config_loader = ConfigLoader()
nlp_config = config_loader.load_config()

def get_config() -> NLPSystemConfig:
    """Get global configuration instance"""
    return nlp_config

def reload_config() -> NLPSystemConfig:
    """Reload configuration from file"""
    global nlp_config
    nlp_config = config_loader.load_config()
    return nlp_config

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Model: {config.model.name}")
    print(f"Training batch size: {config.training.batch_size}")
    print(f"API port: {config.api.port}")
    print(f"Demo port: {config.demo.port}")





