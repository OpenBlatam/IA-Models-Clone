"""
ML NLP Benchmark Configuration System
Real, working configuration management for ML NLP Benchmark system
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class MLNLPBenchmarkConfig:
    """Configuration class for ML NLP Benchmark system"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    workers: int = 1
    
    # Performance Configuration
    max_workers: int = 8
    batch_size: int = 1000
    cache_size: int = 10000
    chunk_size: int = 100
    
    # Optimization Configuration
    compression_level: int = 6
    quantization_bits: int = 8
    pruning_ratio: float = 0.5
    distillation_temperature: float = 3.0
    
    # GPU Configuration
    use_gpu: bool = False
    gpu_memory_fraction: float = 0.8
    cuda_visible_devices: str = "0"
    
    # Cache Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    memcached_host: str = "127.0.0.1"
    memcached_port: int = 11211
    
    # Model Configuration
    model_cache_dir: str = "./models"
    download_models: bool = True
    model_timeout: int = 300
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "ml_nlp_benchmark.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Configuration
    enable_cors: bool = True
    cors_origins: List[str] = None
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Monitoring Configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

class MLNLPBenchmarkConfigManager:
    """Configuration manager for ML NLP Benchmark system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "ml_nlp_benchmark_config.yaml"
        self.config = MLNLPBenchmarkConfig()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file or environment variables"""
        # Load from file if exists
        if os.path.exists(self.config_file):
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_env()
        
        logger.info(f"Configuration loaded from {self.config_file}")
    
    def _load_from_file(self) -> None:
        """Load configuration from YAML or JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif self.config_file.endswith('.json'):
                    config_data = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {self.config_file}")
                    return
                
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
        except Exception as e:
            logger.error(f"Error loading config file {self.config_file}: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        env_mappings = {
            'ML_NLP_BENCHMARK_HOST': 'host',
            'ML_NLP_BENCHMARK_PORT': 'port',
            'ML_NLP_BENCHMARK_DEBUG': 'debug',
            'ML_NLP_BENCHMARK_WORKERS': 'workers',
            'ML_NLP_BENCHMARK_MAX_WORKERS': 'max_workers',
            'ML_NLP_BENCHMARK_BATCH_SIZE': 'batch_size',
            'ML_NLP_BENCHMARK_CACHE_SIZE': 'cache_size',
            'ML_NLP_BENCHMARK_CHUNK_SIZE': 'chunk_size',
            'ML_NLP_BENCHMARK_COMPRESSION_LEVEL': 'compression_level',
            'ML_NLP_BENCHMARK_QUANTIZATION_BITS': 'quantization_bits',
            'ML_NLP_BENCHMARK_PRUNING_RATIO': 'pruning_ratio',
            'ML_NLP_BENCHMARK_DISTILLATION_TEMPERATURE': 'distillation_temperature',
            'ML_NLP_BENCHMARK_USE_GPU': 'use_gpu',
            'ML_NLP_BENCHMARK_GPU_MEMORY_FRACTION': 'gpu_memory_fraction',
            'ML_NLP_BENCHMARK_CUDA_VISIBLE_DEVICES': 'cuda_visible_devices',
            'ML_NLP_BENCHMARK_REDIS_HOST': 'redis_host',
            'ML_NLP_BENCHMARK_REDIS_PORT': 'redis_port',
            'ML_NLP_BENCHMARK_REDIS_DB': 'redis_db',
            'ML_NLP_BENCHMARK_REDIS_PASSWORD': 'redis_password',
            'ML_NLP_BENCHMARK_MEMCACHED_HOST': 'memcached_host',
            'ML_NLP_BENCHMARK_MEMCACHED_PORT': 'memcached_port',
            'ML_NLP_BENCHMARK_MODEL_CACHE_DIR': 'model_cache_dir',
            'ML_NLP_BENCHMARK_DOWNLOAD_MODELS': 'download_models',
            'ML_NLP_BENCHMARK_MODEL_TIMEOUT': 'model_timeout',
            'ML_NLP_BENCHMARK_LOG_LEVEL': 'log_level',
            'ML_NLP_BENCHMARK_LOG_FILE': 'log_file',
            'ML_NLP_BENCHMARK_LOG_FORMAT': 'log_format',
            'ML_NLP_BENCHMARK_ENABLE_CORS': 'enable_cors',
            'ML_NLP_BENCHMARK_CORS_ORIGINS': 'cors_origins',
            'ML_NLP_BENCHMARK_ENABLE_RATE_LIMITING': 'enable_rate_limiting',
            'ML_NLP_BENCHMARK_RATE_LIMIT_REQUESTS': 'rate_limit_requests',
            'ML_NLP_BENCHMARK_RATE_LIMIT_WINDOW': 'rate_limit_window',
            'ML_NLP_BENCHMARK_ENABLE_METRICS': 'enable_metrics',
            'ML_NLP_BENCHMARK_METRICS_PORT': 'metrics_port',
            'ML_NLP_BENCHMARK_HEALTH_CHECK_INTERVAL': 'health_check_interval'
        }
        
        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_attr in ['port', 'workers', 'max_workers', 'batch_size', 'cache_size', 'chunk_size',
                                 'compression_level', 'quantization_bits', 'redis_port', 'redis_db',
                                 'memcached_port', 'model_timeout', 'rate_limit_requests', 'rate_limit_window',
                                 'metrics_port', 'health_check_interval']:
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {env_value}")
                        continue
                
                elif config_attr in ['pruning_ratio', 'distillation_temperature', 'gpu_memory_fraction']:
                    try:
                        env_value = float(env_value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_var}: {env_value}")
                        continue
                
                elif config_attr in ['debug', 'use_gpu', 'download_models', 'enable_cors', 'enable_rate_limiting', 'enable_metrics']:
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                
                elif config_attr == 'cors_origins':
                    env_value = [origin.strip() for origin in env_value.split(',')]
                
                setattr(self.config, config_attr, env_value)
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        save_path = file_path or self.config_file
        
        try:
            config_dict = asdict(self.config)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif save_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                else:
                    logger.warning(f"Unsupported config file format: {save_path}")
                    return
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {e}")
    
    def get_config(self) -> MLNLPBenchmarkConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate port
        if not (1 <= self.config.port <= 65535):
            issues.append(f"Invalid port: {self.config.port}")
        
        # Validate workers
        if self.config.workers < 1:
            issues.append(f"Invalid workers: {self.config.workers}")
        
        # Validate batch size
        if self.config.batch_size < 1:
            issues.append(f"Invalid batch_size: {self.config.batch_size}")
        
        # Validate cache size
        if self.config.cache_size < 1:
            issues.append(f"Invalid cache_size: {self.config.cache_size}")
        
        # Validate compression level
        if not (1 <= self.config.compression_level <= 9):
            issues.append(f"Invalid compression_level: {self.config.compression_level}")
        
        # Validate quantization bits
        if not (1 <= self.config.quantization_bits <= 32):
            issues.append(f"Invalid quantization_bits: {self.config.quantization_bits}")
        
        # Validate pruning ratio
        if not (0.0 <= self.config.pruning_ratio <= 1.0):
            issues.append(f"Invalid pruning_ratio: {self.config.pruning_ratio}")
        
        # Validate GPU memory fraction
        if not (0.0 <= self.config.gpu_memory_fraction <= 1.0):
            issues.append(f"Invalid gpu_memory_fraction: {self.config.gpu_memory_fraction}")
        
        # Validate Redis port
        if not (1 <= self.config.redis_port <= 65535):
            issues.append(f"Invalid redis_port: {self.config.redis_port}")
        
        # Validate Memcached port
        if not (1 <= self.config.memcached_port <= 65535):
            issues.append(f"Invalid memcached_port: {self.config.memcached_port}")
        
        return issues
    
    def create_default_config_file(self, file_path: str = "ml_nlp_benchmark_config.yaml") -> None:
        """Create a default configuration file"""
        default_config = asdict(MLNLPBenchmarkConfig())
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Default configuration file created: {file_path}")
            
        except Exception as e:
            logger.error(f"Error creating default config file {file_path}: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        config_dict = asdict(self.config)
        
        # Remove sensitive information
        if 'redis_password' in config_dict and config_dict['redis_password']:
            config_dict['redis_password'] = '***'
        
        return {
            "config_file": self.config_file,
            "config_valid": len(self.validate_config()) == 0,
            "validation_issues": self.validate_config(),
            "settings": config_dict
        }

# Global configuration manager
config_manager = MLNLPBenchmarkConfigManager()

def get_config() -> MLNLPBenchmarkConfig:
    """Get the global configuration instance"""
    return config_manager.get_config()

def update_config(**kwargs) -> None:
    """Update the global configuration"""
    config_manager.update_config(**kwargs)

def save_config(file_path: Optional[str] = None) -> None:
    """Save the global configuration"""
    config_manager.save_config(file_path)

def load_config(config_file: Optional[str] = None) -> None:
    """Load configuration from file"""
    global config_manager
    config_manager = MLNLPBenchmarkConfigManager(config_file)











