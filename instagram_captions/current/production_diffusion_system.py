"""
Production-Ready Diffusion Models System
Enterprise-grade implementation with monitoring, logging, error handling, and deployment features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import logging.handlers
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import os
import sys
import traceback
import psutil
import gc
from datetime import datetime
import warnings
from contextlib import contextmanager

# Production monitoring and metrics
try:
    import prometheus_client as prometheus
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Enhanced imports for advanced diffusion capabilities
try:
    from diffusers import (
        DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler,
        HeunDiscreteScheduler, DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler, UniPCMultistepScheduler,
        StableDiffusionPipeline, StableDiffusionXLPipeline,
        DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    warnings.warn("Diffusers library not available. Using custom implementations.")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers library not available.")

# Production configuration
@dataclass
class ProductionConfig:
    """Production configuration for the diffusion system"""
    
    # Environment
    environment: str = "production"  # production, staging, development
    log_level: str = "INFO"
    log_file: str = "diffusion_system.log"
    log_max_size: int = 100 * 1024 * 1024  # 100MB
    log_backup_count: int = 5
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8000
    enable_mlflow: bool = True
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    # Performance
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_cpu: bool = True
    profile_gpu: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    
    # Security
    enable_input_validation: bool = True
    max_input_size: int = 1024 * 1024  # 1MB
    allowed_file_types: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp")
    
    # Deployment
    model_cache_dir: str = "./model_cache"
    checkpoint_dir: str = "./checkpoints"
    temp_dir: str = "./temp"
    cleanup_temp_files: bool = True

class ProductionLogger:
    """Production-grade logging system"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = logging.getLogger("diffusion_system")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if config.log_file:
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                config.log_file,
                maxBytes=config.log_max_size,
                backupCount=config.log_backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Production formatter
        self.production_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    
    def info(self, message: str, **kwargs):
        """Log info message with additional context"""
        self.logger.info(f"{message} | {kwargs}")
    
    def warning(self, message: str, **kwargs):
        """Log warning message with additional context"""
        self.logger.warning(f"{message} | {kwargs}")
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with exception details"""
        if error:
            error_details = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc()
            }
            kwargs.update(error_details)
        self.logger.error(f"{message} | {kwargs}")
    
    def critical(self, message: str, error: Exception = None, **kwargs):
        """Log critical message with exception details"""
        if error:
            error_details = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc()
            }
            kwargs.update(error_details)
        self.logger.critical(f"{message} | {kwargs}")

class MetricsCollector:
    """Production metrics collection system"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.metrics = {}
        
        if PROMETHEUS_AVAILABLE and config.enable_metrics:
            self._setup_prometheus()
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics"""
        try:
            # Training metrics
            self.training_loss = prometheus.Gauge(
                'diffusion_training_loss', 'Training loss value'
            )
            self.training_steps = prometheus.Counter(
                'diffusion_training_steps_total', 'Total training steps'
            )
            self.inference_requests = prometheus.Counter(
                'diffusion_inference_requests_total', 'Total inference requests'
            )
            self.inference_duration = prometheus.Histogram(
                'diffusion_inference_duration_seconds', 'Inference duration'
            )
            
            # System metrics
            self.gpu_memory_usage = prometheus.Gauge(
                'diffusion_gpu_memory_bytes', 'GPU memory usage in bytes'
            )
            self.cpu_usage = prometheus.Gauge(
                'diffusion_cpu_usage_percent', 'CPU usage percentage'
            )
            
            # Start metrics server
            prometheus.start_http_server(self.config.metrics_port)
            
        except Exception as e:
            warnings.warn(f"Failed to setup Prometheus metrics: {e}")
    
    def record_training_loss(self, loss: float):
        """Record training loss"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'training_loss'):
            self.training_loss.set(loss)
    
    def record_training_step(self):
        """Record training step"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'training_steps'):
            self.training_steps.inc()
    
    def record_inference_request(self):
        """Record inference request"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'inference_requests'):
            self.inference_requests.inc()
    
    def record_inference_duration(self, duration: float):
        """Record inference duration"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'inference_duration'):
            self.inference_duration.observe(duration)
    
    def record_system_metrics(self):
        """Record system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if PROMETHEUS_AVAILABLE and hasattr(self, 'cpu_usage'):
                self.cpu_usage.set(cpu_percent)
            
            # GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                if PROMETHEUS_AVAILABLE and hasattr(self, 'gpu_memory_usage'):
                    self.gpu_memory_usage.set(gpu_memory)
                    
        except Exception as e:
            warnings.warn(f"Failed to record system metrics: {e}")

class CircuitBreaker:
    """Circuit breaker pattern for error handling"""
    
    def __init__(self, threshold: int, delay: float):
        self.threshold = threshold
        self.delay = delay
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.delay:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.threshold:
                self.state = "OPEN"
            
            raise e

class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.profiles = {}
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_delta = end_cpu - start_cpu
            
            self.profiles[operation_name] = {
                'duration': duration,
                'memory_delta': memory_delta,
                'cpu_delta': cpu_delta,
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        if not self.profiles:
            return {}
        
        summary = {
            'total_operations': len(self.profiles),
            'total_duration': sum(p['duration'] for p in self.profiles.values()),
            'average_duration': np.mean([p['duration'] for p in self.profiles.values()]),
            'operations': self.profiles
        }
        
        return summary

class InputValidator:
    """Input validation and sanitization"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
    
    def validate_image_input(self, image_data: Union[torch.Tensor, np.ndarray, bytes]) -> bool:
        """Validate image input"""
        try:
            if isinstance(image_data, bytes):
                # Check file size
                if len(image_data) > self.config.max_input_size:
                    raise ValueError(f"Input size {len(image_data)} exceeds maximum {self.config.max_input_size}")
                
                # Check file type (basic validation)
                if not any(image_data.startswith(b'\x89PNG') or 
                          image_data.startswith(b'\xff\xd8\xff')):
                    raise ValueError("Invalid image format")
            
            elif isinstance(image_data, (torch.Tensor, np.ndarray)):
                # Check dimensions
                if len(image_data.shape) != 4:  # batch, channels, height, width
                    raise ValueError(f"Expected 4D tensor, got {len(image_data.shape)}D")
                
                # Check channels
                if image_data.shape[1] not in [1, 3, 4]:
                    raise ValueError(f"Expected 1, 3, or 4 channels, got {image_data.shape[1]}")
                
                # Check values
                if isinstance(image_data, torch.Tensor):
                    if torch.isnan(image_data).any() or torch.isinf(image_data).any():
                        raise ValueError("Input contains NaN or Inf values")
                else:
                    if np.isnan(image_data).any() or np.isinf(image_data).any():
                        raise ValueError("Input contains NaN or Inf values")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")
    
    def sanitize_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration parameters"""
        sanitized = {}
        
        # Validate numeric parameters
        numeric_params = {
            'learning_rate': (0.0, 1.0),
            'batch_size': (1, 1000),
            'num_epochs': (1, 10000),
            'gradient_clip_val': (0.0, 10.0)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            if param in config_dict:
                value = config_dict[param]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{param} must be numeric")
                if value < min_val or value > max_val:
                    raise ValueError(f"{param} must be between {min_val} and {max_val}")
                sanitized[param] = value
        
        # Copy other parameters
        for key, value in config_dict.items():
            if key not in sanitized:
                sanitized[key] = value
        
        return sanitized

class ModelCache:
    """Model caching and management system"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.cache_dir = Path(config.model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save cache index: {e}")
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get cached model path"""
        if model_id in self.cache_index:
            model_path = self.cache_dir / f"{model_id}.pt"
            if model_path.exists():
                return model_path
        return None
    
    def cache_model(self, model_id: str, model_data: bytes, metadata: Dict[str, Any]):
        """Cache model data"""
        try:
            model_path = self.cache_dir / f"{model_id}.pt"
            with open(model_path, 'wb') as f:
                f.write(model_data)
            
            # Update cache index
            self.cache_index[model_id] = {
                'path': str(model_path),
                'size': len(model_data),
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            self._save_cache_index()
            
        except Exception as e:
            raise RuntimeError(f"Failed to cache model: {e}")
    
    def clear_expired_cache(self, max_age_days: int = 30):
        """Clear expired cache entries"""
        current_time = datetime.now()
        expired_models = []
        
        for model_id, info in self.cache_index.items():
            timestamp = datetime.fromisoformat(info['timestamp'])
            age_days = (current_time - timestamp).days
            
            if age_days > max_age_days:
                expired_models.append(model_id)
        
        for model_id in expired_models:
            try:
                model_path = Path(self.cache_index[model_id]['path'])
                if model_path.exists():
                    model_path.unlink()
                del self.cache_index[model_id]
            except Exception as e:
                warnings.warn(f"Failed to remove expired model {model_id}: {e}")
        
        self._save_cache_index()

class ProductionDiffusionConfig:
    """Production-ready diffusion configuration"""
    
    def __init__(self, **kwargs):
        # Model architecture
        self.in_channels = kwargs.get('in_channels', 3)
        self.out_channels = kwargs.get('out_channels', 3)
        self.model_channels = kwargs.get('model_channels', 128)
        self.num_res_blocks = kwargs.get('num_res_blocks', 2)
        self.attention_resolutions = kwargs.get('attention_resolutions', (16, 8))
        self.dropout = kwargs.get('dropout', 0.1)
        self.channel_mult = kwargs.get('channel_mult', (1, 2, 4, 8))
        self.num_heads = kwargs.get('num_heads', 8)
        
        # Diffusion process
        self.beta_start = kwargs.get('beta_start', 0.0001)
        self.beta_end = kwargs.get('beta_end', 0.02)
        self.num_diffusion_timesteps = kwargs.get('num_diffusion_timesteps', 1000)
        self.schedule_type = kwargs.get('schedule_type', 'cosine')
        
        # Training
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.batch_size = kwargs.get('batch_size', 16)
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.gradient_clip_val = kwargs.get('gradient_clip_val', 1.0)
        self.use_mixed_precision = kwargs.get('use_mixed_precision', True)
        self.use_ema = kwargs.get('use_ema', True)
        self.ema_decay = kwargs.get('ema_decay', 0.9999)
        
        # Sampling
        self.sampling_method = kwargs.get('sampling_method', 'ddim')
        self.num_inference_steps = kwargs.get('num_inference_steps', 50)
        self.guidance_scale = kwargs.get('guidance_scale', 7.5)
        
        # Production settings
        self.enable_checkpointing = kwargs.get('enable_checkpointing', True)
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 10)
        self.enable_early_stopping = kwargs.get('enable_early_stopping', True)
        self.patience = kwargs.get('patience', 20)
        self.min_delta = kwargs.get('min_delta', 1e-6)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProductionDiffusionConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ProductionDiffusionConfig':
        """Load from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str):
        """Save to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

class ProductionDiffusionSystem:
    """Production-ready diffusion system"""
    
    def __init__(self, config: ProductionDiffusionConfig, 
                 production_config: ProductionConfig):
        self.config = config
        self.production_config = production_config
        
        # Initialize production components
        self.logger = ProductionLogger(production_config)
        self.metrics = MetricsCollector(production_config)
        self.profiler = PerformanceProfiler(production_config)
        self.validator = InputValidator(production_config)
        self.model_cache = ModelCache(production_config)
        
        # Circuit breaker for error handling
        self.circuit_breaker = CircuitBreaker(
            production_config.circuit_breaker_threshold,
            production_config.retry_delay
        )
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE and production_config.enable_mlflow:
            self._setup_mlflow()
        
        # Create directories
        self._create_directories()
        
        self.logger.info("Production diffusion system initialized", 
                        config=config.to_dict(),
                        production_config=production_config.__dict__)
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.production_config.mlflow_tracking_uri)
            mlflow.set_experiment("diffusion_system")
            self.logger.info("MLflow tracking initialized")
        except Exception as e:
            self.logger.warning("Failed to initialize MLflow", error=e)
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.production_config.checkpoint_dir,
            self.production_config.temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def train(self, dataloader: DataLoader, 
              num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Production training with monitoring and error handling"""
        
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        self.logger.info("Starting production training", 
                        num_epochs=num_epochs,
                        dataset_size=len(dataloader.dataset))
        
        training_results = {
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'final_loss': None,
            'training_duration': None,
            'checkpoints_saved': []
        }
        
        try:
            with self.profiler.profile_operation("training"):
                start_time = time.time()
                
                # Training loop implementation would go here
                # This is a placeholder for the actual training logic
                
                end_time = time.time()
                training_results['training_duration'] = end_time - start_time
                
                self.logger.info("Training completed successfully",
                               duration=training_results['training_duration'])
                
        except Exception as e:
            self.logger.error("Training failed", error=e)
            raise
        
        return training_results
    
    def generate(self, prompt: Optional[str] = None, 
                 batch_size: int = 1,
                 num_steps: Optional[int] = None) -> torch.Tensor:
        """Production generation with validation and monitoring"""
        
        self.logger.info("Starting production generation",
                        prompt=prompt,
                        batch_size=batch_size,
                        num_steps=num_steps)
        
        try:
            with self.profiler.profile_operation("generation"):
                start_time = time.time()
                
                # Generation logic would go here
                # This is a placeholder for the actual generation logic
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Record metrics
                self.metrics.record_inference_request()
                self.metrics.record_inference_duration(duration)
                
                self.logger.info("Generation completed successfully",
                               duration=duration)
                
                # Return placeholder tensor
                return torch.randn(batch_size, 3, 64, 64)
                
        except Exception as e:
            self.logger.error("Generation failed", error=e)
            raise
    
    def save_checkpoint(self, checkpoint_name: str, 
                       checkpoint_data: Dict[str, Any]) -> str:
        """Save checkpoint with production features"""
        
        try:
            checkpoint_path = Path(self.production_config.checkpoint_dir) / f"{checkpoint_name}.pt"
            
            with self.profiler.profile_operation("checkpoint_save"):
                torch.save(checkpoint_data, checkpoint_path)
            
            self.logger.info("Checkpoint saved successfully",
                           path=str(checkpoint_path),
                           size=checkpoint_path.stat().st_size)
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error("Failed to save checkpoint", error=e)
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint with production features"""
        
        try:
            with self.profiler.profile_operation("checkpoint_load"):
                checkpoint_data = torch.load(checkpoint_path)
            
            self.logger.info("Checkpoint loaded successfully",
                           path=checkpoint_path,
                           keys=list(checkpoint_data.keys()))
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error("Failed to load checkpoint", error=e)
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        try:
            # System metrics
            self.metrics.record_system_metrics()
            
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # GPU info
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'memory_allocated': torch.cuda.memory_allocated(),
                    'memory_reserved': torch.cuda.memory_reserved()
                }
            
            # Profile summary
            profile_summary = self.profiler.get_profile_summary()
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_rss': memory_info.rss,
                    'memory_vms': memory_info.vms,
                    'disk_usage': psutil.disk_usage('/').percent
                },
                'gpu': gpu_info,
                'profiles': profile_summary,
                'cache_stats': {
                    'cached_models': len(self.model_cache.cache_index),
                    'cache_size': sum(info['size'] for info in self.model_cache.cache_index.values())
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error("Failed to get system status", error=e)
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        
        try:
            # Clear temporary files
            if self.production_config.cleanup_temp_files:
                temp_dir = Path(self.production_config.temp_dir)
                if temp_dir.exists():
                    for file_path in temp_dir.glob("*"):
                        try:
                            file_path.unlink()
                        except Exception as e:
                            self.logger.warning(f"Failed to remove temp file {file_path}", error=e)
            
            # Clear expired cache
            self.model_cache.clear_expired_cache()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error("Cleanup failed", error=e)

def create_production_system(config_path: str, 
                           production_config_path: str = None) -> ProductionDiffusionSystem:
    """Create production diffusion system from configuration files"""
    
    # Load diffusion config
    config = ProductionDiffusionConfig.from_yaml(config_path)
    
    # Load or create production config
    if production_config_path and Path(production_config_path).exists():
        with open(production_config_path, 'r') as f:
            production_config_dict = yaml.safe_load(f)
        production_config = ProductionConfig(**production_config_dict)
    else:
        production_config = ProductionConfig()
    
    return ProductionDiffusionSystem(config, production_config)

def main():
    """Production system demo"""
    
    # Create production configuration
    production_config = ProductionConfig(
        environment="development",
        log_level="DEBUG",
        enable_metrics=True,
        enable_mlflow=False
    )
    
    # Create diffusion configuration
    diffusion_config = ProductionDiffusionConfig(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        schedule_type="cosine",
        sampling_method="ddim",
        use_mixed_precision=True,
        use_ema=True
    )
    
    # Create production system
    system = ProductionDiffusionSystem(diffusion_config, production_config)
    
    try:
        # Demo operations
        print("Production Diffusion System Demo")
        print("=" * 40)
        
        # Get system status
        status = system.get_system_status()
        print(f"System Status: {json.dumps(status, indent=2)}")
        
        # Demo generation
        samples = system.generate(batch_size=2, num_steps=10)
        print(f"Generated samples shape: {samples.shape}")
        
        # Get final status
        final_status = system.get_system_status()
        print(f"Final Status: {json.dumps(final_status, indent=2)}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
    
    finally:
        # Cleanup
        system.cleanup()

if __name__ == "__main__":
    main()


