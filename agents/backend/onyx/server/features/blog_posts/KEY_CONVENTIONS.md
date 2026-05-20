# Key Conventions - Production-Grade Blog Post System

## Table of Contents
1. [Code Style & Formatting](#code-style--formatting)
2. [Naming Conventions](#naming-conventions)
3. [Architecture Patterns](#architecture-patterns)
4. [Error Handling](#error-handling)
5. [Logging & Monitoring](#logging--monitoring)
6. [Performance Optimization](#performance-optimization)
7. [Testing Standards](#testing-standards)
8. [Documentation](#documentation)
9. [Security](#security)
10. [Deployment](#deployment)

## Code Style & Formatting

### Python Standards
- **PEP 8**: Strict adherence to Python style guide
- **Black**: Code formatting with 88 character line length
- **isort**: Import sorting and organization
- **flake8**: Linting and style checking
- **mypy**: Static type checking

### Type Hints
```python
from typing import Optional, List, Dict, Union, Tuple, Callable
import torch
from torch import Tensor

def process_batch(
    data: Tensor,
    model: torch.nn.Module,
    device: Optional[torch.device] = None
) -> Tuple[Tensor, Dict[str, float]]:
    """Process a batch of data through the model."""
    pass
```

### Docstrings
```python
def train_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 100
) -> Dict[str, List[float]]:
    """
    Train a PyTorch model with comprehensive logging.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If model is not properly initialized
        RuntimeError: If CUDA is not available when expected
        
    Example:
        >>> model = MyModel()
        >>> metrics = train_model(model, train_loader, epochs=50)
    """
    pass
```

## Naming Conventions

### Files & Directories
- **snake_case**: All Python files and directories
- **PascalCase**: Class names in files
- **UPPER_CASE**: Constants and environment variables

### Variables & Functions
```python
# Variables: snake_case
learning_rate = 0.001
batch_size = 32
model_config = {...}

# Functions: snake_case with descriptive names
def calculate_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    pass

def load_pretrained_model(model_name: str) -> torch.nn.Module:
    pass

# Classes: PascalCase
class DiffusionModel(torch.nn.Module):
    pass

class TrainingPipeline:
    pass

# Constants: UPPER_CASE
DEFAULT_LEARNING_RATE = 0.001
MAX_BATCH_SIZE = 128
MODEL_CACHE_DIR = "models/"
```

### PyTorch Specific
```python
# Model layers: descriptive names
self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
self.attention_layer = MultiHeadAttention(d_model, num_heads)
self.normalization = nn.LayerNorm(d_model)

# Loss functions: descriptive names
self.reconstruction_loss = nn.MSELoss()
self.kl_divergence_loss = nn.KLDivLoss()
self.adversarial_loss = nn.BCELoss()
```

## Architecture Patterns

### Functional Programming
```python
from functools import partial
from typing import Callable, TypeVar, Generic

T = TypeVar('T')
U = TypeVar('U')

def pipeline(
    data: T,
    *functions: Callable[[T], U]
) -> U:
    """Apply a series of functions to data."""
    result = data
    for func in functions:
        result = func(result)
    return result

# Usage
processed_data = pipeline(
    raw_data,
    preprocess,
    normalize,
    augment
)
```

### Dependency Injection
```python
from dataclasses import dataclass
from typing import Protocol

class ModelProtocol(Protocol):
    def forward(self, x: Tensor) -> Tensor:
        ...

@dataclass
class TrainingConfig:
    model: ModelProtocol
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    device: torch.device

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
```

### Factory Pattern
```python
from enum import Enum
from typing import Dict, Type

class ModelType(Enum):
    DIFFUSION = "diffusion"
    TRANSFORMER = "transformer"
    CNN = "cnn"

class ModelFactory:
    _models: Dict[ModelType, Type[torch.nn.Module]] = {
        ModelType.DIFFUSION: DiffusionModel,
        ModelType.TRANSFORMER: TransformerModel,
        ModelType.CNN: CNNModel
    }
    
    @classmethod
    def create_model(cls, model_type: ModelType, **kwargs) -> torch.nn.Module:
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type](**kwargs)
```

## Error Handling

### Custom Exceptions
```python
class BlogPostSystemError(Exception):
    """Base exception for blog post system."""
    pass

class ModelLoadingError(BlogPostSystemError):
    """Raised when model loading fails."""
    pass

class TrainingError(BlogPostSystemError):
    """Raised when training fails."""
    pass

class ValidationError(BlogPostSystemError):
    """Raised when data validation fails."""
    pass
```

### Error Handling Patterns
```python
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def error_handler(operation: str):
    """Context manager for consistent error handling."""
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        logger.error(f"CUDA out of memory during {operation}")
        raise TrainingError(f"GPU memory exhausted during {operation}")
    except Exception as e:
        logger.error(f"Unexpected error during {operation}: {e}")
        raise BlogPostSystemError(f"Failed to {operation}: {e}")

# Usage
with error_handler("model training"):
    train_model(model, dataloader)
```

### Input Validation
```python
from pydantic import BaseModel, validator
from typing import Optional

class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    device: str
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 1:
            raise ValueError('Learning rate must be between 0 and 1')
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0 or v > 1024:
            raise ValueError('Batch size must be between 1 and 1024')
        return v
```

## Logging & Monitoring

### Structured Logging
```python
import structlog
import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info(
    "Training started",
    model_name="diffusion_model",
    batch_size=32,
    learning_rate=0.001
)
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
TRAINING_STEPS = Counter('training_steps_total', 'Total training steps')
TRAINING_LOSS = Histogram('training_loss', 'Training loss distribution')
GPU_MEMORY_USAGE = Gauge('gpu_memory_bytes', 'GPU memory usage in bytes')

class MetricsCollector:
    def __init__(self):
        self.start_time = None
    
    def start_training(self):
        self.start_time = time.time()
    
    def record_step(self, loss: float):
        TRAINING_STEPS.inc()
        TRAINING_LOSS.observe(loss)
    
    def record_gpu_memory(self, memory_bytes: int):
        GPU_MEMORY_USAGE.set(memory_bytes)
```

## Performance Optimization

### GPU Memory Management
```python
import torch
from contextlib import contextmanager

@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory optimization."""
    try:
        # Clear cache before operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
    finally:
        # Clear cache after operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage
with gpu_memory_manager():
    model = load_large_model()
    result = model(input_data)
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.scaler = GradScaler()
    
    def train_step(self, data: Tensor, target: Tensor) -> float:
        self.model.zero_grad()
        
        with autocast():
            output = self.model(data)
            loss = self.criterion(output, target)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### Caching Strategy
```python
from functools import lru_cache
import hashlib
import pickle

class ModelCache:
    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, model_config: dict) -> str:
        """Generate cache key from model configuration."""
        config_str = pickle.dumps(model_config, sort_keys=True)
        return hashlib.md5(config_str).hexdigest()
    
    def get_cached_model(self, model_config: dict) -> Optional[torch.nn.Module]:
        """Retrieve cached model if available."""
        cache_key = self._get_cache_key(model_config)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
        
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        return None
    
    def cache_model(self, model: torch.nn.Module, model_config: dict):
        """Cache model for future use."""
        cache_key = self._get_cache_key(model_config)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
        torch.save(model, cache_path)
```

## Testing Standards

### Unit Testing
```python
import pytest
import torch
from unittest.mock import Mock, patch

class TestDiffusionModel:
    @pytest.fixture
    def model(self):
        return DiffusionModel(
            input_dim=784,
            hidden_dim=512,
            output_dim=784
        )
    
    @pytest.fixture
    def sample_data(self):
        return torch.randn(32, 784)
    
    def test_model_forward(self, model, sample_data):
        """Test model forward pass."""
        output = model(sample_data)
        assert output.shape == (32, 784)
        assert not torch.isnan(output).any()
    
    def test_model_training_step(self, model, sample_data):
        """Test training step."""
        optimizer = torch.optim.Adam(model.parameters())
        loss = model.training_step(sample_data, optimizer)
        assert isinstance(loss, float)
        assert loss > 0
```

### Integration Testing
```python
class TestTrainingPipeline:
    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        config = TrainingConfig(
            model_type=ModelType.DIFFUSION,
            batch_size=16,
            epochs=2
        )
        
        pipeline = TrainingPipeline(config)
        metrics = pipeline.train()
        
        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert metrics["train_loss"][-1] < metrics["train_loss"][0]
```

### Performance Testing
```python
import time
import psutil

class TestPerformance:
    def test_memory_usage(self, model):
        """Test memory usage during inference."""
        initial_memory = psutil.virtual_memory().used
        
        # Run inference
        data = torch.randn(100, 784)
        with torch.no_grad():
            _ = model(data)
        
        final_memory = psutil.virtual_memory().used
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 1GB)
        assert memory_increase < 1024 * 1024 * 1024
    
    def test_inference_speed(self, model):
        """Test inference speed."""
        data = torch.randn(100, 784)
        
        start_time = time.time()
        with torch.no_grad():
            _ = model(data)
        end_time = time.time()
        
        inference_time = end_time - start_time
        # Should complete within 1 second
        assert inference_time < 1.0
```

## Documentation

### Code Documentation
```python
class DiffusionModel(torch.nn.Module):
    """
    Advanced diffusion model for image generation.
    
    This model implements a state-of-the-art diffusion process
    for generating high-quality images from noise.
    
    Attributes:
        encoder (nn.Module): Encoder network for feature extraction
        decoder (nn.Module): Decoder network for image generation
        noise_scheduler (NoiseScheduler): Noise scheduling mechanism
        
    Example:
        >>> model = DiffusionModel(
        ...     input_channels=3,
        ...     hidden_dim=512,
        ...     num_layers=12
        ... )
        >>> noise = torch.randn(1, 3, 64, 64)
        >>> generated_image = model.generate(noise, num_steps=1000)
    """
    
    def __init__(self, input_channels: int, hidden_dim: int, num_layers: int):
        """
        Initialize the diffusion model.
        
        Args:
            input_channels: Number of input channels
            hidden_dim: Hidden dimension size
            num_layers: Number of transformer layers
        """
        super().__init__()
        # Implementation details...
```

### API Documentation
```python
def generate_image(
    prompt: str,
    model_name: str = "stable-diffusion-v1-5",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
) -> PIL.Image.Image:
    """
    Generate an image from a text prompt using diffusion models.
    
    This function provides a high-level interface for image generation
    using various diffusion models from Hugging Face.
    
    Args:
        prompt: Text description of the desired image
        model_name: Name of the diffusion model to use
        num_inference_steps: Number of denoising steps
        guidance_scale: Scale for classifier-free guidance
        seed: Random seed for reproducible generation
        
    Returns:
        Generated image as PIL Image
        
    Raises:
        ModelLoadingError: If model fails to load
        GenerationError: If image generation fails
        
    Example:
        >>> image = generate_image(
        ...     prompt="A beautiful sunset over mountains",
        ...     model_name="stable-diffusion-v1-5",
        ...     num_inference_steps=50
        ... )
        >>> image.save("generated_image.png")
    """
    pass
```

## Security

### Input Validation
```python
import re
from typing import Union

def validate_prompt(prompt: str) -> str:
    """
    Validate and sanitize user input prompts.
    
    Args:
        prompt: User-provided text prompt
        
    Returns:
        Sanitized prompt
        
    Raises:
        ValidationError: If prompt is invalid
    """
    if not prompt or not prompt.strip():
        raise ValidationError("Prompt cannot be empty")
    
    if len(prompt) > 1000:
        raise ValidationError("Prompt too long (max 1000 characters)")
    
    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\']', '', prompt.strip())
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'system:',
        r'exec\(',
        r'eval\(',
        r'import\s+os',
        r'__import__'
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise ValidationError(f"Suspicious pattern detected: {pattern}")
    
    return sanitized
```

### Model Security
```python
class SecureModelLoader:
    """Secure model loading with validation."""
    
    ALLOWED_MODELS = {
        "stable-diffusion-v1-5",
        "stable-diffusion-v2-1",
        "openjourney",
        "dreamlike-photoreal-2.0"
    }
    
    @classmethod
    def load_model(cls, model_name: str) -> torch.nn.Module:
        """Load model with security validation."""
        if model_name not in cls.ALLOWED_MODELS:
            raise ValidationError(f"Model {model_name} not in allowed list")
        
        # Additional security checks
        model_path = f"models/{model_name}"
        if not os.path.exists(model_path):
            raise ModelLoadingError(f"Model path {model_path} not found")
        
        return torch.load(model_path, map_location='cpu')
```

## Deployment

### Environment Configuration
```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    gpu_enabled: bool = os.getenv("GPU_ENABLED", "true").lower() == "true"
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "32"))
    cache_dir: str = os.getenv("CACHE_DIR", "/tmp/cache")
    
    def validate(self):
        """Validate configuration."""
        if self.environment not in ["development", "staging", "production"]:
            raise ValueError(f"Invalid environment: {self.environment}")
        
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
```

### Health Checks
```python
class HealthChecker:
    """System health monitoring."""
    
    def check_gpu_health(self) -> Dict[str, bool]:
        """Check GPU health and availability."""
        health_status = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_ok": True,
            "models_loaded": True
        }
        
        if torch.cuda.is_available():
            try:
                # Check GPU memory
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                
                if memory_allocated > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                    health_status["gpu_memory_ok"] = False
                    
            except Exception as e:
                health_status["gpu_memory_ok"] = False
        
        return health_status
    
    def check_model_health(self) -> Dict[str, bool]:
        """Check model loading and inference health."""
        try:
            # Test model loading
            model = DiffusionModel()
            test_input = torch.randn(1, 784)
            with torch.no_grad():
                _ = model(test_input)
            
            return {"models_loaded": True, "inference_working": True}
        except Exception as e:
            return {"models_loaded": False, "inference_working": False}
```

### Monitoring Integration
```python
class SystemMonitor:
    """System monitoring and alerting."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric for monitoring."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        })
    
    def get_system_stats(self) -> Dict[str, float]:
        """Get current system statistics."""
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "gpu_memory_usage": self._get_gpu_memory_usage(),
            "active_requests": len(self.metrics.get("requests", [])),
            "average_response_time": self._calculate_avg_response_time()
        }
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            return (memory_allocated / memory_total) * 100
        return 0.0
```

---

## Summary

This Key Conventions document establishes comprehensive standards for:

1. **Code Quality**: PEP 8 compliance, type hints, and structured documentation
2. **Architecture**: Functional programming, dependency injection, and design patterns
3. **Reliability**: Comprehensive error handling and validation
4. **Observability**: Structured logging and metrics collection
5. **Performance**: GPU optimization, caching, and mixed precision training
6. **Testing**: Unit, integration, and performance testing standards
7. **Security**: Input validation and model security measures
8. **Deployment**: Environment configuration and health monitoring

These conventions ensure the production-grade blog post system maintains high quality, reliability, and maintainability while following industry best practices for deep learning applications. 