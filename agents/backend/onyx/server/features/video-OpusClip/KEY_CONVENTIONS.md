# 📚 Referencias Oficiales y Mejores Prácticas

**Siempre consulta la documentación oficial para mejores prácticas y APIs actualizadas:**

- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index)
- [Diffusers (Hugging Face)](https://huggingface.co/docs/diffusers/index)
- [Gradio](https://gradio.app/docs/)

> **Nota:** Antes de usar o actualizar cualquier API, revisa la documentación oficial correspondiente.

---

# 🎯 Key Conventions - Video-OpusClip System

## Overview

This document outlines the essential coding conventions, architectural patterns, and best practices for the Video-OpusClip AI video processing system. These conventions ensure consistency, maintainability, and optimal performance across all components.

## 📋 Table of Contents

1. [Code Style & Formatting](#code-style--formatting)
2. [Naming Conventions](#naming-conventions)
3. [Architecture Patterns](#architecture-patterns)
4. [File Organization](#file-organization)
5. [Import Organization](#import-organization)
6. [Error Handling](#error-handling)
7. [Logging & Monitoring](#logging--monitoring)
8. [Performance Optimization](#performance-optimization)
9. [Testing Standards](#testing-standards)
10. [Documentation Standards](#documentation-standards)
11. [PyTorch & AI Conventions](#pytorch--ai-conventions)
12. [API Design](#api-design)
13. [Configuration Management](#configuration-management)
14. [Security Conventions](#security-conventions)

---

## 🎨 Code Style & Formatting

### Python Standards
- **PEP 8**: Strict adherence to Python style guide
- **Black**: Code formatting with 88 character line length
- **isort**: Import sorting and organization
- **flake8**: Linting and style checking
- **mypy**: Static type checking

### Type Hints
```python
from typing import Optional, List, Dict, Union, Tuple, Callable, Any
import torch
from torch import Tensor
from pathlib import Path

def process_video_batch(
    videos: List[Tensor],
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    batch_size: int = 32
) -> Tuple[List[Tensor], Dict[str, float]]:
    """Process a batch of videos through the model."""
    pass
```

### Docstrings
```python
def generate_video_captions(
    video: Tensor,
    model: torch.nn.Module,
    max_length: int = 100,
    temperature: float = 0.7
) -> List[str]:
    """
    Generate captions for video content using AI model.
    
    Args:
        video: Input video tensor of shape (batch, frames, channels, height, width)
        model: Pre-trained caption generation model
        max_length: Maximum caption length in tokens
        temperature: Sampling temperature for text generation
        
    Returns:
        List of generated captions
        
    Raises:
        ValueError: If video tensor has invalid shape
        RuntimeError: If model is not properly initialized
        
    Example:
        >>> video = torch.randn(1, 16, 3, 224, 224)
        >>> captions = generate_video_captions(video, model)
    """
    pass
```

---

## 🏷️ Naming Conventions

### Files & Directories
- **snake_case**: All Python files and directories
- **PascalCase**: Class names in files
- **UPPER_CASE**: Constants and environment variables

```python
# Files: snake_case
video_processor.py
caption_generator.py
performance_monitor.py

# Directories: snake_case
video_processing/
caption_generation/
performance_optimization/
```

### Variables & Functions
```python
# Variables: snake_case with descriptive names
learning_rate = 0.001
batch_size = 32
video_processing_config = {...}
is_gpu_available = torch.cuda.is_available()
has_valid_video_format = check_video_format(video_path)

# Functions: snake_case with descriptive names
def calculate_video_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    pass

def load_pretrained_diffusion_model(model_name: str) -> torch.nn.Module:
    pass

def validate_video_input(video_path: Path) -> bool:
    pass

# Classes: PascalCase
class VideoProcessor(torch.nn.Module):
    pass

class CaptionGenerator:
    pass

class PerformanceMonitor:
    pass

# Constants: UPPER_CASE
DEFAULT_LEARNING_RATE = 0.001
MAX_BATCH_SIZE = 128
VIDEO_CACHE_DIR = "cache/videos/"
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov"]
```

### PyTorch Specific
```python
# Model layers: descriptive names
self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
self.attention_layer = MultiHeadAttention(d_model, num_heads)
self.normalization = nn.LayerNorm(d_model)
self.dropout = nn.Dropout(0.1)

# Loss functions: descriptive names
self.reconstruction_loss = nn.MSELoss()
self.kl_divergence_loss = nn.KLDivLoss()
self.adversarial_loss = nn.BCELoss()
self.perceptual_loss = LPIPS()

# Optimizers: descriptive names
self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
```

---

## 🏗️ Architecture Patterns

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
processed_video = pipeline(
    raw_video,
    preprocess_video,
    normalize_video,
    augment_video
)
```

### Dependency Injection
```python
from dataclasses import dataclass
from typing import Protocol

class VideoModelProtocol(Protocol):
    def forward(self, x: Tensor) -> Tensor:
        ...

@dataclass
class VideoProcessingConfig:
    model: VideoModelProtocol
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    device: torch.device
    batch_size: int = 32
    learning_rate: float = 0.001
```

### Strategy Pattern
```python
from abc import ABC, abstractmethod

class VideoProcessingStrategy(ABC):
    @abstractmethod
    def process(self, video: Tensor) -> Tensor:
        pass

class DiffusionStrategy(VideoProcessingStrategy):
    def process(self, video: Tensor) -> Tensor:
        return self.diffusion_pipeline(video)

class TransformerStrategy(VideoProcessingStrategy):
    def process(self, video: Tensor) -> Tensor:
        return self.transformer_model(video)
```

---

## 📁 File Organization

### Recommended Structure
```
video-OpusClip/
├── __init__.py
├── README_OPTIMIZED.md
├── requirements_optimized.txt
├── production_runner.py
├── run_optimized.py
├── benchmark_suite.py
├── 
├── 📁 core/
│   ├── __init__.py
│   ├── models.py
│   ├── enhanced_models.py
│   └── video_ai_refactored.py
├── 
├── 📁 api/
│   ├── __init__.py
│   ├── fastapi_microservice.py
│   ├── services.py
│   ├── utils_api.py
│   ├── utils_batch.py
│   └── aws_lambda_handler.py
├── 
├── 📁 optimization/
│   ├── __init__.py
│   ├── ultra_performance_optimizers.py
│   ├── optimized_video_ai.py
│   └── optimized_video_ai_ultra.py
├── 
├── 📁 production/
│   ├── __init__.py
│   ├── production_api_ultra.py
│   ├── production_config.py
│   ├── production_example.py
│   └── install_ultra_optimizations.py
├── 
├── 📁 benchmarking/
│   ├── __init__.py
│   ├── benchmark_optimization.py
│   ├── advanced_benchmark_system.py
│   ├── test_microservice.py
│   └── test_system.py
├── 
├── 📁 utils/
│   ├── __init__.py
│   ├── functional_utils.py
│   ├── lazy_loading_system.py
│   └── visualization_utils.py
├── 
├── 📁 config/
│   ├── __init__.py
│   ├── settings.py
│   └── environment.py
├── 
├── 📁 tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_optimization.py
└── 
├── 📁 docs/
│   ├── API_GUIDE.md
│   ├── PERFORMANCE_GUIDE.md
│   └── DEPLOYMENT_GUIDE.md
```

### Module Organization
```python
# __init__.py files should export public APIs
# core/__init__.py
from .models import VideoProcessor, CaptionGenerator
from .enhanced_models import EnhancedVideoProcessor

__all__ = ["VideoProcessor", "CaptionGenerator", "EnhancedVideoProcessor"]

# api/__init__.py
from .fastapi_microservice import create_app
from .services import VideoService, CaptionService

__all__ = ["create_app", "VideoService", "CaptionService"]
```

---

## 📦 Import Organization

### Import Order
```python
# 1. Standard library imports (alphabetical)
import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 2. Third-party imports (alphabetical)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

# 3. Local imports (alphabetical)
from .core.models import VideoProcessor
from .utils.functional_utils import timing_decorator
from .config.settings import get_settings
```

### Import Best Practices
```python
# ✅ Good: Specific imports
from torch import Tensor, nn, optim
from typing import Dict, List, Optional

# ❌ Bad: Wildcard imports
from torch import *
from typing import *

# ✅ Good: Aliasing for clarity
import torch.nn.functional as F
import numpy as np

# ✅ Good: Relative imports for local modules
from .models import VideoProcessor
from ..utils import helpers
```

---

## 🛡️ Error Handling

### Exception Hierarchy
```python
class VideoOpusClipError(Exception):
    """Base exception for Video-OpusClip system."""
    pass

class VideoProcessingError(VideoOpusClipError):
    """Raised when video processing fails."""
    pass

class ModelLoadingError(VideoOpusClipError):
    """Raised when model loading fails."""
    pass

class ConfigurationError(VideoOpusClipError):
    """Raised when configuration is invalid."""
    pass
```

### Error Handling Patterns
```python
def process_video_safely(video_path: Path) -> Optional[Tensor]:
    """Process video with comprehensive error handling."""
    try:
        # Validate input
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Load video
        video = load_video(video_path)
        
        # Process video
        processed_video = process_video(video)
        
        return processed_video
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except VideoProcessingError as e:
        logger.error(f"Video processing failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
    finally:
        # Cleanup resources
        cleanup_resources()
```

### Context Managers
```python
from contextlib import contextmanager

@contextmanager
def video_processing_context():
    """Context manager for video processing operations."""
    try:
        # Setup
        setup_gpu_memory()
        yield
    except Exception as e:
        # Error handling
        logger.error(f"Video processing error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_gpu_memory()
```

---

## 📊 Logging & Monitoring

### Structured Logging
```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlog.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Usage
logger.info("Video processing started", 
           video_path=str(video_path),
           batch_size=batch_size,
           device=str(device))
```

### Performance Monitoring
```python
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info("Function execution time",
                   function=func.__name__,
                   execution_time=end_time - start_time)
        
        return result
    return wrapper

@timing_decorator
def process_video_batch(videos: List[Tensor]) -> List[Tensor]:
    """Process a batch of videos with timing."""
    pass
```

---

## ⚡ Performance Optimization

### Memory Management
```python
import torch
import gc

def optimize_memory_usage():
    """Optimize memory usage for video processing."""
    # Clear cache
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Monitor memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info("GPU memory usage",
                   allocated_gb=allocated,
                   reserved_gb=reserved)
```

### Batch Processing
```python
def process_videos_in_batches(
    videos: List[Tensor],
    batch_size: int = 32,
    device: torch.device = None
) -> List[Tensor]:
    """Process videos in optimized batches."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    for i in range(0, len(videos), batch_size):
        batch = videos[i:i + batch_size]
        batch_tensor = torch.stack(batch).to(device)
        
        with torch.no_grad():
            processed_batch = model(batch_tensor)
        
        results.extend(processed_batch.cpu().split(1))
        
        # Memory cleanup
        del batch_tensor, processed_batch
        torch.cuda.empty_cache()
    
    return results
```

### Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_video_async(video_path: Path) -> Tensor:
    """Process video asynchronously."""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            process_video_sync, 
            video_path
        )
    
    return result

async def process_multiple_videos(video_paths: List[Path]) -> List[Tensor]:
    """Process multiple videos concurrently."""
    tasks = [process_video_async(path) for path in video_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = [r for r in results if not isinstance(r, Exception)]
    return valid_results
```

---

## 🧪 Testing Standards

### Unit Testing
```python
import pytest
import torch
from unittest.mock import Mock, patch

class TestVideoProcessor:
    """Test cases for VideoProcessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = VideoProcessor()
        self.sample_video = torch.randn(1, 16, 3, 224, 224)
    
    def test_video_processing(self):
        """Test video processing functionality."""
        result = self.processor(self.sample_video)
        
        assert result.shape == (1, 16, 3, 224, 224)
        assert torch.isfinite(result).all()
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        with pytest.raises(ValueError):
            self.processor(torch.randn(1, 3, 224, 224))  # Missing frame dimension
    
    @patch('torch.cuda.is_available')
    def test_gpu_processing(self, mock_cuda):
        """Test GPU processing when available."""
        mock_cuda.return_value = True
        
        processor = VideoProcessor(device="cuda")
        result = processor(self.sample_video)
        
        assert result.device.type == "cuda"
```

### Integration Testing
```python
class TestVideoProcessingPipeline:
    """Integration tests for video processing pipeline."""
    
    def test_end_to_end_processing(self):
        """Test complete video processing pipeline."""
        # Setup
        video_path = Path("test_data/sample_video.mp4")
        config = VideoProcessingConfig(
            model=MockVideoModel(),
            batch_size=4,
            device="cpu"
        )
        
        # Execute
        pipeline = VideoProcessingPipeline(config)
        result = pipeline.process(video_path)
        
        # Assert
        assert result is not None
        assert len(result.captions) > 0
        assert result.processing_time > 0
```

---

## 📚 Documentation Standards

### Module Documentation
```python
"""
Video Processing Module

This module provides high-performance video processing capabilities using
PyTorch and advanced AI models. It includes video encoding, caption generation,
and optimization features.

Classes:
    VideoProcessor: Main video processing class
    CaptionGenerator: AI-powered caption generation
    PerformanceMonitor: Real-time performance monitoring

Functions:
    process_video_batch: Process multiple videos efficiently
    generate_captions: Generate captions for video content
    optimize_memory: Optimize memory usage for large videos

Example:
    >>> processor = VideoProcessor()
    >>> result = processor.process(video_tensor)
    >>> captions = generate_captions(result)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
```

### Function Documentation
```python
def process_video_with_captions(
    video: torch.Tensor,
    model: torch.nn.Module,
    generate_captions: bool = True,
    max_caption_length: int = 100
) -> Dict[str, Any]:
    """
    Process video and optionally generate captions.
    
    This function takes a video tensor and processes it through the specified
    model. If caption generation is enabled, it will also generate descriptive
    captions for the video content.
    
    Args:
        video: Input video tensor of shape (batch, frames, channels, height, width)
        model: PyTorch model for video processing
        generate_captions: Whether to generate captions for the video
        max_caption_length: Maximum length of generated captions
        
    Returns:
        Dictionary containing:
            - processed_video: Processed video tensor
            - captions: List of generated captions (if enabled)
            - processing_time: Time taken for processing
            - memory_usage: Memory usage during processing
            
    Raises:
        ValueError: If video tensor has invalid shape or dimensions
        RuntimeError: If model is not properly initialized or CUDA is not available
        
    Example:
        >>> video = torch.randn(1, 16, 3, 224, 224)
        >>> model = VideoProcessingModel()
        >>> result = process_video_with_captions(video, model)
        >>> print(f"Generated {len(result['captions'])} captions")
        
    Note:
        This function automatically handles GPU memory management and
        will fall back to CPU if CUDA is not available.
    """
    pass
```

---

## 🧠 PyTorch & AI Conventions

### Model Architecture
```python
class VideoProcessingModel(nn.Module):
    """Advanced video processing model with attention mechanisms."""
    
    def __init__(self, 
                 input_channels: int = 3,
                 hidden_dim: int = 512,
                 num_frames: int = 16,
                 num_heads: int = 8):
        super().__init__()
        
        # Store configuration
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        
        # Define layers with descriptive names
        self.frame_encoder = nn.Conv3d(
            input_channels, hidden_dim, 
            kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )
        
        self.temporal_attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout=0.1
        )
        
        self.spatial_attention = SpatialAttention(hidden_dim)
        
        self.output_projection = nn.Conv3d(
            hidden_dim, input_channels,
            kernel_size=1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the video processing model.
        
        Args:
            x: Input video tensor of shape (batch, frames, channels, height, width)
            
        Returns:
            Processed video tensor of same shape as input
        """
        # Validate input
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor, got {x.dim()}D")
        
        batch_size, frames, channels, height, width = x.shape
        
        # Frame encoding
        encoded = self.frame_encoder(x)
        
        # Temporal attention
        temporal_features = self.temporal_attention(encoded)
        
        # Spatial attention
        spatial_features = self.spatial_attention(temporal_features)
        
        # Output projection
        output = self.output_projection(spatial_features)
        
        return output
```

### Training Loop
```python
class VideoTrainer:
    """Comprehensive video model trainer with monitoring."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler,
                 device: torch.device,
                 config: TrainingConfig):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Setup logging and monitoring
        self.logger = structlog.get_logger(__name__)
        self.metrics = TrainingMetrics()
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch with comprehensive monitoring."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (videos, targets) in enumerate(progress_bar):
            # Move to device
            videos = videos.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log batch metrics
            if batch_idx % self.config.log_frequency == 0:
                self.logger.info("Training batch",
                               batch=batch_idx,
                               loss=loss.item(),
                               learning_rate=self.optimizer.param_groups[0]["lr"])
        
        # Update scheduler
        self.scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        return {"train_loss": avg_loss}
```

---

## 🌐 API Design

### FastAPI Endpoints
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="Video-OpusClip API", version="1.0.0")

class VideoProcessingRequest(BaseModel):
    """Request model for video processing."""
    video_url: str = Field(..., description="URL of the video to process")
    processing_type: str = Field(default="caption", description="Type of processing")
    max_length: int = Field(default=100, ge=1, le=500, description="Maximum output length")
    
    class Config:
        schema_extra = {
            "example": {
                "video_url": "https://example.com/video.mp4",
                "processing_type": "caption",
                "max_length": 100
            }
        }

class VideoProcessingResponse(BaseModel):
    """Response model for video processing."""
    processed_video: str = Field(..., description="Base64 encoded processed video")
    captions: List[str] = Field(..., description="Generated captions")
    processing_time: float = Field(..., description="Processing time in seconds")
    confidence_score: float = Field(..., description="Confidence score of processing")

@app.post("/api/video/process", response_model=VideoProcessingResponse)
async def process_video(
    request: VideoProcessingRequest,
    video_service: VideoService = Depends(get_video_service)
) -> VideoProcessingResponse:
    """
    Process video and generate captions.
    
    This endpoint processes the provided video and generates captions
    based on the specified processing type.
    """
    try:
        result = await video_service.process_video(
            video_url=request.video_url,
            processing_type=request.processing_type,
            max_length=request.max_length
        )
        
        return VideoProcessingResponse(**result)
        
    except VideoProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error in video processing", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Error Responses
```python
from fastapi import HTTPException
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

@app.exception_handler(VideoProcessingError)
async def video_processing_exception_handler(request, exc):
    """Handle video processing errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=str(exc),
            code="VIDEO_PROCESSING_ERROR",
            details={"video_url": getattr(exc, 'video_url', None)}
        ).dict()
    )
```

---

## ⚙️ Configuration Management

### Environment Configuration
```python
from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Model Settings
    model_path: str = Field(..., env="MODEL_PATH")
    device: str = Field(default="auto", env="DEVICE")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    
    # Performance Settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    enable_gpu: bool = Field(default=True, env="ENABLE_GPU")
    mixed_precision: bool = Field(default=True, env="MIXED_PRECISION")
    
    # Cache Settings
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

### Configuration Validation
```python
def validate_configuration(settings: Settings) -> None:
    """Validate application configuration."""
    errors = []
    
    # Validate model path
    if not Path(settings.model_path).exists():
        errors.append(f"Model path does not exist: {settings.model_path}")
    
    # Validate device
    if settings.device == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA device requested but not available")
    
    # Validate batch size
    if settings.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if errors:
        raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
```

---

## 🔒 Security Conventions

### Input Validation
```python
from pathlib import Path
import re

def validate_video_url(url: str) -> bool:
    """Validate video URL format and security."""
    # Check URL format
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        return False
    
    # Check for potentially dangerous URLs
    dangerous_patterns = [
        r'file://',
        r'ftp://',
        r'javascript:',
        r'data:'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return False
    
    return True

def validate_video_file(file_path: Path) -> bool:
    """Validate video file security."""
    # Check file extension
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    if file_path.suffix.lower() not in allowed_extensions:
        return False
    
    # Check file size (max 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    if file_path.stat().st_size > max_size:
        return False
    
    return True
```

### Rate Limiting
```python
from fastapi import HTTPException, Request
import time
from collections import defaultdict

class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=60)

def check_rate_limit(request: Request):
    """Dependency to check rate limiting."""
    client_id = request.client.host
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
```

---

## 📋 Summary

### Key Principles
1. **Consistency**: Follow established patterns throughout the codebase
2. **Readability**: Write self-documenting code with clear naming
3. **Maintainability**: Use modular design and proper separation of concerns
4. **Performance**: Optimize for speed and memory efficiency
5. **Reliability**: Implement comprehensive error handling and testing
6. **Security**: Validate inputs and implement proper access controls

### Code Quality Tools
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing
- **pre-commit**: Git hooks

### Best Practices
- Always use type hints
- Write comprehensive docstrings
- Implement proper error handling
- Use async/await for I/O operations
- Monitor performance and memory usage
- Write unit and integration tests
- Follow security best practices
- Use configuration management
- Implement proper logging

This comprehensive set of conventions ensures that the Video-OpusClip system maintains high code quality, performance, and maintainability across all components. 