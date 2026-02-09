"""
Core Interfaces
Abstract base classes and interfaces for the ML framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn


class IModelLoader(ABC):
    """Interface for model loading."""
    
    @abstractmethod
    def load(self, model_name: str, **kwargs) -> nn.Module:
        """Load a model."""
        pass
    
    @abstractmethod
    def unload(self, model_name: str) -> bool:
        """Unload a model."""
        pass


class IInferenceEngine(ABC):
    """Interface for inference operations."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str], **kwargs) -> torch.Tensor:
        """Get embeddings for texts."""
        pass


class ITrainer(ABC):
    """Interface for training operations."""
    
    @abstractmethod
    def train(self, num_epochs: int, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        pass


class IEvaluator(ABC):
    """Interface for evaluation operations."""
    
    @abstractmethod
    def evaluate(self, data_loader, **kwargs) -> Dict[str, float]:
        """Evaluate the model."""
        pass


class IDataProcessor(ABC):
    """Interface for data processing."""
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process data."""
        pass


class IOptimizer(ABC):
    """Interface for optimization operations."""
    
    @abstractmethod
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Optimize the model."""
        pass


class IQuantizer(ABC):
    """Interface for quantization operations."""
    
    @abstractmethod
    def quantize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantize the model."""
        pass


class IProfiler(ABC):
    """Interface for profiling operations."""
    
    @abstractmethod
    def profile(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Profile the model."""
        pass


class IRegistry(ABC):
    """Interface for model registry."""
    
    @abstractmethod
    def register(self, model_name: str, model_path: str, **kwargs) -> Any:
        """Register a model."""
        pass
    
    @abstractmethod
    def get(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Get a model."""
        pass


class IConfigManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def load(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any], config_path: str) -> bool:
        """Save configuration."""
        pass


class ILogger(ABC):
    """Interface for logging."""
    
    @abstractmethod
    def log(self, message: str, level: str = "INFO", **kwargs) -> None:
        """Log a message."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics."""
        pass



