"""
Base Interfaces for All Components

Defines clear contracts (Protocols) for all major components following
SOLID principles and Python best practices.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional, List, Union
import torch
from torch.nn import Module
from torch.utils.data import DataLoader


# ============================================================================
# Model Interfaces
# ============================================================================

class IModel(Protocol):
    """Protocol for all model implementations."""
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        ...
    
    def train(self, mode: bool = True) -> 'IModel':
        """Set training mode."""
        ...
    
    def eval(self) -> 'IModel':
        """Set evaluation mode."""
        ...
    
    def parameters(self):
        """Get model parameters."""
        ...
    
    def to(self, device: Union[str, torch.device]) -> 'IModel':
        """Move model to device."""
        ...


class IEmbeddingModel(IModel, Protocol):
    """Protocol for embedding models."""
    
    def get_embeddings(
        self,
        inputs: torch.Tensor,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """Extract embeddings from inputs."""
        ...


class IClassifier(IModel, Protocol):
    """Protocol for classification models."""
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        ...
    
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        ...


# ============================================================================
# Training Interfaces
# ============================================================================

class ITrainingStrategy(ABC):
    """Abstract base for training strategies."""
    
    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, float]:
        """Execute one training step."""
        pass
    
    @abstractmethod
    def validate_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """Execute one validation step."""
        pass
    
    @abstractmethod
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch."""
        pass


class ILossFunction(Protocol):
    """Protocol for loss functions."""
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss."""
        ...


class IOptimizer(Protocol):
    """Protocol for optimizers."""
    
    def step(self):
        """Perform optimization step."""
        ...
    
    def zero_grad(self):
        """Zero gradients."""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        ...
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state."""
        ...


class IScheduler(Protocol):
    """Protocol for learning rate schedulers."""
    
    def step(self, metrics: Optional[float] = None):
        """Update learning rate."""
        ...
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        ...


# ============================================================================
# Data Interfaces
# ============================================================================

class IDataTransform(Protocol):
    """Protocol for data transformations."""
    
    def __call__(self, data: Any) -> Any:
        """Apply transformation."""
        ...


class IDataAugmentation(IDataTransform, Protocol):
    """Protocol for data augmentations."""
    
    def apply(self, data: Any, **kwargs) -> Any:
        """Apply augmentation."""
        ...


class IDataLoader(Protocol):
    """Protocol for data loaders."""
    
    def __iter__(self):
        """Iterate over batches."""
        ...
    
    def __len__(self) -> int:
        """Get number of batches."""
        ...


# ============================================================================
# Callback Interfaces
# ============================================================================

class ITrainingCallback(ABC):
    """Abstract base for training callbacks."""
    
    @abstractmethod
    def on_epoch_start(self, epoch: int):
        """Called at the start of an epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Called at the end of an epoch."""
        pass
    
    @abstractmethod
    def on_batch_start(self, batch_idx: int):
        """Called at the start of a batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]):
        """Called at the end of a batch."""
        pass


# ============================================================================
# Inference Interfaces
# ============================================================================

class IInferencePipeline(ABC):
    """Abstract base for inference pipelines."""
    
    @abstractmethod
    def predict(
        self,
        inputs: Any,
        **kwargs
    ) -> Any:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_batch(
        self,
        inputs: List[Any],
        **kwargs
    ) -> List[Any]:
        """Make batch predictions."""
        pass


# ============================================================================
# Monitoring Interfaces
# ============================================================================

class IMonitor(Protocol):
    """Protocol for monitoring components."""
    
    def start(self):
        """Start monitoring."""
        ...
    
    def stop(self):
        """Stop monitoring."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        ...


class IProfiler(Protocol):
    """Protocol for profiling components."""
    
    def profile(
        self,
        func: callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Profile a function."""
        ...


# ============================================================================
# Checkpoint Interfaces
# ============================================================================

class ICheckpointManager(Protocol):
    """Protocol for checkpoint management."""
    
    def save(
        self,
        checkpoint: Dict[str, Any],
        path: str
    ):
        """Save checkpoint."""
        ...
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load checkpoint."""
        ...
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        ...


# ============================================================================
# Experiment Tracking Interfaces
# ============================================================================

class IExperimentTracker(Protocol):
    """Protocol for experiment tracking."""
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics."""
        ...
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        ...
    
    def log_artifact(self, artifact_path: str, name: str):
        """Log artifact."""
        ...
    
    def finish(self):
        """Finish tracking."""
        ...


# ============================================================================
# Factory Interfaces
# ============================================================================

class IFactory(ABC):
    """Abstract base for factories."""
    
    @abstractmethod
    def create(self, config: Dict[str, Any]) -> Any:
        """Create instance from config."""
        pass


# ============================================================================
# Device Management Interfaces
# ============================================================================

class IDeviceManager(Protocol):
    """Protocol for device management."""
    
    def get_device(self) -> torch.device:
        """Get current device."""
        ...
    
    def move_to_device(
        self,
        obj: Union[torch.Tensor, Module]
    ) -> Union[torch.Tensor, Module]:
        """Move object to device."""
        ...
    
    def enable_mixed_precision(self) -> bool:
        """Check if mixed precision is enabled."""
        ...


__all__ = [
    # Model interfaces
    "IModel",
    "IEmbeddingModel",
    "IClassifier",
    # Training interfaces
    "ITrainingStrategy",
    "ILossFunction",
    "IOptimizer",
    "IScheduler",
    # Data interfaces
    "IDataTransform",
    "IDataAugmentation",
    "IDataLoader",
    # Callback interfaces
    "ITrainingCallback",
    # Inference interfaces
    "IInferencePipeline",
    # Monitoring interfaces
    "IMonitor",
    "IProfiler",
    # Checkpoint interfaces
    "ICheckpointManager",
    # Experiment tracking interfaces
    "IExperimentTracker",
    # Factory interfaces
    "IFactory",
    # Device management interfaces
    "IDeviceManager",
]



