"""
Type Hints and Type Definitions
Centralized type definitions for better type checking
"""

from typing import (
    Dict, List, Optional, Tuple, Union, Any, Callable,
    Protocol, TypedDict
)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# Tensor types
Tensor = torch.Tensor
TensorDict = Dict[str, torch.Tensor]
TensorList = List[torch.Tensor]

# Image types
ImageType = Union[np.ndarray, Image.Image, torch.Tensor, str, bytes]
ImageList = List[ImageType]

# Label types
LabelDict = Dict[str, List]
LabelType = Union[torch.Tensor, np.ndarray, List, int, float]

# Batch types
Batch = Dict[str, Any]
BatchDict = Dict[str, torch.Tensor]

# Model types
ModelType = torch.nn.Module
ModelOutput = Union[torch.Tensor, Dict[str, torch.Tensor]]

# Config types
ConfigDict = Dict[str, Any]
ModelConfig = Dict[str, Any]
TrainingConfig = Dict[str, Any]
DataConfig = Dict[str, Any]

# Metrics types
MetricsDict = Dict[str, float]
MetricsList = List[MetricsDict]

# Loss types
LossDict = Dict[str, torch.Tensor]
LossValue = Union[torch.Tensor, float]

# Optimizer types
OptimizerType = torch.optim.Optimizer
SchedulerType = torch.optim.lr_scheduler._LRScheduler

# Device types
DeviceType = Union[str, torch.device]

# Transform types
TransformType = Optional[Callable]

# Dataset types
DatasetType = Dataset
DataLoaderType = DataLoader

# Callback types
CallbackType = Callable[[int, Dict[str, float], Any], None]

# Path types
PathType = Union[str, bytes, Any]  # Any for Path objects


class ModelProtocol(Protocol):
    """Protocol for model-like objects"""
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass"""
        ...
    
    def parameters(self):
        """Get model parameters"""
        ...
    
    def to(self, device: DeviceType):
        """Move to device"""
        ...
    
    def eval(self):
        """Set to evaluation mode"""
        ...
    
    def train(self, mode: bool = True):
        """Set to training mode"""
        ...


class TrainerProtocol(Protocol):
    """Protocol for trainer-like objects"""
    def fit(
        self,
        optimizer: OptimizerType,
        num_epochs: int,
        scheduler: Optional[SchedulerType] = None,
        criterion: Optional[torch.nn.Module] = None
    ):
        """Train the model"""
        ...
    
    def validate(self) -> MetricsDict:
        """Validate the model"""
        ...


class DatasetProtocol(Protocol):
    """Protocol for dataset-like objects"""
    def __len__(self) -> int:
        """Get dataset length"""
        ...
    
    def __getitem__(self, idx: int) -> Batch:
        """Get item by index"""
        ...


# TypedDict for common structures
class TrainingMetrics(TypedDict, total=False):
    """Training metrics structure"""
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    learning_rate: float
    epoch: int


class ModelInfo(TypedDict, total=False):
    """Model information structure"""
    name: str
    num_parameters: int
    trainable_parameters: int
    model_size_mb: float
    device: str
    dtype: str


class ExperimentConfig(TypedDict, total=False):
    """Experiment configuration structure"""
    experiment_id: str
    name: str
    description: str
    model_type: str
    hyperparameters: ConfigDict
    dataset_info: Dict[str, Any]
    created_at: str
    status: str













