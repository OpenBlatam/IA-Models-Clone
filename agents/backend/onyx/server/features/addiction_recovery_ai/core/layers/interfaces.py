"""
Protocol Interfaces - Define contracts for all layers
Enables dependency injection and testability
"""

from typing import Protocol, Optional, Any, Dict, List, Iterator
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Layer Protocols
# ============================================================================

class IDataProcessor(Protocol):
    """Protocol for data processors"""
    
    def process(self, data: Any) -> Any:
        """Process raw data"""
        ...
    
    def validate(self, data: Any) -> bool:
        """Validate data"""
        ...


class IDataLoader(Protocol):
    """Protocol for data loaders"""
    
    def load(self, source: str) -> Dataset:
        """Load dataset from source"""
        ...
    
    def split(self, dataset: Dataset, ratios: Dict[str, float]) -> Dict[str, Dataset]:
        """Split dataset into train/val/test"""
        ...


# ============================================================================
# Model Layer Protocols
# ============================================================================

class IModel(Protocol):
    """Protocol for models"""
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass"""
        ...
    
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Get model parameters"""
        ...
    
    def train(self) -> 'IModel':
        """Set model to training mode"""
        ...
    
    def eval(self) -> 'IModel':
        """Set model to evaluation mode"""
        ...


class IModelBuilder(Protocol):
    """Protocol for model builders"""
    
    def build(self, config: Dict[str, Any]) -> IModel:
        """Build model from config"""
        ...


# ============================================================================
# Training Layer Protocols
# ============================================================================

class ITrainer(Protocol):
    """Protocol for trainers"""
    
    def train(self, train_loader: Any, val_loader: Optional[Any], **kwargs) -> Dict[str, Any]:
        """Train model"""
        ...
    
    def validate(self, val_loader: Any) -> Dict[str, float]:
        """Validate model"""
        ...


class IOptimizer(Protocol):
    """Protocol for optimizers"""
    
    def step(self) -> None:
        """Perform optimization step"""
        ...
    
    def zero_grad(self) -> None:
        """Zero gradients"""
        ...


class IScheduler(Protocol):
    """Protocol for learning rate schedulers"""
    
    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate"""
        ...


# ============================================================================
# Inference Layer Protocols
# ============================================================================

class IPredictor(Protocol):
    """Protocol for predictors"""
    
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Make predictions"""
        ...
    
    def predict_batch(self, inputs: List[Any], **kwargs) -> List[Any]:
        """Make batch predictions"""
        ...


class IInferenceEngine(Protocol):
    """Protocol for inference engines"""
    
    def process(self, inputs: Any, **kwargs) -> Any:
        """Process inference request"""
        ...
    
    def optimize(self) -> None:
        """Optimize for inference"""
        ...


# ============================================================================
# Service Layer Protocols
# ============================================================================

class IService(Protocol):
    """Protocol for services"""
    
    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute service request"""
        ...
    
    def validate(self, request: Dict[str, Any]) -> bool:
        """Validate service request"""
        ...


# ============================================================================
# Interface Layer Protocols
# ============================================================================

class IAPIHandler(Protocol):
    """Protocol for API handlers"""
    
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API request"""
        ...
    
    def format_response(self, result: Any) -> Dict[str, Any]:
        """Format API response"""
        ...


# ============================================================================
# Abstract Base Classes (for concrete implementations)
# ============================================================================

class BaseDataProcessor(ABC):
    """Base class for data processors"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process raw data"""
        pass
    
    def validate(self, data: Any) -> bool:
        """Validate data - default implementation"""
        return data is not None


class BaseModel(ABC):
    """Base class for models"""
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass"""
        pass
    
    def train_mode(self) -> 'BaseModel':
        """Set to training mode"""
        return self
    
    def eval_mode(self) -> 'BaseModel':
        """Set to evaluation mode"""
        return self


class BasePredictor(ABC):
    """Base class for predictors"""
    
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Make predictions"""
        pass
    
    def predict_batch(self, inputs: List[Any], **kwargs) -> List[Any]:
        """Default batch prediction implementation"""
        return [self.predict(inp, **kwargs) for inp in inputs]


class BaseService(ABC):
    """Base class for services"""
    
    @abstractmethod
    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute service request"""
        pass
    
    def validate(self, request: Dict[str, Any]) -> bool:
        """Validate request - default implementation"""
        return isinstance(request, dict) and len(request) > 0



