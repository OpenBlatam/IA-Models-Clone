"""
Base Classes and Common Abstractions
Shared base classes for consistency across the codebase
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseComponent(ABC):
    """
    Base class for all components
    Provides common functionality
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self._initialized = False
        self._config: Dict[str, Any] = {}
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize component with configuration"""
        if config:
            self._config.update(config)
        self._initialized = True
        logger.debug(f"{self.name} initialized")
    
    def get_config(self) -> Dict[str, Any]:
        """Get component configuration"""
        return self._config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        self._config.update(updates)
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate component state"""
        pass


class BaseProcessor(BaseComponent):
    """
    Base class for data processors
    """
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data"""
        pass
    
    def process_batch(self, input_batch: List[Any]) -> List[Any]:
        """Process batch of inputs"""
        return [self.process(item) for item in input_batch]
    
    def validate(self) -> bool:
        """Validate processor"""
        return self._initialized


class BaseEvaluator(BaseComponent):
    """
    Base class for model evaluators
    """
    
    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """Evaluate model"""
        pass
    
    def validate(self) -> bool:
        """Validate evaluator"""
        return self._initialized


class ConfigurableMixin:
    """
    Mixin for configurable components
    """
    
    def __init__(self, *args, **kwargs):
        self._config = kwargs.pop('config', {})
        super().__init__(*args, **kwargs)
    
    def configure(self, **kwargs):
        """Configure component"""
        self._config.update(kwargs)
        self._apply_config()
    
    def _apply_config(self):
        """Apply configuration to component"""
        # Override in subclasses
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration"""
        return self._config.copy()


class SaveableMixin:
    """
    Mixin for components that can be saved/loaded
    """
    
    def save(self, path: Union[str, Path]):
        """Save component state"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = self.get_state()
        torch.save(state, path)
        logger.info(f"{self.__class__.__name__} saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load component state"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")
        
        state = torch.load(path, map_location='cpu')
        self.load_state(state)
        logger.info(f"{self.__class__.__name__} loaded from {path}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state for saving"""
        # Override in subclasses
        return {}
    
    def load_state(self, state: Dict[str, Any]):
        """Load component state"""
        # Override in subclasses
        pass


class DeviceAwareMixin:
    """
    Mixin for device-aware components
    """
    
    def __init__(self, *args, device: str = None, **kwargs):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        super().__init__(*args, **kwargs)
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to device"""
        return tensor.to(self.device, non_blocking=True)
    
    def get_device(self) -> torch.device:
        """Get device"""
        return self.device













