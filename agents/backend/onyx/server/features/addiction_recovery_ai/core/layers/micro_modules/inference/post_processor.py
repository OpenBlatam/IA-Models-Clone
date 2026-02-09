"""
Post Processor - Ultra-Specific Post Processing
Separated into its own file for maximum modularity
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PostProcessorBase(ABC):
    """Base class for post processors"""
    
    def __init__(self, name: str = "PostProcessor"):
        self.name = name
    
    @abstractmethod
    def process(self, outputs: torch.Tensor, **kwargs) -> Any:
        """Process outputs"""
        pass


class SoftmaxPostProcessor(PostProcessorBase):
    """Apply softmax to outputs"""
    
    def __init__(self, dim: int = -1):
        super().__init__("SoftmaxPostProcessor")
        self.dim = dim
    
    def process(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply softmax"""
        return F.softmax(outputs, dim=self.dim)


class SigmoidPostProcessor(PostProcessorBase):
    """Apply sigmoid to outputs"""
    
    def __init__(self):
        super().__init__("SigmoidPostProcessor")
    
    def process(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply sigmoid"""
        return torch.sigmoid(outputs)


class NormalizePostProcessor(PostProcessorBase):
    """Normalize outputs"""
    
    def __init__(self, p: float = 2.0, dim: int = -1):
        super().__init__("NormalizePostProcessor")
        self.p = p
        self.dim = dim
    
    def process(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Normalize outputs"""
        return F.normalize(outputs, p=self.p, dim=self.dim)


class ClampPostProcessor(PostProcessorBase):
    """Clamp outputs to range"""
    
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__("ClampPostProcessor")
        self.min_val = min_val
        self.max_val = max_val
    
    def process(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Clamp outputs"""
        return torch.clamp(outputs, min=self.min_val, max=self.max_val)


class ThresholdPostProcessor(PostProcessorBase):
    """Apply threshold to outputs"""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__("ThresholdPostProcessor")
        self.threshold = threshold
    
    def process(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply threshold"""
        return (outputs > self.threshold).float()


class ArgMaxPostProcessor(PostProcessorBase):
    """Apply argmax to outputs"""
    
    def __init__(self, dim: int = -1):
        super().__init__("ArgMaxPostProcessor")
        self.dim = dim
    
    def process(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply argmax"""
        return torch.argmax(outputs, dim=self.dim)


class ComposePostProcessor(PostProcessorBase):
    """Compose multiple post processors"""
    
    def __init__(self, processors: List[PostProcessorBase]):
        super().__init__("ComposePostProcessor")
        self.processors = processors
    
    def process(self, outputs: torch.Tensor, **kwargs) -> Any:
        """Apply all processors in sequence"""
        result = outputs
        for processor in self.processors:
            result = processor.process(result, **kwargs)
        return result


# Factory for post processors
class PostProcessorFactory:
    """Factory for creating post processors"""
    
    _registry = {
        'softmax': SoftmaxPostProcessor,
        'sigmoid': SigmoidPostProcessor,
        'normalize': NormalizePostProcessor,
        'clamp': ClampPostProcessor,
        'threshold': ThresholdPostProcessor,
        'argmax': ArgMaxPostProcessor,
    }
    
    @classmethod
    def create(cls, processor_type: str, **kwargs) -> PostProcessorBase:
        """Create post processor"""
        processor_type = processor_type.lower()
        if processor_type not in cls._registry:
            raise ValueError(f"Unknown post processor type: {processor_type}")
        return cls._registry[processor_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, processor_class: type):
        """Register custom post processor"""
        cls._registry[name.lower()] = processor_class
    
    @classmethod
    def create_compose(cls, processor_types: List[str], **kwargs) -> ComposePostProcessor:
        """Create composed post processor"""
        processors = [cls.create(pt, **kwargs) for pt in processor_types]
        return ComposePostProcessor(processors)


__all__ = [
    "PostProcessorBase",
    "SoftmaxPostProcessor",
    "SigmoidPostProcessor",
    "NormalizePostProcessor",
    "ClampPostProcessor",
    "ThresholdPostProcessor",
    "ArgMaxPostProcessor",
    "ComposePostProcessor",
    "PostProcessorFactory",
]



