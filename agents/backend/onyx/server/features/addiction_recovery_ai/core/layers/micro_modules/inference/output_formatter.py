"""
Output Formatter - Ultra-Specific Output Formatting
Separated into its own file for maximum modularity
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OutputFormatterBase(ABC):
    """Base class for output formatters"""
    
    def __init__(self, name: str = "OutputFormatter"):
        self.name = name
    
    @abstractmethod
    def format(self, outputs: torch.Tensor, **kwargs) -> Any:
        """Format outputs"""
        pass


class TensorFormatter(OutputFormatterBase):
    """Format outputs as tensors"""
    
    def __init__(self):
        super().__init__("TensorFormatter")
    
    def format(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Format outputs as tensor"""
        return outputs


class NumpyFormatter(OutputFormatterBase):
    """Format outputs as numpy arrays"""
    
    def __init__(self):
        super().__init__("NumpyFormatter")
    
    def format(self, outputs: torch.Tensor, **kwargs) -> np.ndarray:
        """Format outputs as numpy array"""
        if isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu().numpy()
        return np.array(outputs)


class ListFormatter(OutputFormatterBase):
    """Format outputs as lists"""
    
    def __init__(self):
        super().__init__("ListFormatter")
    
    def format(self, outputs: torch.Tensor, **kwargs) -> List[Any]:
        """Format outputs as list"""
        if isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu().tolist()
        return list(outputs)


class DictFormatter(OutputFormatterBase):
    """Format outputs as dictionary"""
    
    def __init__(self, key_prefix: str = "output"):
        super().__init__("DictFormatter")
        self.key_prefix = key_prefix
    
    def format(self, outputs: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Format outputs as dictionary"""
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu()
        
        if outputs.dim() == 1:
            return {f"{self.key_prefix}_{i}": float(outputs[i]) for i in range(len(outputs))}
        elif outputs.dim() == 2:
            return {
                f"{self.key_prefix}_{i}": outputs[i].tolist()
                for i in range(outputs.size(0))
            }
        else:
            return {self.key_prefix: outputs.tolist()}


class ProbabilityFormatter(OutputFormatterBase):
    """Format outputs as probabilities"""
    
    def __init__(self):
        super().__init__("ProbabilityFormatter")
    
    def format(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Format outputs as probabilities (apply softmax)"""
        if isinstance(outputs, torch.Tensor):
            return torch.softmax(outputs, dim=-1)
        return outputs


class LogitsFormatter(OutputFormatterBase):
    """Format outputs as logits (no transformation)"""
    
    def __init__(self):
        super().__init__("LogitsFormatter")
    
    def format(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Format outputs as logits"""
        return outputs


class ArgMaxFormatter(OutputFormatterBase):
    """Format outputs as argmax indices"""
    
    def __init__(self, dim: int = -1):
        super().__init__("ArgMaxFormatter")
        self.dim = dim
    
    def format(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Format outputs as argmax indices"""
        if isinstance(outputs, torch.Tensor):
            return torch.argmax(outputs, dim=self.dim)
        return outputs


class TopKFormatter(OutputFormatterBase):
    """Format outputs as top-k predictions"""
    
    def __init__(self, k: int = 5):
        super().__init__("TopKFormatter")
        self.k = k
    
    def format(self, outputs: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Format outputs as top-k predictions"""
        if isinstance(outputs, torch.Tensor):
            values, indices = torch.topk(outputs, k=self.k, dim=-1)
            return {
                'values': values.detach().cpu().tolist(),
                'indices': indices.detach().cpu().tolist()
            }
        return {'values': [], 'indices': []}


# Factory for output formatters
class OutputFormatterFactory:
    """Factory for creating output formatters"""
    
    _registry = {
        'tensor': TensorFormatter,
        'numpy': NumpyFormatter,
        'list': ListFormatter,
        'dict': DictFormatter,
        'probability': ProbabilityFormatter,
        'logits': LogitsFormatter,
        'argmax': ArgMaxFormatter,
        'topk': TopKFormatter,
    }
    
    @classmethod
    def create(cls, formatter_type: str, **kwargs) -> OutputFormatterBase:
        """Create output formatter"""
        formatter_type = formatter_type.lower()
        if formatter_type not in cls._registry:
            raise ValueError(f"Unknown output formatter type: {formatter_type}")
        return cls._registry[formatter_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, formatter_class: type):
        """Register custom output formatter"""
        cls._registry[name.lower()] = formatter_class


__all__ = [
    "OutputFormatterBase",
    "TensorFormatter",
    "NumpyFormatter",
    "ListFormatter",
    "DictFormatter",
    "ProbabilityFormatter",
    "LogitsFormatter",
    "ArgMaxFormatter",
    "TopKFormatter",
    "OutputFormatterFactory",
]



