"""
Inference Interfaces - Define contracts for inference
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch


class IInferenceEngine(ABC):
    """
    Interface for inference engines
    """
    
    @abstractmethod
    def infer(self, input_data: Any) -> Any:
        """Run inference"""
        pass
    
    @abstractmethod
    def batch_infer(self, batch: List[Any]) -> List[Any]:
        """Run batch inference"""
        pass
    
    @abstractmethod
    def get_device(self) -> str:
        """Get device"""
        pass


class IBatchProcessor(ABC):
    """
    Interface for batch processors
    """
    
    @abstractmethod
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a batch"""
        pass
    
    @abstractmethod
    def get_batch_size(self) -> int:
        """Get batch size"""
        pass


class IModelOptimizer(ABC):
    """
    Interface for model optimizers
    """
    
    @abstractmethod
    def optimize(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model"""
        pass
    
    @abstractmethod
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization information"""
        pass













