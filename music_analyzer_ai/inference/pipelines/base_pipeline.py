"""
Base Inference Pipeline
Abstract base class for inference pipelines
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class BaseInferencePipeline(ABC):
    """
    Abstract base class for inference pipelines
    """
    
    def __init__(
        self,
        model,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.device = device
        
        # Move model to device and set to eval
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Run inference on input data
        
        Args:
            input_data: Input data
            **kwargs: Additional inference parameters
        
        Returns:
            Prediction results
        """
        pass
    
    def preprocess(self, input_data: Any) -> Any:
        """Preprocess input data"""
        if self.preprocess_fn:
            return self.preprocess_fn(input_data)
        return input_data
    
    def postprocess(self, output: Any) -> Any:
        """Postprocess output data"""
        if self.postprocess_fn:
            return self.postprocess_fn(output)
        return output



