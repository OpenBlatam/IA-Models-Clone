"""
Base Predictor
Abstract base class for predictors
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """
    Abstract base class for predictors
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize predictor
        
        Args:
            model: PyTorch model
            device: Device to use
            use_mixed_precision: Use mixed precision
        """
        self.model = model.to(device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.model.eval()
    
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Make prediction
        
        Args:
            inputs: Input data
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        pass
    
    def predict_batch(self, inputs: List[Any], batch_size: int = 32, **kwargs) -> List[Any]:
        """
        Batch prediction
        
        Args:
            inputs: List of inputs
            batch_size: Batch size
            **kwargs: Additional arguments
            
        Returns:
            List of predictions
        """
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_results = self._process_batch(batch, **kwargs)
            results.extend(batch_results)
        return results
    
    @abstractmethod
    def _process_batch(self, batch: List[Any], **kwargs) -> List[Any]:
        """Process batch - must be implemented by subclasses"""
        pass


class TensorPredictor(BasePredictor):
    """
    Predictor for tensor inputs
    """
    
    @torch.inference_mode()
    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predict from tensor
        
        Args:
            inputs: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Prediction tensor
        """
        if inputs.device != self.device:
            inputs = inputs.to(self.device, non_blocking=True)
        
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)
        
        return outputs
    
    def _process_batch(self, batch: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        """Process batch of tensors"""
        batch_tensor = torch.stack(batch).to(self.device, non_blocking=True)
        outputs = self.predict(batch_tensor, **kwargs)
        return outputs.cpu().split(1)


class FeaturePredictor(BasePredictor):
    """
    Predictor for feature vectors
    """
    
    def predict(self, features: List[float], **kwargs) -> float:
        """
        Predict from features
        
        Args:
            features: Feature vector
            **kwargs: Additional arguments
            
        Returns:
            Prediction value
        """
        tensor = torch.tensor([features], dtype=torch.float32)
        output = self.model(tensor.to(self.device))
        return output.item() if output.numel() == 1 else output.squeeze().item()
    
    def _process_batch(self, batch: List[List[float]], **kwargs) -> List[float]:
        """Process batch of feature vectors"""
        tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
        outputs = self.model(tensor)
        return outputs.cpu().numpy().flatten().tolist()













