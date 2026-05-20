"""
Inference Service - High-level Inference Management
====================================================

Service for managing inference workflows:
- Batch inference
- Streaming inference
- Model serving
- Performance optimization
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.base import BaseComponent
from ..inference import InferenceEngine
from ..utils import get_device, optimize_model_for_inference

logger = logging.getLogger(__name__)


class InferenceService(BaseComponent):
    """
    High-level service for inference management.
    
    Provides unified interface for inference operations.
    """
    
    def _initialize(self) -> None:
        """Initialize service."""
        self.device = get_device()
        self.engine: Optional[InferenceEngine] = None
        self.model: Optional[nn.Module] = None
    
    def load_model(
        self,
        model: nn.Module,
        optimize: bool = True
    ) -> 'InferenceService':
        """
        Load model for inference.
        
        Args:
            model: PyTorch model
            optimize: Optimize for inference
            
        Returns:
            Self for method chaining
        """
        self.model = model.to(self.device)
        
        if optimize:
            self.model = optimize_model_for_inference(self.model)
        
        self.engine = InferenceEngine(self.model, device=self.device)
        
        return self
    
    def predict(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[Any]],
        return_probabilities: bool = False,
        top_k: Optional[int] = None
    ) -> Any:
        """
        Run inference.
        
        Args:
            inputs: Input data
            return_probabilities: Return probabilities
            top_k: Return top-k predictions
            
        Returns:
            Predictions
        """
        if self.engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.engine.predict(inputs, return_probabilities, top_k)
    
    def predict_batch(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> List[Any]:
        """
        Run batch inference.
        
        Args:
            dataloader: DataLoader with batches
            max_batches: Maximum number of batches
            
        Returns:
            List of predictions
        """
        if self.engine is None:
            raise RuntimeError("Model not loaded.")
        
        return self.engine.batch_predict(dataloader, max_batches)



