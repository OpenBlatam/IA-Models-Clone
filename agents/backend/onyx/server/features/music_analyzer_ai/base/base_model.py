"""
Base Model - Shared base implementation for all models
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

from ..interfaces.model_interface import IMusicModel

logger = logging.getLogger(__name__)


class BaseMusicModel(IMusicModel):
    """
    Base class for all music models with common functionality
    """
    
    def __init__(self, name: str = "BaseMusicModel"):
        super().__init__()
        self.name = name
        self.device = "cpu"
        self._is_compiled = False
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        return super().to(device)
    
    def compile(self, mode: str = "reduce-overhead"):
        """Compile model for faster execution"""
        if hasattr(torch, 'compile'):
            try:
                compiled = torch.compile(self, mode=mode)
                self._is_compiled = True
                logger.info(f"Compiled {self.name} with mode={mode}")
                return compiled
            except Exception as e:
                logger.warning(f"Compilation failed: {str(e)}")
                return self
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": self.name,
            "device": self.device,
            "is_compiled": self._is_compiled,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_type": type(self).__name__
        }
    
    def predict(self, features: Any) -> Dict[str, Any]:
        """Default prediction implementation"""
        if isinstance(features, torch.Tensor):
            features = features.to(self.device)
        else:
            features = torch.tensor(features).to(self.device)
        
        self.eval()
        with torch.no_grad():
            output = self.forward(features)
        
        return {"output": output.cpu().numpy() if isinstance(output, torch.Tensor) else output}













