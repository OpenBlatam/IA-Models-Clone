"""
Base Model Interface
====================

Abstract base class for all ML models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Base model configuration."""
    input_dim: int
    output_dim: int
    device: str = "auto"
    dtype: str = "float32"


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    
    All models should inherit from this class and implement:
    - forward(): Forward pass
    - predict(): Inference method
    - get_config(): Return model configuration
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.device = self._get_device(config.device)
        self.dtype = self._get_dtype(config.dtype)
        self._logger = logger
    
    def _get_device(self, device: str) -> torch.device:
        """Get device from string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _get_dtype(self, dtype: str) -> torch.dtype:
        """Get dtype from string."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(dtype, torch.float32)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Make prediction.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
        
        Returns:
            Prediction dictionary
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            "input_dim": self.config.input_dim,
            "output_dim": self.config.output_dim,
            "device": str(self.device),
            "dtype": str(self.dtype)
        }
    
    def save(self, path: str):
        """
        Save model.
        
        Args:
            path: Path to save model
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.get_config()
        }, path)
        self._logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self._logger.info(f"Model loaded from {path}")




