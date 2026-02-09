"""
Base Model Interface and Model Manager
Following PyTorch best practices for model management
"""

import logging
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all ML models
    Follows PyTorch nn.Module pattern for consistency
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize base model
        
        Args:
            model_name: Name identifier for the model
            device: PyTorch device (cuda/cpu). Auto-detected if None
            use_mixed_precision: Whether to use mixed precision training/inference
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.use_mixed_precision = use_mixed_precision
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Mixed precision scaler for training
        if self.use_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        logger.info(f"Initialized {model_name} on device: {self.device}")
    
    @staticmethod
    def _get_device() -> torch.device:
        """Auto-detect and return appropriate device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    @abstractmethod
    async def load(self) -> None:
        """Load model and tokenizer"""
        pass
    
    @abstractmethod
    async def predict(self, inputs: Any) -> Dict[str, Any]:
        """Run inference on inputs"""
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device"""
        return tensor.to(self.device)
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()
    
    def train_mode(self) -> None:
        """Set model to training mode"""
        if self.model is not None:
            self.model.train()
    
    def save_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict() if hasattr(self.model, "state_dict") else None,
            "model_name": self.model_name,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        if hasattr(self.model, "load_state_dict") and checkpoint.get("model_state_dict"):
            self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")


class ModelManager:
    """
    Manages multiple model instances with caching and lifecycle management
    """
    
    def __init__(self, max_cache_size: int = 10):
        """
        Initialize model manager
        
        Args:
            max_cache_size: Maximum number of models to cache in memory
        """
        self.models: Dict[str, BaseModel] = {}
        self.max_cache_size = max_cache_size
        self.load_times: Dict[str, datetime] = {}
        logger.info(f"Initialized ModelManager with max_cache_size={max_cache_size}")
    
    async def get_model(
        self,
        model_class: type[BaseModel],
        model_name: str,
        **kwargs
    ) -> BaseModel:
        """
        Get or create model instance with caching
        
        Args:
            model_class: Model class to instantiate
            model_name: Unique identifier for the model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model instance
        """
        if model_name in self.models:
            logger.debug(f"Using cached model: {model_name}")
            return self.models[model_name]
        
        # Check cache size limit
        if len(self.models) >= self.max_cache_size:
            await self._evict_oldest_model()
        
        # Create and load new model
        logger.info(f"Loading new model: {model_name}")
        model = model_class(model_name=model_name, **kwargs)
        await model.load()
        
        self.models[model_name] = model
        self.load_times[model_name] = datetime.now()
        
        return model
    
    async def _evict_oldest_model(self) -> None:
        """Evict the oldest model from cache"""
        if not self.models:
            return
        
        oldest_model = min(self.load_times.items(), key=lambda x: x[1])[0]
        logger.info(f"Evicting oldest model from cache: {oldest_model}")
        
        # Clean up model resources
        model = self.models.pop(oldest_model)
        del model.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        del self.load_times[oldest_model]
    
    def clear_cache(self) -> None:
        """Clear all cached models"""
        logger.info("Clearing model cache")
        for model in self.models.values():
            del model.model
        self.models.clear()
        self.load_times.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_models": list(self.models.keys()),
            "cache_size": len(self.models),
            "max_cache_size": self.max_cache_size,
            "load_times": {k: v.isoformat() for k, v in self.load_times.items()},
        }



