"""
Base Service for AI Services

Provides common functionality for all AI services including:
- Device management (CPU/GPU)
- Mixed precision support
- Model loading and caching
- Error handling
- Logging
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import torch
from torch.cuda.amp import autocast, GradScaler
from contextlib import contextmanager

from ...config import settings

logger = logging.getLogger(__name__)


class BaseAIService(ABC):
    """
    Base class for all AI services
    
    Provides common functionality for:
    - Device management
    - Mixed precision training/inference
    - Model loading and caching
    - Error handling
    """
    
    def __init__(self, model_name: str, model_type: str = "transformer"):
        """
        Initialize base AI service
        
        Args:
            model_name: Name/path of the model
            model_type: Type of model (transformer, diffusion, etc.)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.use_mixed_precision = settings.use_mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_mixed_precision else None
        self._model_loaded = False
        
        logger.info(
            f"Initializing {self.__class__.__name__} with model: {model_name} "
            f"on device: {self.device}"
        )
    
    def _get_device(self) -> torch.device:
        """
        Get the appropriate device (CPU or GPU)
        
        Returns:
            torch.device object
        """
        if not settings.ai_enabled:
            return torch.device("cpu")
        
        if settings.use_gpu and torch.cuda.is_available():
            device_id = int(os.getenv("CUDA_DEVICE", "0"))
            device = torch.device(f"cuda:{device_id}")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
            return device
        else:
            logger.info("Using CPU (GPU not available or disabled)")
            return torch.device("cpu")
    
    @contextmanager
    def inference_context(self):
        """
        Context manager for inference with proper device and mixed precision
        
        Usage:
            with self.inference_context():
                output = self.model(input)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        # Use no_grad for inference
        with torch.no_grad():
            if self.use_mixed_precision:
                with autocast():
                    yield
            else:
                yield
    
    @contextmanager
    def training_context(self):
        """
        Context manager for training with proper device and mixed precision
        
        Usage:
            with self.training_context():
                loss = criterion(output, target)
                loss.backward()
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Set model to train mode
        if hasattr(self.model, 'train'):
            self.model.train()
        
        if self.use_mixed_precision:
            with autocast():
                yield
        else:
            yield
    
    def to_device(self, tensor: Any) -> Any:
        """
        Move tensor to the appropriate device
        
        Args:
            tensor: Tensor or model to move
            
        Returns:
            Tensor/model on the correct device
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        elif hasattr(tensor, 'to'):
            return tensor.to(self.device)
        return tensor
    
    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the model (to be implemented by subclasses)
        
        Args:
            force_reload: Force reload even if already loaded
        """
        if self._model_loaded and not force_reload:
            return
        
        if not settings.ai_enabled:
            logger.warning(f"AI features disabled, skipping model load: {self.model_name}")
            return
        
        try:
            self._load_model_impl()
            self._model_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}", exc_info=True)
            raise
    
    @abstractmethod
    def _load_model_impl(self) -> None:
        """
        Implementation of model loading (to be implemented by subclasses)
        """
        pass
    
    def unload_model(self) -> None:
        """
        Unload the model to free memory
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self._model_loaded = False
        logger.info(f"Model {self.model_name} unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": str(self.device),
            "loaded": self._model_loaded,
            "mixed_precision": self.use_mixed_precision
        }
        
        if self.model is not None:
            if hasattr(self.model, 'config'):
                info["config"] = str(self.model.config)
            if hasattr(self.model, '__class__'):
                info["model_class"] = self.model.__class__.__name__
        
        if self.device.type == "cuda" and torch.cuda.is_available():
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated(self.device) / 1e9
            info["gpu_memory_reserved"] = torch.cuda.memory_reserved(self.device) / 1e9
        
        return info
    
    def check_nan_inf(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """
        Check for NaN or Inf values in tensor
        
        Args:
            tensor: Tensor to check
            name: Name for logging
            
        Returns:
            True if NaN/Inf found
        """
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan or has_inf:
            logger.warning(f"{name} contains NaN: {has_nan}, Inf: {has_inf}")
            return True
        
        return False
    
    def clip_gradients(self, model: torch.nn.Module, max_norm: float = 1.0) -> None:
        """
        Clip gradients to prevent exploding gradients
        
        Args:
            model: Model to clip gradients for
            max_norm: Maximum gradient norm
        """
        if hasattr(model, 'parameters'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    def __enter__(self):
        """Context manager entry"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Optionally unload model on exit
        # self.unload_model()
        pass















