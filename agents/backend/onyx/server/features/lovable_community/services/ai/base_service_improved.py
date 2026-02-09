"""
Improved Base Service for AI Services

Following best practices for deep learning:
- Proper GPU utilization
- Mixed precision training/inference
- Model compilation with torch.compile()
- Efficient memory management
- Proper error handling and logging
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Union
import torch
from torch.cuda.amp import autocast, GradScaler
from contextlib import contextmanager
from functools import lru_cache

from ...config import settings
from ...utils import load_model_optimized, compile_model

logger = logging.getLogger(__name__)


class BaseAIServiceImproved(ABC):
    """
    Improved base class for all AI services.
    
    Best practices implemented:
    - Optimal device management
    - Mixed precision with automatic scaling
    - Model compilation for faster inference
    - Efficient memory management
    - Proper gradient handling
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "transformer",
        use_compile: bool = True,
        use_quantization: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize improved base AI service.
        
        Args:
            model_name: Name/path of the model
            model_type: Type of model (transformer, diffusion, etc.)
            use_compile: Whether to compile model with torch.compile() (PyTorch 2.0+)
            use_quantization: Quantization type ("4bit", "8bit", or None)
            device: Specific device to use (auto-detect if None)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = device if device else self._get_optimal_device()
        self.use_compile = use_compile and hasattr(torch, 'compile')
        self.use_quantization = use_quantization
        self._compiled = False
        
        # Mixed precision configuration
        self.use_mixed_precision = (
            settings.use_mixed_precision and 
            self.device.type == "cuda" and
            torch.cuda.is_available()
        )
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Performance optimizations
        self._configure_performance()
        
        self._model_loaded = False
        
        logger.info(
            f"Initializing {self.__class__.__name__} with model: {model_name} "
            f"on device: {self.device}, mixed_precision: {self.use_mixed_precision}, "
            f"compile: {self.use_compile}"
        )
    
    def _get_optimal_device(self) -> torch.device:
        """
        Get optimal device with best practices.
        
        Returns:
            torch.device object
        """
        if not settings.ai_enabled:
            return torch.device("cpu")
        
        if settings.use_gpu and torch.cuda.is_available():
            device_id = int(os.getenv("CUDA_DEVICE", "0"))
            device = torch.device(f"cuda:{device_id}")
            
            # Log GPU information
            props = torch.cuda.get_device_properties(device_id)
            logger.info(f"Using GPU: {props.name}")
            logger.info(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            
            # Set memory fraction if needed
            if hasattr(settings, 'gpu_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(
                    settings.gpu_memory_fraction,
                    device_id
                )
            
            return device
        else:
            logger.info("Using CPU (GPU not available or disabled)")
            return torch.device("cpu")
    
    def _configure_performance(self) -> None:
        """Configure PyTorch for optimal performance."""
        try:
            if self.device.type == "cuda":
                # Set matmul precision for better performance
                torch.set_float32_matmul_precision('high')
                
                # Enable cuDNN benchmarking for consistent input sizes
                if torch.backends.cudnn.is_available():
                    torch.backends.cudnn.benchmark = True
                    logger.info("cuDNN benchmarking enabled")
                
                # Enable TensorFloat-32 for Ampere GPUs (A100, RTX 30xx+)
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("TensorFloat-32 enabled")
        except Exception as e:
            logger.warning(f"Failed to configure performance optimizations: {e}")
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the model (to be implemented by subclasses).
        
        Should use load_model_optimized or load_model_quantized from utils.
        """
        pass
    
    def load_model(self) -> None:
        """
        Load model with optimizations.
        
        Handles:
        - Optimized loading
        - Quantization if requested
        - Model compilation
        - Device placement
        """
        if self._model_loaded:
            logger.debug("Model already loaded")
            return
        
        try:
            # Load model (subclass implements _load_model)
            self._load_model()
            
            if self.model is None:
                raise RuntimeError("Model not loaded by _load_model()")
            
            # Compile model if requested and available
            if self.use_compile and not self._compiled:
                try:
                    self.model = compile_model(self.model)
                    self._compiled = True
                    logger.info("Model compiled with torch.compile()")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    @contextmanager
    def inference_context(self):
        """
        Context manager for optimized inference.
        
        Features:
        - Automatic eval mode
        - No gradient computation
        - Mixed precision if enabled
        - Memory efficient
        
        Usage:
            with self.inference_context():
                output = self.model(input)
        """
        if not self._model_loaded:
            self.load_model()
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        # Use no_grad for inference (saves memory)
        with torch.no_grad():
            if self.use_mixed_precision:
                with autocast():
                    yield
            else:
                yield
    
    @contextmanager
    def training_context(self):
        """
        Context manager for training with mixed precision.
        
        Features:
        - Automatic train mode
        - Gradient computation enabled
        - Mixed precision with scaling
        - Gradient clipping support
        
        Usage:
            with self.training_context():
                loss = criterion(output, target)
                loss.backward()
        """
        if not self._model_loaded:
            self.load_model()
        
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
    
    def backward_with_scaling(self, loss: torch.Tensor) -> None:
        """
        Backward pass with automatic mixed precision scaling.
        
        Args:
            loss: Loss tensor
        """
        if self.use_mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Optimizer step with automatic mixed precision scaling.
        
        Args:
            optimizer: PyTorch optimizer
        """
        if self.use_mixed_precision and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def clip_gradients(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ) -> float:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 for L2)
            
        Returns:
            Gradient norm before clipping
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if self.use_mixed_precision and self.scaler:
            # Unscale gradients before clipping
            self.scaler.unscale_(self._get_optimizer())
        
        # Get all parameters with gradients
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        
        if not parameters:
            return 0.0
        
        # Compute and clip gradients
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad, norm_type) for p in parameters]),
            norm_type
        )
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in parameters:
                p.grad.mul_(clip_coef)
        
        return total_norm.item()
    
    def _get_optimizer(self) -> Optional[torch.optim.Optimizer]:
        """Get optimizer (to be implemented by subclasses if needed)."""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self._model_loaded or self.model is None:
            return {"status": "not_loaded"}
        
        info = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": str(self.device),
            "mixed_precision": self.use_mixed_precision,
            "compiled": self._compiled,
            "quantized": self.use_quantization is not None,
        }
        
        # Add parameter count
        if hasattr(self.model, 'parameters'):
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            info["total_parameters"] = total_params
            info["trainable_parameters"] = trainable_params
        
        # Add GPU memory info
        if self.device.type == "cuda":
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated(self.device) / 1e9
            info["gpu_memory_reserved"] = torch.cuda.memory_reserved(self.device) / 1e9
        
        return info
    
    def clear_cache(self) -> None:
        """Clear GPU cache if using CUDA."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.clear_cache()













