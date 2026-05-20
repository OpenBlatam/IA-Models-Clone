"""
Advanced Device Context Manager

Provides context managers for device operations, mixed precision,
and proper resource management following PyTorch best practices.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Union, Any, Dict
import torch
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class DeviceContext:
    """
    Advanced device context manager with mixed precision support.
    
    Follows PyTorch best practices for:
    - Device management
    - Mixed precision training/inference
    - Memory optimization
    - Error handling
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = True,
        enable_benchmark: bool = True,
        enable_tf32: bool = True
    ):
        """
        Initialize device context.
        
        Args:
            device: Device to use (auto-detect if None)
            use_mixed_precision: Enable mixed precision (FP16)
            enable_benchmark: Enable cuDNN benchmark mode
            enable_tf32: Enable TF32 for faster training on Ampere+
        """
        self.device = self._get_device(device)
        self.use_mixed_precision = (
            use_mixed_precision and 
            self.device.type == "cuda" and 
            torch.cuda.is_available()
        )
        
        # Setup device optimizations
        self._setup_device(enable_benchmark, enable_tf32)
        
        # Mixed precision scaler
        self.scaler: Optional[GradScaler] = None
        if self.use_mixed_precision:
            self.scaler = GradScaler(
                init_scale=2.**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
            logger.info("Mixed precision enabled with GradScaler")
    
    def _get_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Get the appropriate device."""
        if device is not None:
            if isinstance(device, str):
                return torch.device(device)
            return device
        
        # Auto-detect best device
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _setup_device(self, enable_benchmark: bool, enable_tf32: bool):
        """Setup device optimizations."""
        if self.device.type == "cuda":
            # Enable cuDNN benchmark for consistent input sizes
            if enable_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark mode enabled")
            
            # Enable TF32 for faster training on Ampere+ GPUs
            if enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled for faster training")
            
            # Log GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(self.device.index or 0)
                gpu_memory = (
                    torch.cuda.get_device_properties(self.device.index or 0).total_memory / 1e9
                )
                logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            logger.info(f"Using device: {self.device}")
    
    @contextmanager
    def inference_context(self):
        """
        Context manager for inference with proper settings.
        
        Usage:
            with device_context.inference_context():
                output = model(input)
        """
        if self.model is None:
            raise RuntimeError("Model not set. Use set_model() first.")
        
        # Set model to eval mode
        was_training = self.model.training
        self.model.eval()
        
        try:
            with torch.no_grad():
                if self.use_mixed_precision:
                    with autocast():
                        yield
                else:
                    yield
        finally:
            # Restore training mode
            if was_training:
                self.model.train()
    
    @contextmanager
    def training_context(self):
        """
        Context manager for training with mixed precision.
        
        Usage:
            with device_context.training_context():
                loss = criterion(output, target)
                loss.backward()
        """
        if self.model is None:
            raise RuntimeError("Model not set. Use set_model() first.")
        
        # Set model to train mode
        self.model.train()
        
        if self.use_mixed_precision:
            with autocast():
                yield
        else:
            yield
    
    def set_model(self, model: torch.nn.Module):
        """Set the model and move it to device."""
        self.model = model.to(self.device, non_blocking=True)
        logger.info(f"Model moved to {self.device}")
    
    def move_to_device(
        self, 
        obj: Union[torch.Tensor, torch.nn.Module, Dict[str, Any]]
    ) -> Union[torch.Tensor, torch.nn.Module, Dict[str, Any]]:
        """
        Move object(s) to device efficiently.
        
        Args:
            obj: Tensor, Module, or dict of tensors/modules
            
        Returns:
            Object(s) on the correct device
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)
        elif isinstance(obj, torch.nn.Module):
            return obj.to(self.device, non_blocking=True)
        elif isinstance(obj, dict):
            return {
                k: self.move_to_device(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.move_to_device(item) for item in obj)
        return obj
    
    def compile_model(
        self, 
        model: torch.nn.Module, 
        mode: str = "reduce-overhead"
    ) -> torch.nn.Module:
        """
        Compile model for faster execution (PyTorch 2.0+).
        
        Args:
            model: Model to compile
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
            
        Returns:
            Compiled model
        """
        if hasattr(torch, 'compile'):
            try:
                compiled = torch.compile(model, mode=mode)
                logger.info(f"Model compiled with mode={mode}")
                return compiled
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                return model
        else:
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model
    
    def clear_cache(self):
        """Clear CUDA cache if using GPU."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for the device."""
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
            }
        return {}
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device
    
    def get_scaler(self) -> Optional[GradScaler]:
        """Get the mixed precision scaler."""
        return self.scaler


class TrainingContext:
    """
    Context manager for training operations with automatic
    gradient scaling and error handling.
    """
    
    def __init__(
        self,
        device_context: DeviceContext,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: Optional[float] = None
    ):
        """
        Initialize training context.
        
        Args:
            device_context: Device context manager
            optimizer: Optimizer
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.device_context = device_context
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.scaler = device_context.get_scaler()
    
    def backward(self, loss: torch.Tensor):
        """
        Backward pass with automatic scaling.
        
        Args:
            loss: Loss tensor
        """
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self):
        """Optimizer step with automatic unscaling."""
        if self.scaler:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.device_context.model.parameters(),
                    max_norm=self.max_grad_norm
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.device_context.model.parameters(),
                    max_norm=self.max_grad_norm
                )
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


__all__ = [
    "DeviceContext",
    "TrainingContext",
]



