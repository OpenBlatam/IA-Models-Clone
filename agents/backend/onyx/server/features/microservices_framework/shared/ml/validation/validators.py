"""
Advanced Validators
Comprehensive validation utilities.
"""

from typing import Any, Optional, Dict, List, Callable
import torch
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ModelInputValidator:
    """Validate model inputs."""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        shape: Optional[tuple] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> bool:
        """Validate tensor properties."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        if shape and tensor.shape != shape:
            raise ValueError(f"Expected shape {shape}, got {tensor.shape}")
        
        if dtype and tensor.dtype != dtype:
            raise ValueError(f"Expected dtype {dtype}, got {tensor.dtype}")
        
        if device and tensor.device != device:
            raise ValueError(f"Expected device {device}, got {tensor.device}")
        
        # Check for NaN/Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Tensor contains NaN or Inf values")
        
        return True
    
    @staticmethod
    def validate_batch(
        batch: List[torch.Tensor],
        min_size: int = 1,
        max_size: Optional[int] = None,
    ) -> bool:
        """Validate batch of tensors."""
        if not batch:
            raise ValueError("Batch cannot be empty")
        
        if len(batch) < min_size:
            raise ValueError(f"Batch size {len(batch)} < min_size {min_size}")
        
        if max_size and len(batch) > max_size:
            raise ValueError(f"Batch size {len(batch)} > max_size {max_size}")
        
        # Check all tensors have same shape
        first_shape = batch[0].shape
        for i, tensor in enumerate(batch[1:], 1):
            if tensor.shape != first_shape:
                raise ValueError(
                    f"Tensor {i} has shape {tensor.shape}, expected {first_shape}"
                )
        
        return True


class ImageValidator:
    """Validate image inputs."""
    
    @staticmethod
    def validate_image(
        image: Image.Image,
        min_size: Optional[tuple] = None,
        max_size: Optional[tuple] = None,
        mode: Optional[str] = None,
    ) -> bool:
        """Validate PIL Image."""
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(image)}")
        
        width, height = image.size
        
        if min_size:
            min_w, min_h = min_size
            if width < min_w or height < min_h:
                raise ValueError(
                    f"Image size ({width}, {height}) < min_size {min_size}"
                )
        
        if max_size:
            max_w, max_h = max_size
            if width > max_w or height > max_h:
                raise ValueError(
                    f"Image size ({width}, {height}) > max_size {max_size}"
                )
        
        if mode and image.mode != mode:
            raise ValueError(f"Expected mode {mode}, got {image.mode}")
        
        return True
    
    @staticmethod
    def validate_image_array(
        array: np.ndarray,
        shape: Optional[tuple] = None,
        dtype: Optional[np.dtype] = None,
        value_range: Optional[tuple] = None,
    ) -> bool:
        """Validate numpy image array."""
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(array)}")
        
        if shape and array.shape != shape:
            raise ValueError(f"Expected shape {shape}, got {array.shape}")
        
        if dtype and array.dtype != dtype:
            raise ValueError(f"Expected dtype {dtype}, got {array.dtype}")
        
        if value_range:
            min_val, max_val = value_range
            if array.min() < min_val or array.max() > max_val:
                raise ValueError(
                    f"Values outside range {value_range}: "
                    f"[{array.min()}, {array.max()}]"
                )
        
        return True


class ConfigValidator:
    """Validate configuration objects."""
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> bool:
        """Validate training configuration."""
        required_fields = ["batch_size", "learning_rate", "num_epochs"]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        if config["batch_size"] <= 0:
            raise ValueError("batch_size must be > 0")
        
        if config["learning_rate"] <= 0:
            raise ValueError("learning_rate must be > 0")
        
        if config["num_epochs"] <= 0:
            raise ValueError("num_epochs must be > 0")
        
        return True
    
    @staticmethod
    def validate_generation_config(config: Dict[str, Any]) -> bool:
        """Validate generation configuration."""
        if "max_length" in config and config["max_length"] <= 0:
            raise ValueError("max_length must be > 0")
        
        if "temperature" in config:
            temp = config["temperature"]
            if temp < 0 or temp > 2:
                raise ValueError("temperature must be in [0, 2]")
        
        if "top_p" in config:
            top_p = config["top_p"]
            if top_p < 0 or top_p > 1:
                raise ValueError("top_p must be in [0, 1]")
        
        return True


class ValidatorChain:
    """Chain multiple validators."""
    
    def __init__(self):
        self.validators: List[Callable[[Any], bool]] = []
    
    def add(self, validator: Callable[[Any], bool]) -> "ValidatorChain":
        """Add validator to chain."""
        self.validators.append(validator)
        return self
    
    def validate(self, value: Any) -> bool:
        """Run all validators."""
        for validator in self.validators:
            if not validator(value):
                return False
        return True



