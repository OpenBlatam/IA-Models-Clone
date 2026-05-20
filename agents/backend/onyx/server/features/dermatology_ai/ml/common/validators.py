"""
Validation Utilities
Centralized validation for inputs, models, and configurations
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """Validate inputs for models and processing"""
    
    @staticmethod
    def validate_image(
        image: Any,
        expected_shape: Optional[Tuple[int, ...]] = None,
        min_size: Optional[Tuple[int, int]] = None,
        max_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Validate image input
        
        Args:
            image: Image to validate
            expected_shape: Expected shape (C, H, W) or (H, W, C)
            min_size: Minimum size (height, width)
            max_size: Maximum size (height, width)
            
        Returns:
            True if valid
        """
        try:
            # Convert to numpy if needed
            if isinstance(image, torch.Tensor):
                img_array = image.cpu().numpy()
            elif isinstance(image, Image.Image):
                img_array = np.array(image)
            elif isinstance(image, np.ndarray):
                img_array = image
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return False
            
            # Check shape
            if len(img_array.shape) < 2:
                logger.error(f"Invalid image shape: {img_array.shape}")
                return False
            
            # Check size
            if min_size:
                h, w = img_array.shape[:2]
                if h < min_size[0] or w < min_size[1]:
                    logger.error(f"Image too small: {img_array.shape[:2]} < {min_size}")
                    return False
            
            if max_size:
                h, w = img_array.shape[:2]
                if h > max_size[0] or w > max_size[1]:
                    logger.error(f"Image too large: {img_array.shape[:2]} > {max_size}")
                    return False
            
            # Check expected shape
            if expected_shape:
                if img_array.shape != expected_shape:
                    logger.warning(f"Shape mismatch: {img_array.shape} != {expected_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> bool:
        """Validate tensor"""
        try:
            if not isinstance(tensor, torch.Tensor):
                logger.error(f"Not a tensor: {type(tensor)}")
                return False
            
            if expected_shape and tensor.shape != expected_shape:
                logger.warning(f"Shape mismatch: {tensor.shape} != {expected_shape}")
            
            if dtype and tensor.dtype != dtype:
                logger.warning(f"Dtype mismatch: {tensor.dtype} != {dtype}")
            
            if device and tensor.device != device:
                logger.warning(f"Device mismatch: {tensor.device} != {device}")
            
            # Check for NaN/Inf
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.error("Tensor contains NaN or Inf")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Tensor validation failed: {e}")
            return False
    
    @staticmethod
    def validate_batch(
        batch: Dict[str, Any],
        required_keys: List[str],
        image_key: str = "image"
    ) -> bool:
        """Validate data batch"""
        try:
            # Check required keys
            for key in required_keys:
                if key not in batch:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Validate image
            if image_key in batch:
                if not InputValidator.validate_image(batch[image_key]):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            return False


class ModelValidator:
    """Validate model architecture and state"""
    
    @staticmethod
    def validate_model(
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Validate model
        
        Args:
            model: PyTorch model
            input_shape: Input shape for testing
            device: Device to test on
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'num_params': 0,
            'trainable_params': 0,
            'model_size_mb': 0.0
        }
        
        try:
            # Check if model has forward method
            if not hasattr(model, 'forward'):
                results['valid'] = False
                results['errors'].append("Model missing forward method")
                return results
            
            # Count parameters
            results['num_params'] = sum(p.numel() for p in model.parameters())
            results['trainable_params'] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            
            # Calculate model size
            param_size = sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
            buffer_size = sum(
                b.numel() * b.element_size() for b in model.buffers()
            )
            results['model_size_mb'] = (param_size + buffer_size) / (1024 ** 2)
            
            # Test forward pass
            model.eval()
            dummy_input = torch.randn(input_shape).to(device)
            
            with torch.no_grad():
                try:
                    output = model(dummy_input)
                    results['output_shape'] = tuple(output.shape) if hasattr(output, 'shape') else None
                except Exception as e:
                    results['valid'] = False
                    results['errors'].append(f"Forward pass failed: {e}")
            
            # Check for NaN/Inf in parameters
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    results['warnings'].append(f"NaN in parameter: {name}")
                if torch.isinf(param).any():
                    results['warnings'].append(f"Inf in parameter: {name}")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Validation failed: {e}")
        
        return results
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        required_keys = ['model_name', 'model_type']
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                return False
        
        return True


class ConfigValidator:
    """Validate configuration dictionaries"""
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> bool:
        """Validate training configuration"""
        required_sections = ['model', 'training', 'data']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required config section: {section}")
                return False
        
        # Validate training section
        training = config.get('training', {})
        if 'batch_size' not in training:
            logger.error("Missing batch_size in training config")
            return False
        
        if 'num_epochs' not in training:
            logger.error("Missing num_epochs in training config")
            return False
        
        return True
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        model_config = config.get('model', {})
        
        if 'name' not in model_config:
            logger.error("Missing model name in config")
            return False
        
        return True













