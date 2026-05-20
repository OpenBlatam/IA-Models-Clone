"""
Conversion Utilities
Data type and format conversion utilities
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Converter:
    """
    Conversion utilities for different data types
    """
    
    @staticmethod
    def to_tensor(
        data: Union[np.ndarray, list, Image.Image, torch.Tensor],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert various types to tensor
        
        Args:
            data: Data to convert
            dtype: Target dtype (optional)
            device: Target device (optional)
            
        Returns:
            Tensor
        """
        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif isinstance(data, Image.Image):
            array = np.array(data)
            tensor = torch.from_numpy(array)
        elif isinstance(data, list):
            tensor = torch.tensor(data)
        else:
            raise TypeError(f"Cannot convert {type(data)} to tensor")
        
        if dtype is not None:
            tensor = tensor.to(dtype)
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    @staticmethod
    def to_numpy(
        data: Union[torch.Tensor, list, np.ndarray],
    ) -> np.ndarray:
        """
        Convert to numpy array
        
        Args:
            data: Data to convert
            
        Returns:
            Numpy array
        """
        if isinstance(data, torch.Tensor):
            if data.requires_grad:
                data = data.detach()
            return data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise TypeError(f"Cannot convert {type(data)} to numpy")
    
    @staticmethod
    def to_pil_image(
        data: Union[torch.Tensor, np.ndarray],
    ) -> Image.Image:
        """
        Convert to PIL Image
        
        Args:
            data: Data to convert
            
        Returns:
            PIL Image
        """
        if isinstance(data, torch.Tensor):
            if data.requires_grad:
                data = data.detach()
            array = data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            array = data
        else:
            raise TypeError(f"Cannot convert {type(data)} to PIL Image")
        
        # Handle different shapes
        if array.ndim == 3:
            if array.shape[0] == 3:  # CHW format
                array = array.transpose(1, 2, 0)
        elif array.ndim == 4:
            array = array[0]  # Take first batch
            if array.shape[0] == 3:
                array = array.transpose(1, 2, 0)
        
        # Normalize to 0-255
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        
        return Image.fromarray(array)
    
    @staticmethod
    def convert_dtype(
        tensor: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert tensor dtype safely
        
        Args:
            tensor: Tensor to convert
            target_dtype: Target dtype
            
        Returns:
            Converted tensor
        """
        return tensor.to(target_dtype)



