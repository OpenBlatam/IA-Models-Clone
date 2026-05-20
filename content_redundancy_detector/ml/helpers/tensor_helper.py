"""
Tensor Helper
Tensor manipulation utilities
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TensorHelper:
    """
    Tensor manipulation helper
    """
    
    @staticmethod
    def to_tensor(
        data: Union[np.ndarray, list, float, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert to tensor
        
        Args:
            data: Data to convert
            dtype: Target dtype
            device: Target device
            
        Returns:
            Tensor
        """
        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data)
        
        if dtype is not None:
            tensor = tensor.to(dtype)
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy
        
        Args:
            tensor: Tensor to convert
            
        Returns:
            Numpy array
        """
        if tensor.requires_grad:
            tensor = tensor.detach()
        return tensor.cpu().numpy()
    
    @staticmethod
    def move_to_device(
        data: Union[torch.Tensor, dict, list, tuple],
        device: torch.device,
    ) -> Union[torch.Tensor, dict, list, tuple]:
        """
        Move data to device
        
        Args:
            data: Data to move
            device: Target device
            
        Returns:
            Data on device
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: TensorHelper.move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(TensorHelper.move_to_device(item, device) for item in data)
        else:
            return data
    
    @staticmethod
    def detach_all(data: Union[torch.Tensor, dict, list, tuple]) -> Union[torch.Tensor, dict, list, tuple]:
        """
        Detach all tensors from computation graph
        
        Args:
            data: Data to detach
            
        Returns:
            Detached data
        """
        if isinstance(data, torch.Tensor):
            return data.detach()
        elif isinstance(data, dict):
            return {k: TensorHelper.detach_all(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(TensorHelper.detach_all(item) for item in data)
        else:
            return data



