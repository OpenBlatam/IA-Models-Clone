"""
Device Helper
Device management utilities
"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DeviceHelper:
    """
    Device management helper
    """
    
    @staticmethod
    def get_device(use_gpu: bool = True, gpu_id: Optional[int] = None) -> torch.device:
        """
        Get appropriate device
        
        Args:
            use_gpu: Whether to use GPU
            gpu_id: Specific GPU ID (optional)
            
        Returns:
            Device
        """
        if use_gpu and torch.cuda.is_available():
            if gpu_id is not None:
                device = torch.device(f'cuda:{gpu_id}')
            else:
                device = torch.device('cuda')
            logger.info(f"Using GPU: {device}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        
        return device
    
    @staticmethod
    def get_available_gpus() -> int:
        """
        Get number of available GPUs
        
        Returns:
            Number of GPUs
        """
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    @staticmethod
    def get_gpu_memory_info(gpu_id: int = 0) -> dict:
        """
        Get GPU memory information
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            Memory information dictionary
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        return {
            'allocated_mb': torch.cuda.memory_allocated(gpu_id) / (1024 ** 2),
            'reserved_mb': torch.cuda.memory_reserved(gpu_id) / (1024 ** 2),
            'max_allocated_mb': torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 2),
            'max_reserved_mb': torch.cuda.max_memory_reserved(gpu_id) / (1024 ** 2),
        }
    
    @staticmethod
    def clear_gpu_cache() -> None:
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")



