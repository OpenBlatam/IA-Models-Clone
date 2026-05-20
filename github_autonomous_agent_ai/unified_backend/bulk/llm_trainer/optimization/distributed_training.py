"""
Distributed Training Helper Module
===================================

Utilities for distributed and multi-GPU training.

Author: BUL System
Date: 2024
"""

import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DistributedTrainingHelper:
    """
    Helper for distributed training setup and configuration.
    
    Provides utilities for:
    - Multi-GPU training
    - Distributed data parallel (DDP)
    - Configuration management
    
    Example:
        >>> helper = DistributedTrainingHelper()
        >>> if helper.is_available():
        ...     config = helper.get_config()
    """
    
    def __init__(self):
        """Initialize DistributedTrainingHelper."""
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    def is_distributed(self) -> bool:
        """
        Check if running in distributed mode.
        
        Returns:
            True if distributed training is active
        """
        return self.world_size > 1
    
    def is_available(self) -> bool:
        """
        Check if distributed training is available.
        
        Returns:
            True if distributed training is supported
        """
        try:
            import torch.distributed as dist
            return dist.is_available()
        except ImportError:
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get distributed training configuration.
        
        Returns:
            Dictionary with configuration
        """
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "is_distributed": self.is_distributed(),
            "is_available": self.is_available(),
        }
    
    def setup_distributed(self) -> None:
        """
        Setup distributed training environment.
        
        Raises:
            RuntimeError: If distributed setup fails
        """
        if not self.is_available():
            logger.warning("Distributed training not available")
            return
        
        try:
            import torch.distributed as dist
            
            if not dist.is_initialized():
                # Initialize process group
                dist.init_process_group(backend="nccl" if self._has_cuda() else "gloo")
                logger.info(f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}")
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
            raise RuntimeError(f"Distributed setup failed: {e}") from e
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_recommendations(self) -> list[str]:
        """
        Get recommendations for distributed training.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self.is_distributed() and self.is_available():
            recommendations.append(
                "Distributed training is available but not active. "
                "Set WORLD_SIZE and RANK environment variables to enable."
            )
        
        if self.is_distributed():
            recommendations.append(
                f"Running distributed training with {self.world_size} processes"
            )
        
        return recommendations

