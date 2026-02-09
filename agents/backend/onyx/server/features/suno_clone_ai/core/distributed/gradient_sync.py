"""
Gradient Synchronization

Utilities for synchronizing gradients in distributed training.
"""

import logging
import torch
import torch.distributed as dist
from typing import List

logger = logging.getLogger(__name__)


class GradientSynchronizer:
    """Synchronize gradients across processes."""
    
    @staticmethod
    def sync_gradients(model: torch.nn.Module) -> None:
        """
        Synchronize gradients across all processes.
        
        Args:
            model: Model with gradients to sync
        """
        if not dist.is_initialized():
            logger.warning("Distributed not initialized, skipping gradient sync")
            return
        
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()
    
    @staticmethod
    def all_reduce_gradients(
        gradients: List[torch.Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> List[torch.Tensor]:
        """
        All-reduce a list of gradients.
        
        Args:
            gradients: List of gradient tensors
            op: Reduction operation
            
        Returns:
            Synchronized gradients
        """
        if not dist.is_initialized():
            logger.warning("Distributed not initialized, returning original gradients")
            return gradients
        
        world_size = dist.get_world_size()
        
        for grad in gradients:
            if grad is not None:
                dist.all_reduce(grad, op=op)
                grad /= world_size
        
        return gradients


def sync_gradients(model: torch.nn.Module) -> None:
    """Convenience function to sync gradients."""
    GradientSynchronizer.sync_gradients(model)


def all_reduce_gradients(
    gradients: List[torch.Tensor],
    op: dist.ReduceOp = dist.ReduceOp.SUM
) -> List[torch.Tensor]:
    """Convenience function to all-reduce gradients."""
    return GradientSynchronizer.all_reduce_gradients(gradients, op)



