"""
Parallel Composer Module

Composer for parallel processing branches.
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from .models import ParallelModel


class ParallelComposer:
    """
    Composer for parallel processing branches.
    """
    
    def __init__(self):
        self.branches: Dict[str, nn.Module] = {}
        self.merge_strategy: str = "concat"  # "concat", "add", "multiply"
    
    def add_branch(self, name: str, branch: nn.Module) -> 'ParallelComposer':
        """
        Add a parallel branch.
        
        Args:
            name: Branch name.
            branch: Branch module.
        
        Returns:
            Self for chaining.
        """
        self.branches[name] = branch
        return self
    
    def set_merge_strategy(self, strategy: str) -> 'ParallelComposer':
        """
        Set merge strategy.
        
        Args:
            strategy: Merge strategy ("concat", "add", "multiply").
        
        Returns:
            Self for chaining.
        """
        self.merge_strategy = strategy
        return self
    
    def build(self) -> nn.Module:
        """
        Build parallel model.
        
        Returns:
            Parallel model.
        """
        return ParallelModel(
            branches=self.branches,
            merge_strategy=self.merge_strategy
        )



