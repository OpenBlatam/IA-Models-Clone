"""
Optimizer Health Checker
========================

Performs health checks on optimizer configurations.
"""

from typing import Dict, Any
from .core_detector import is_optimization_core_available


class OptimizerHealthChecker:
    """
    Performs health checks on optimizer configurations.
    
    Responsibilities:
    - Check optimizer health status
    - Identify configuration issues
    - Validate dependencies
    """
    
    @staticmethod
    def check(
        optimizer_type: str,
        learning_rate: float,
        use_core: bool,
        core_optimizer: Any,
        logger
    ) -> Dict[str, Any]:
        """
        Perform health check on optimizer.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            use_core: Whether to use optimization_core
            core_optimizer: Core optimizer instance or None
            logger: Logger instance
        
        Returns:
            Dictionary with health status and issues
        """
        health = {
            'status': 'healthy',
            'optimizer_type': optimizer_type,
            'learning_rate': learning_rate,
            'core_available': is_optimization_core_available(),
            'using_core': core_optimizer is not None,
            'issues': []
        }
        
        # Check learning rate
        if learning_rate <= 0:
            health['status'] = 'unhealthy'
            health['issues'].append('Invalid learning rate')
        
        # Check core optimizer availability
        if use_core and core_optimizer is None:
            health['issues'].append('Core optimizer not available but requested')
        
        # Check PyTorch availability
        pytorch_available = OptimizerHealthChecker._check_pytorch_available()
        health['pytorch_available'] = pytorch_available
        
        if not pytorch_available and (not use_core or core_optimizer is None):
            health['status'] = 'unhealthy'
            health['issues'].append('PyTorch not available and no core optimizer')
        
        return health
    
    @staticmethod
    def _check_pytorch_available() -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False

