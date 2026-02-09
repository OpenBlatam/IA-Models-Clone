"""
Optimization Core Adapter
=========================

Adapter class to bridge TruthGPT API optimizers with optimization_core.

This adapter maintains the TensorFlow-like interface while using
optimization_core under the hood when available.

Single Responsibility: Manage optimizer adapter lifecycle and configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

from .core_detector import is_optimization_core_available
from .optimizer_factories import create_optimizer_from_core, create_pytorch_optimizer
from .optimizer_statistics import OptimizerStatistics
from .optimizer_serializer import OptimizerSerializer
from .optimizer_health_checker import OptimizerHealthChecker
from .optimizer_config_builder import OptimizerConfigBuilder
from .optimizer_constants import DEFAULT_LEARNING_RATE, normalize_optimizer_type
from .paper_integration import (
    is_paper_registry_available,
    get_paper_enhanced_params
)

logger = logging.getLogger(__name__)

# Module-level statistics instance
_stats = OptimizerStatistics()


class OptimizationCoreAdapter:
    """
    Adapter class to bridge TruthGPT API optimizers with optimization_core.
    
    This adapter maintains the TensorFlow-like interface while using
    optimization_core under the hood when available.
    
    Responsibilities:
    - Create optimizer instances (delegates to factory functions)
    - Manage fallback to PyTorch
    - Provide configuration interface
    """
    
    def __init__(
        self,
        optimizer_type: str,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        use_core: bool = True,
        **kwargs
    ):
        """
        Initialize the adapter.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            use_core: Whether to use optimization_core if available
            **kwargs: Additional optimizer parameters
        """
        self.optimizer_type = normalize_optimizer_type(optimizer_type)
        self.learning_rate = learning_rate
        self.use_core = use_core and is_optimization_core_available()
        
        # Paper integration - extract use_papers before creating kwargs copy
        self.use_papers = kwargs.pop('use_papers', True)
        self.kwargs = kwargs.copy()  # Work with a copy to avoid side effects
        self._paper_params = self._apply_paper_enhancements(self.kwargs)
        
        self._core_optimizer = None
        
        if self.use_core:
            self._core_optimizer = create_optimizer_from_core(
                self.optimizer_type,
                learning_rate,
                **self.kwargs
            )
    
    def get_optimizer(self) -> Optional[Any]:
        """Get the underlying optimizer."""
        return self._core_optimizer
    
    def _apply_paper_enhancements(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply paper-based enhancements to optimizer parameters.
        
        Args:
            kwargs: Optimizer parameters (modified in place)
        
        Returns:
            Dictionary with paper parameters if applied, None otherwise
        """
        if not (self.use_papers and is_paper_registry_available()):
            return None
        
        try:
            paper_params = get_paper_enhanced_params(self.optimizer_type)
            if not paper_params:
                return None
            
            # Merge paper params (user kwargs take precedence)
            for key, value in paper_params.items():
                if key not in kwargs:
                    kwargs[key] = value
            
            logger.debug(f"📚 Applied paper enhancements for {self.optimizer_type}")
            return paper_params
        except Exception as e:
            logger.debug(f"Paper enhancement failed: {e}")
            return None
    
    # Strategies for trying core optimizer (consolidated)
    _CORE_OPTIMIZER_STRATEGIES = [
        lambda opt, params: opt(params) if callable(opt) else None,
        lambda opt, params: opt.create_optimizer(params) if hasattr(opt, 'create_optimizer') else None,
        lambda opt, params: opt.get_optimizer(params) if hasattr(opt, 'get_optimizer') else None,
        lambda opt, params: opt if (hasattr(opt, 'step') and hasattr(opt, 'zero_grad')) else None,
    ]
    
    def _try_core_optimizer(self, parameters: Any) -> Optional[Any]:
        """
        Try to use core optimizer with given parameters.
        
        Uses multiple strategies in order until one succeeds.
        
        Args:
            parameters: Model parameters to optimize
        
        Returns:
            Optimizer instance if successful, None otherwise
        """
        if self._core_optimizer is None:
            return None
        
        for strategy in self._CORE_OPTIMIZER_STRATEGIES:
            try:
                result = strategy(self._core_optimizer, parameters)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"Core optimizer strategy failed: {e}")
                continue
        
        logger.warning("All core optimizer strategies failed, falling back to PyTorch")
        return None
    
    @staticmethod
    def _is_pytorch_optimizer(obj: Any) -> bool:
        """Check if object is a PyTorch optimizer."""
        return hasattr(obj, 'step') and hasattr(obj, 'zero_grad')
    
    def __call__(self, parameters: Any) -> Any:
        """
        Create optimizer for given parameters.
        
        Tries to use optimization_core optimizer first, falls back to PyTorch.
        
        Args:
            parameters: Model parameters to optimize
        
        Returns:
            Optimizer instance
        """
        # Try core optimizer first
        optimizer = self._try_core_optimizer(parameters)
        if optimizer is not None:
            _stats.record_creation(from_core=True)
            logger.debug(f"✅ Created {self.optimizer_type} via optimization_core")
            return optimizer
        
        # Fallback to PyTorch optimizer
        try:
            optimizer = create_pytorch_optimizer(
                self.optimizer_type,
                parameters,
                self.learning_rate,
                **self.kwargs
            )
            _stats.record_creation(from_core=False)
            logger.debug(f"✅ Created {self.optimizer_type} via PyTorch fallback")
            return optimizer
        except ImportError:
            raise ImportError("PyTorch is required as fallback. Install with: pip install torch")
        except Exception as e:
            _stats.record_error()
            logger.error(f"Failed to create PyTorch optimizer: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration.
        
        Returns:
            Dictionary containing optimizer configuration
        """
        
        using_core = self._core_optimizer is not None
        config = OptimizerConfigBuilder.build_full_config(
            optimizer_type=self.optimizer_type,
            learning_rate=self.learning_rate,
            use_core=self.use_core,
            using_core=using_core,
            kwargs=self.kwargs,
            core_optimizer=self._core_optimizer,
            logger=logger
        )
        
        # Add paper info if available
        if is_paper_registry_available():
            try:
                config['paper_enhanced'] = self._paper_params is not None
                if self._paper_params:
                    config['paper_params'] = self._paper_params
            except Exception:
                pass
        
        return config
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        if self.use_core and self._core_optimizer is not None:
            core_status = "optimization_core"
        else:
            core_status = "PyTorch fallback"
        return f"{self.optimizer_type.capitalize()}(lr={self.learning_rate}, backend={core_status})"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize optimizer configuration to dictionary."""
        return OptimizerSerializer.serialize(
            optimizer_type=self.optimizer_type,
            learning_rate=self.learning_rate,
            kwargs=self.kwargs,
            use_core=self.use_core,
            version='1.0'
        )
    
    @classmethod
    def deserialize(cls, config: Dict[str, Any]) -> 'OptimizationCoreAdapter':
        """Deserialize optimizer from configuration."""
        return cls(
            optimizer_type=config['optimizer_type'],
            learning_rate=config['learning_rate'],
            use_core=config.get('use_core', True),
            **config.get('kwargs', {})
        )
    
    def save(self, filepath: Union[str, Path]):
        """
        Save optimizer configuration to file.
        
        Delegates to OptimizerSerializer for consistency.
        """
        config = self.serialize()
        OptimizerSerializer.save(config, filepath)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'OptimizationCoreAdapter':
        """
        Load optimizer configuration from file.
        
        Delegates to OptimizerSerializer for consistency.
        """
        config = OptimizerSerializer.load(filepath)
        return cls.deserialize(config)
    
    def compare(self, other: 'OptimizationCoreAdapter') -> Dict[str, Any]:
        """Compare this optimizer with another."""
        return {
            'same_type': self.optimizer_type == other.optimizer_type,
            'same_lr': self.learning_rate == other.learning_rate,
            'same_backend': (
                (self._core_optimizer is not None) == (other._core_optimizer is not None)
            ),
            'params_match': self.kwargs == other.kwargs
        }
    
    def clone(self) -> 'OptimizationCoreAdapter':
        """Create a copy of this optimizer adapter."""
        return OptimizationCoreAdapter(
            optimizer_type=self.optimizer_type,
            learning_rate=self.learning_rate,
            use_core=self.use_core,
            **self.kwargs.copy()
        )
    
    def update_learning_rate(self, new_lr: float):
        """Update learning rate dynamically."""
        if new_lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {new_lr}")
        
        self.learning_rate = new_lr
        
        if self._core_optimizer is not None:
            if hasattr(self._core_optimizer, 'update_learning_rate'):
                try:
                    self._core_optimizer.update_learning_rate(new_lr)
                    logger.debug(f"✅ Updated core optimizer LR to {new_lr}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update core optimizer LR: {e}")
        
        logger.debug(f"✅ Updated learning rate to {new_lr}")
    
    def get_paper_info(self) -> Dict[str, Any]:
        """
        Get information about papers relevant to this optimizer.
        
        Returns:
            Dictionary with paper information
        """
        from .paper_integration import (
            suggest_papers_for_optimizer,
            get_paper_based_recommendations
        )
        
        if not is_paper_registry_available():
            return {
                'available': False,
                'message': 'Paper registry not available'
            }
        
        try:
            suggested = suggest_papers_for_optimizer(self.optimizer_type)
            recommendations = get_paper_based_recommendations(
                optimizer_type=self.optimizer_type
            )
            
            return {
                'available': True,
                'suggested_papers': suggested,
                'recommendations': recommendations,
                'paper_params_applied': self._paper_params is not None,
                'paper_params': self._paper_params
            }
        except Exception as e:
            logger.error(f"Error getting paper info: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the optimizer."""
        return OptimizerHealthChecker.check(
            optimizer_type=self.optimizer_type,
            learning_rate=self.learning_rate,
            use_core=self.use_core,
            core_optimizer=self._core_optimizer,
            logger=logger
        )

