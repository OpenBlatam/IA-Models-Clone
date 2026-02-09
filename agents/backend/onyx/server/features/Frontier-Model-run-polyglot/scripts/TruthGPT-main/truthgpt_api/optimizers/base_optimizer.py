"""
Base Optimizer - Base class for all optimizers.

Single Responsibility: Provide common functionality for all optimizers.
Eliminates duplication across Adam, SGD, RMSprop, etc.
"""

from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from .adapters import OptimizationCoreAdapter, is_optimization_core_available


class BaseOptimizer(ABC):
    """
    Base class for all optimizers.
    
    Single Responsibility: Provide common functionality for optimizer creation,
    configuration, and adapter management.
    
    Eliminates duplication across Adam, SGD, RMSprop, Adagrad, AdamW.
    """
    
    def __init__(
        self,
        optimizer_type: str,
        learning_rate: float,
        name: Optional[str] = None,
        use_optimization_core: bool = True,
        **kwargs
    ):
        """
        Initialize base optimizer.
        
        Args:
            optimizer_type: Type of optimizer (adam, sgd, rmsprop, etc.)
            learning_rate: Learning rate
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
            **kwargs: Additional optimizer-specific parameters
        """
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.name = name or optimizer_type.capitalize()
        self.use_optimization_core = use_optimization_core and is_optimization_core_available()
        self.kwargs = kwargs
        
        self._adapter = None
        self._optimizer = None
        self._parameters = None
        
        # Initialize adapter if optimization_core is available
        if self.use_optimization_core:
            self._initialize_adapter()
    
    def _initialize_adapter(self) -> None:
        """
        Initialize optimization_core adapter.
        
        Called during __init__ if use_optimization_core is True.
        Falls back silently if adapter creation fails.
        """
        try:
            self._adapter = OptimizationCoreAdapter(
                optimizer_type=self.optimizer_type,
                learning_rate=self.learning_rate,
                use_core=True,
                **self.kwargs
            )
        except Exception:
            # Fallback to PyTorch if adapter fails
            self.use_optimization_core = False
            self._adapter = None
    
    def _create_optimizer(self, parameters):
        """
        Create optimizer (from optimization_core or PyTorch).
        
        Args:
            parameters: Model parameters to optimize
        
        Returns:
            Optimizer instance
        """
        self._parameters = parameters
        
        # Try to use optimization_core adapter first
        if self._adapter is not None:
            try:
                return self._adapter(parameters)
            except Exception:
                # Fallback to PyTorch
                pass
        
        # Fallback to PyTorch optimizer
        return self._create_pytorch_optimizer(parameters)
    
    @abstractmethod
    def _create_pytorch_optimizer(self, parameters):
        """
        Create PyTorch optimizer as fallback.
        
        Must be implemented by subclasses to provide optimizer-specific logic.
        
        Args:
            parameters: Model parameters to optimize
        
        Returns:
            PyTorch optimizer instance
        """
        pass
    
    def __call__(self, parameters):
        """Create optimizer for given parameters."""
        return self._create_optimizer(parameters)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration.
        
        Returns:
            Dictionary with optimizer configuration
        """
        config = {
            'name': self.name,
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'using_optimization_core': self.use_optimization_core and self._adapter is not None,
            **self._get_optimizer_specific_config()
        }
        
        if self._adapter is not None:
            try:
                adapter_config = self._adapter.get_config()
                config['optimization_core'] = adapter_config
            except Exception:
                pass
        
        return config
    
    @abstractmethod
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """
        Get optimizer-specific configuration parameters.
        
        Must be implemented by subclasses to return their specific parameters.
        
        Returns:
            Dictionary with optimizer-specific parameters
        """
        pass
    
    def __repr__(self):
        """String representation of the optimizer."""
        core_status = " (optimization_core)" if (self.use_optimization_core and self._adapter) else ""
        params_str = self._format_parameters()
        return f"{self.name}({params_str}){core_status}"
    
    @abstractmethod
    def _format_parameters(self) -> str:
        """
        Format optimizer parameters for string representation.
        
        Must be implemented by subclasses to format their specific parameters.
        
        Returns:
            Formatted parameter string
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for this optimizer instance.
        
        Returns:
            Dictionary with optimizer statistics
        """
        return {
            'name': self.name,
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'using_optimization_core': self.use_optimization_core and self._adapter is not None,
            'parameters': self.kwargs.copy()
        }

