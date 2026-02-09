"""Strategy pattern utilities."""

from typing import Dict, TypeVar, Callable, Optional, Any
import threading

T = TypeVar('T')
R = TypeVar('R')


class StrategyContext:
    """Context for strategy pattern."""
    
    def __init__(self, strategy: Optional[Callable] = None):
        self._strategy = strategy
        self._lock = threading.Lock()
    
    def set_strategy(self, strategy: Callable) -> None:
        """
        Set strategy.
        
        Args:
            strategy: Strategy function
        """
        with self._lock:
            self._strategy = strategy
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute strategy.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Strategy result
        """
        if self._strategy is None:
            raise ValueError("No strategy set")
        
        return self._strategy(*args, **kwargs)


class StrategyRegistry:
    """Registry for strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, strategy: Callable) -> None:
        """
        Register strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy function
        """
        with self._lock:
            self._strategies[name] = strategy
    
    def get(self, name: str) -> Optional[Callable]:
        """
        Get strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy function
        """
        return self._strategies.get(name)
    
    def execute(self, name: str, *args, **kwargs) -> Any:
        """
        Execute strategy by name.
        
        Args:
            name: Strategy name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Strategy result
        """
        strategy = self.get(name)
        if strategy is None:
            raise ValueError(f"Strategy '{name}' not found")
        
        return strategy(*args, **kwargs)


def strategy(name: str, registry: Optional[StrategyRegistry] = None) -> Callable:
    """
    Decorator to register strategy.
    
    Args:
        name: Strategy name
        registry: Strategy registry (creates global if None)
        
    Returns:
        Decorator function
    """
    if registry is None:
        registry = _global_strategy_registry
    
    def decorator(func: Callable) -> Callable:
        registry.register(name, func)
        return func
    
    return decorator


# Global strategy registry
_global_strategy_registry = StrategyRegistry()



