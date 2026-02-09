"""
Strategy Pattern Utilities for Piel Mejorador AI SAM3
====================================================

Unified strategy pattern implementation utilities.
"""

import logging
from typing import TypeVar, Callable, Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Strategy(ABC):
    """Base strategy interface."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute strategy."""
        pass
    
    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__


@dataclass
class StrategyDefinition:
    """Strategy definition."""
    name: str
    strategy: Strategy
    description: Optional[str] = None
    enabled: bool = True


class StrategyContext:
    """Context for strategy pattern."""
    
    def __init__(self, default_strategy: Optional[Strategy] = None):
        """
        Initialize strategy context.
        
        Args:
            default_strategy: Optional default strategy
        """
        self._strategies: Dict[str, StrategyDefinition] = {}
        self._current_strategy: Optional[Strategy] = default_strategy
    
    def register(
        self,
        name: str,
        strategy: Strategy,
        description: Optional[str] = None,
        set_as_current: bool = False
    ):
        """
        Register strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy instance
            description: Optional description
            set_as_current: Whether to set as current strategy
        """
        self._strategies[name] = StrategyDefinition(
            name=name,
            strategy=strategy,
            description=description
        )
        
        if set_as_current or self._current_strategy is None:
            self._current_strategy = strategy
        
        logger.debug(f"Registered strategy: {name}")
    
    def set_strategy(self, name: str):
        """
        Set current strategy.
        
        Args:
            name: Strategy name
            
        Raises:
            KeyError: If strategy not found
        """
        if name not in self._strategies:
            raise KeyError(f"Strategy not found: {name}")
        
        definition = self._strategies[name]
        if not definition.enabled:
            raise ValueError(f"Strategy {name} is disabled")
        
        self._current_strategy = definition.strategy
        logger.debug(f"Set strategy: {name}")
    
    def get_strategy(self, name: Optional[str] = None) -> Optional[Strategy]:
        """
        Get strategy.
        
        Args:
            name: Optional strategy name (current if None)
            
        Returns:
            Strategy or None
        """
        if name:
            definition = self._strategies.get(name)
            return definition.strategy if definition else None
        return self._current_strategy
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute current strategy.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Strategy result
            
        Raises:
            ValueError: If no strategy set
        """
        if not self._current_strategy:
            raise ValueError("No strategy set")
        
        return self._current_strategy.execute(*args, **kwargs)
    
    def list_strategies(self) -> Dict[str, str]:
        """
        List all registered strategies.
        
        Returns:
            Dictionary of strategy names and descriptions
        """
        return {
            name: defn.description or "No description"
            for name, defn in self._strategies.items()
        }
    
    def enable_strategy(self, name: str):
        """Enable strategy."""
        if name in self._strategies:
            self._strategies[name].enabled = True
    
    def disable_strategy(self, name: str):
        """Disable strategy."""
        if name in self._strategies:
            self._strategies[name].enabled = True
            # If current strategy is disabled, clear it
            if self._current_strategy == self._strategies[name].strategy:
                self._current_strategy = None


class StrategyUtils:
    """Unified strategy pattern utilities."""
    
    @staticmethod
    def create_context(default_strategy: Optional[Strategy] = None) -> StrategyContext:
        """
        Create strategy context.
        
        Args:
            default_strategy: Optional default strategy
            
        Returns:
            StrategyContext
        """
        return StrategyContext(default_strategy)
    
    @staticmethod
    def create_function_strategy(
        name: str,
        func: Callable[..., R],
        description: Optional[str] = None
    ) -> Strategy:
        """
        Create strategy from function.
        
        Args:
            name: Strategy name
            func: Function to use as strategy
            description: Optional description
            
        Returns:
            Strategy instance
        """
        class FunctionStrategy(Strategy):
            def execute(self, *args, **kwargs) -> R:
                return func(*args, **kwargs)
            
            def get_name(self) -> str:
                return name
        
        return FunctionStrategy()


# Convenience functions
def create_context(default_strategy: Optional[Strategy] = None) -> StrategyContext:
    """Create strategy context."""
    return StrategyUtils.create_context(default_strategy)


def create_function_strategy(name: str, func: Callable[..., R], **kwargs) -> Strategy:
    """Create function strategy."""
    return StrategyUtils.create_function_strategy(name, func, **kwargs)




