"""
Strategy Pattern - Define strategy interface and context
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class IStrategy(ABC):
    """
    Interface for strategies
    """
    
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Execute strategy"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass


class StrategyContext:
    """
    Context for strategy pattern
    """
    
    def __init__(self, strategy: Optional[IStrategy] = None):
        self._strategy = strategy
    
    def set_strategy(self, strategy: IStrategy) -> None:
        """Set strategy"""
        self._strategy = strategy
        logger.info(f"Strategy set to: {strategy.name}")
    
    def get_strategy(self) -> Optional[IStrategy]:
        """Get current strategy"""
        return self._strategy
    
    def execute(self, data: Any) -> Any:
        """Execute current strategy"""
        if self._strategy is None:
            raise ValueError("No strategy set")
        
        return self._strategy.execute(data)








