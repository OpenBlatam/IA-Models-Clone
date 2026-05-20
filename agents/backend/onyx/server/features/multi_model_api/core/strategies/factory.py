"""
Strategy factory for creating execution strategies
Factory pattern for strategy instantiation
"""

import logging
from typing import Dict, Type
from ...api.exceptions import StrategyNotFoundException
from .base import ExecutionStrategy
from .parallel import ParallelStrategy
from .sequential import SequentialStrategy
from .consensus import ConsensusStrategy

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Factory for creating execution strategies"""
    
    _strategies: Dict[str, Type[ExecutionStrategy]] = {
        "parallel": ParallelStrategy,
        "sequential": SequentialStrategy,
        "consensus": ConsensusStrategy
    }
    
    @classmethod
    def create(cls, strategy_name: str) -> ExecutionStrategy:
        """
        Create an execution strategy
        
        Args:
            strategy_name: Name of the strategy (parallel, sequential, consensus)
            
        Returns:
            ExecutionStrategy instance
            
        Raises:
            StrategyNotFoundException: If strategy not found
        """
        strategy_class = cls._strategies.get(strategy_name.lower())
        
        if not strategy_class:
            available = ", ".join(cls._strategies.keys())
            raise StrategyNotFoundException(
                strategy=strategy_name,
                details={"available_strategies": available}
            )
        
        logger.debug(f"Creating {strategy_class.__name__} strategy")
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[ExecutionStrategy]):
        """
        Register a custom strategy
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        cls._strategies[name.lower()] = strategy_class
        logger.info(f"Registered custom strategy: {name}")
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of available strategy names"""
        return list(cls._strategies.keys())




