"""
Strategy Pattern Implementation

Flexible algorithm selection and execution pattern
for the AI History Comparison System.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable
from dataclasses import dataclass
from enum import Enum
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class StrategyType(Enum):
    """Strategy type enumeration"""
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    PROCESSING = "processing"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    OPTIMIZATION = "optimization"


@dataclass
class StrategyContext:
    """Context for strategy execution"""
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Strategy(ABC, Generic[T, R]):
    """Abstract strategy interface"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._enabled = True
        self._priority = 0
    
    @abstractmethod
    async def execute(self, context: StrategyContext, data: T) -> R:
        """Execute strategy with given context and data"""
        pass
    
    @abstractmethod
    def can_handle(self, context: StrategyContext, data: T) -> bool:
        """Check if strategy can handle the given context and data"""
        pass
    
    def get_priority(self) -> int:
        """Get strategy priority (higher = more preferred)"""
        return self._priority
    
    def set_priority(self, priority: int):
        """Set strategy priority"""
        self._priority = priority
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self._enabled
    
    def enable(self):
        """Enable strategy"""
        self._enabled = True
    
    def disable(self):
        """Disable strategy"""
        self._enabled = False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata"""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self._enabled,
            "priority": self._priority
        }


class StrategyContext:
    """Strategy execution context manager"""
    
    def __init__(self, strategy_type: StrategyType, **kwargs):
        self.strategy_type = strategy_type
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})
        self.timeout = kwargs.get('timeout')
        self.retry_count = kwargs.get('retry_count', 0)
        self.max_retries = kwargs.get('max_retries', 3)
        self._start_time = None
        self._end_time = None
    
    async def __aenter__(self):
        self._start_time = asyncio.get_event_loop().time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._end_time = asyncio.get_event_loop().time()
        if exc_type:
            logger.error(f"Strategy execution failed: {exc_val}")
    
    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds"""
        if self._start_time and self._end_time:
            return self._end_time - self._start_time
        return None
    
    def should_retry(self) -> bool:
        """Check if strategy should be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry count"""
        self.retry_count += 1


class StrategyManager:
    """Manages strategy registration and execution"""
    
    def __init__(self):
        self._strategies: Dict[StrategyType, List[Strategy]] = {}
        self._strategy_registry: Dict[str, Strategy] = {}
        self._lock = asyncio.Lock()
    
    def register_strategy(self, strategy: Strategy, strategy_type: StrategyType) -> None:
        """Register a strategy for a specific type"""
        if strategy_type not in self._strategies:
            self._strategies[strategy_type] = []
        
        if strategy not in self._strategies[strategy_type]:
            self._strategies[strategy_type].append(strategy)
            # Sort by priority (higher priority first)
            self._strategies[strategy_type].sort(
                key=lambda s: s.get_priority(), reverse=True
            )
        
        self._strategy_registry[strategy.name] = strategy
        logger.info(f"Registered strategy '{strategy.name}' for type '{strategy_type.value}'")
    
    def unregister_strategy(self, strategy_name: str) -> None:
        """Unregister a strategy"""
        if strategy_name in self._strategy_registry:
            strategy = self._strategy_registry[strategy_name]
            
            # Remove from type-specific lists
            for strategies in self._strategies.values():
                if strategy in strategies:
                    strategies.remove(strategy)
            
            # Remove from registry
            del self._strategy_registry[strategy_name]
            logger.info(f"Unregistered strategy '{strategy_name}'")
    
    def get_strategy(self, strategy_name: str) -> Optional[Strategy]:
        """Get strategy by name"""
        return self._strategy_registry.get(strategy_name)
    
    def get_strategies(self, strategy_type: StrategyType) -> List[Strategy]:
        """Get all strategies for a specific type"""
        return self._strategies.get(strategy_type, []).copy()
    
    def get_enabled_strategies(self, strategy_type: StrategyType) -> List[Strategy]:
        """Get enabled strategies for a specific type"""
        return [s for s in self.get_strategies(strategy_type) if s.is_enabled()]
    
    async def execute_strategy(self, strategy_name: str, context: StrategyContext, 
                             data: Any) -> Any:
        """Execute a specific strategy by name"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        if not strategy.is_enabled():
            raise ValueError(f"Strategy '{strategy_name}' is disabled")
        
        if not strategy.can_handle(context, data):
            raise ValueError(f"Strategy '{strategy_name}' cannot handle the given context and data")
        
        return await self._execute_with_retry(strategy, context, data)
    
    async def execute_best_strategy(self, context: StrategyContext, data: Any) -> Any:
        """Execute the best available strategy for the context"""
        strategies = self.get_enabled_strategies(context.strategy_type)
        
        if not strategies:
            raise ValueError(f"No enabled strategies found for type '{context.strategy_type.value}'")
        
        # Find the first strategy that can handle the context and data
        for strategy in strategies:
            if strategy.can_handle(context, data):
                return await self._execute_with_retry(strategy, context, data)
        
        raise ValueError(f"No strategy can handle the given context and data")
    
    async def execute_all_strategies(self, context: StrategyContext, data: Any) -> List[Any]:
        """Execute all applicable strategies and return results"""
        strategies = self.get_enabled_strategies(context.strategy_type)
        results = []
        
        for strategy in strategies:
            if strategy.can_handle(context, data):
                try:
                    result = await self._execute_with_retry(strategy, context, data)
                    results.append({
                        "strategy": strategy.name,
                        "result": result,
                        "success": True
                    })
                except Exception as e:
                    results.append({
                        "strategy": strategy.name,
                        "error": str(e),
                        "success": False
                    })
        
        return results
    
    async def _execute_with_retry(self, strategy: Strategy, context: StrategyContext, 
                                 data: Any) -> Any:
        """Execute strategy with retry logic"""
        last_exception = None
        
        while context.should_retry():
            try:
                if context.timeout:
                    return await asyncio.wait_for(
                        strategy.execute(context, data),
                        timeout=context.timeout
                    )
                else:
                    return await strategy.execute(context, data)
            except Exception as e:
                last_exception = e
                context.increment_retry()
                logger.warning(f"Strategy '{strategy.name}' failed (attempt {context.retry_count}): {e}")
                
                if context.should_retry():
                    await asyncio.sleep(0.1 * context.retry_count)  # Exponential backoff
        
        raise last_exception or Exception("Strategy execution failed")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about all registered strategies"""
        info = {}
        for strategy_type, strategies in self._strategies.items():
            info[strategy_type.value] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "enabled": s.is_enabled(),
                    "priority": s.get_priority()
                }
                for s in strategies
            ]
        return info


class CompositeStrategy(Strategy[T, R]):
    """Strategy that combines multiple strategies"""
    
    def __init__(self, name: str, strategies: List[Strategy], 
                 combination_method: str = "sequential"):
        super().__init__(name, f"Composite strategy combining {len(strategies)} strategies")
        self.strategies = strategies
        self.combination_method = combination_method
    
    async def execute(self, context: StrategyContext, data: T) -> R:
        """Execute composite strategy"""
        if self.combination_method == "sequential":
            return await self._execute_sequential(context, data)
        elif self.combination_method == "parallel":
            return await self._execute_parallel(context, data)
        elif self.combination_method == "pipeline":
            return await self._execute_pipeline(context, data)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    async def _execute_sequential(self, context: StrategyContext, data: T) -> R:
        """Execute strategies sequentially"""
        result = data
        for strategy in self.strategies:
            if strategy.is_enabled() and strategy.can_handle(context, result):
                result = await strategy.execute(context, result)
        return result
    
    async def _execute_parallel(self, context: StrategyContext, data: T) -> R:
        """Execute strategies in parallel"""
        tasks = []
        for strategy in self.strategies:
            if strategy.is_enabled() and strategy.can_handle(context, data):
                tasks.append(strategy.execute(context, data))
        
        if not tasks:
            return data
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Return the first successful result
        for result in results:
            if not isinstance(result, Exception):
                return result
        
        # If all failed, raise the first exception
        raise results[0] if results else Exception("No strategies executed")
    
    async def _execute_pipeline(self, context: StrategyContext, data: T) -> R:
        """Execute strategies as a pipeline"""
        result = data
        for strategy in self.strategies:
            if strategy.is_enabled() and strategy.can_handle(context, result):
                result = await strategy.execute(context, result)
        return result
    
    def can_handle(self, context: StrategyContext, data: T) -> bool:
        """Check if any strategy can handle the context and data"""
        return any(
            strategy.is_enabled() and strategy.can_handle(context, data)
            for strategy in self.strategies
        )


# Global strategy manager instance
strategy_manager = StrategyManager()


# Convenience functions
def register_strategy(strategy: Strategy, strategy_type: StrategyType):
    """Register a strategy"""
    strategy_manager.register_strategy(strategy, strategy_type)


def get_strategy(strategy_name: str) -> Optional[Strategy]:
    """Get strategy by name"""
    return strategy_manager.get_strategy(strategy_name)


async def execute_strategy(strategy_name: str, context: StrategyContext, data: Any) -> Any:
    """Execute a specific strategy"""
    return await strategy_manager.execute_strategy(strategy_name, context, data)


async def execute_best_strategy(context: StrategyContext, data: Any) -> Any:
    """Execute the best available strategy"""
    return await strategy_manager.execute_best_strategy(context, data)





















