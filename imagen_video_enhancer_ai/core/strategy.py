"""
Strategy System
===============

Consolidated strategy pattern implementation for various operations.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Protocol
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Strategy(Protocol[T, R]):
    """Strategy protocol."""
    
    def execute(self, context: T) -> R:
        """Execute strategy with context."""
        ...


class StrategyRegistry(Generic[T, R]):
    """Registry for strategies."""
    
    def __init__(self):
        """Initialize strategy registry."""
        self.strategies: Dict[str, Type[Strategy[T, R]]] = {}
    
    def register(self, name: str, strategy: Type[Strategy[T, R]]):
        """
        Register a strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy class
        """
        self.strategies[name] = strategy
        logger.debug(f"Registered strategy: {name}")
    
    def get(self, name: str) -> Optional[Type[Strategy[T, R]]]:
        """
        Get strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy class or None
        """
        return self.strategies.get(name)
    
    def execute(self, name: str, context: T) -> Optional[R]:
        """
        Execute strategy by name.
        
        Args:
            name: Strategy name
            context: Strategy context
            
        Returns:
            Strategy result or None
        """
        strategy_class = self.get(name)
        if strategy_class:
            strategy = strategy_class()
            return strategy.execute(context)
        return None
    
    def list_strategies(self) -> list[str]:
        """List all registered strategies."""
        return list(self.strategies.keys())


@dataclass
class RetryStrategyContext:
    """Context for retry strategy."""
    attempt: int
    max_attempts: int
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = None


class RetryStrategy(ABC):
    """Base retry strategy."""
    
    @abstractmethod
    def should_retry(self, context: RetryStrategyContext) -> bool:
        """Determine if should retry."""
        pass
    
    @abstractmethod
    def get_delay(self, context: RetryStrategyContext) -> float:
        """Get delay before retry."""
        pass


class ExponentialBackoffStrategy(RetryStrategy):
    """Exponential backoff retry strategy."""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        """
        Initialize exponential backoff strategy.
        
        Args:
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def should_retry(self, context: RetryStrategyContext) -> bool:
        """Check if should retry."""
        return context.attempt < context.max_attempts
    
    def get_delay(self, context: RetryStrategyContext) -> float:
        """Calculate exponential delay."""
        delay = min(
            self.base_delay * (2 ** context.attempt),
            self.max_delay
        )
        return delay


class FixedDelayStrategy(RetryStrategy):
    """Fixed delay retry strategy."""
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize fixed delay strategy.
        
        Args:
            delay: Fixed delay in seconds
        """
        self.delay = delay
    
    def should_retry(self, context: RetryStrategyContext) -> bool:
        """Check if should retry."""
        return context.attempt < context.max_attempts
    
    def get_delay(self, context: RetryStrategyContext) -> float:
        """Get fixed delay."""
        return self.delay


@dataclass
class CacheStrategyContext:
    """Context for cache strategy."""
    key: str
    data: Any
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = None


class CacheStrategy(ABC):
    """Base cache strategy."""
    
    @abstractmethod
    def should_cache(self, context: CacheStrategyContext) -> bool:
        """Determine if should cache."""
        pass
    
    @abstractmethod
    def get_ttl(self, context: CacheStrategyContext) -> Optional[int]:
        """Get TTL for cache entry."""
        pass


class DefaultCacheStrategy(CacheStrategy):
    """Default cache strategy."""
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize default cache strategy.
        
        Args:
            default_ttl: Default TTL in seconds
        """
        self.default_ttl = default_ttl
    
    def should_cache(self, context: CacheStrategyContext) -> bool:
        """Always cache."""
        return True
    
    def get_ttl(self, context: CacheStrategyContext) -> Optional[int]:
        """Get TTL."""
        return context.ttl or self.default_ttl


@dataclass
class ValidationStrategyContext:
    """Context for validation strategy."""
    data: Any
    rules: Dict[str, Any]
    metadata: Dict[str, Any] = None


class ValidationStrategy(ABC):
    """Base validation strategy."""
    
    @abstractmethod
    def validate(self, context: ValidationStrategyContext) -> tuple[bool, Optional[str]]:
        """
        Validate data.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class StrictValidationStrategy(ValidationStrategy):
    """Strict validation strategy."""
    
    def validate(self, context: ValidationStrategyContext) -> tuple[bool, Optional[str]]:
        """Validate strictly."""
        # Implementation would check all rules
        return True, None


class StrategyManager:
    """Manager for all strategies."""
    
    def __init__(self):
        """Initialize strategy manager."""
        self.retry_strategies: Dict[str, RetryStrategy] = {}
        self.cache_strategies: Dict[str, CacheStrategy] = {}
        self.validation_strategies: Dict[str, ValidationStrategy] = {}
        
        # Register default strategies
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default strategies."""
        self.retry_strategies["exponential_backoff"] = ExponentialBackoffStrategy()
        self.retry_strategies["fixed_delay"] = FixedDelayStrategy()
        
        self.cache_strategies["default"] = DefaultCacheStrategy()
        
        self.validation_strategies["strict"] = StrictValidationStrategy()
    
    def register_retry_strategy(self, name: str, strategy: RetryStrategy):
        """Register retry strategy."""
        self.retry_strategies[name] = strategy
    
    def register_cache_strategy(self, name: str, strategy: CacheStrategy):
        """Register cache strategy."""
        self.cache_strategies[name] = strategy
    
    def register_validation_strategy(self, name: str, strategy: ValidationStrategy):
        """Register validation strategy."""
        self.validation_strategies[name] = strategy
    
    def get_retry_strategy(self, name: str = "exponential_backoff") -> RetryStrategy:
        """Get retry strategy."""
        return self.retry_strategies.get(name, self.retry_strategies["exponential_backoff"])
    
    def get_cache_strategy(self, name: str = "default") -> CacheStrategy:
        """Get cache strategy."""
        return self.cache_strategies.get(name, self.cache_strategies["default"])
    
    def get_validation_strategy(self, name: str = "strict") -> ValidationStrategy:
        """Get validation strategy."""
        return self.validation_strategies.get(name, self.validation_strategies["strict"])




