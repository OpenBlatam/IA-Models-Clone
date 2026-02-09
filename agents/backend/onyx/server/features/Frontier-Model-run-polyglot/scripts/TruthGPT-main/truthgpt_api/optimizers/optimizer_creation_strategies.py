"""
Optimizer Creation Strategies
=============================

Strategy pattern for creating optimizers from different backends.
Each strategy encapsulates a specific creation approach.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class OptimizerCreationStrategy(ABC):
    """
    Base class for optimizer creation strategies.
    
    Responsibilities:
    - Define interface for optimizer creation
    - Handle errors gracefully
    """
    
    @abstractmethod
    def create(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """
        Create an optimizer using this strategy.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            **kwargs: Additional parameters
        
        Returns:
            Optimizer instance or None if creation fails
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass


class FactoryStrategy(OptimizerCreationStrategy):
    """Strategy for creating optimizers via factory functions."""
    
    def __init__(self, factories: Dict[str, Any]):
        """
        Initialize factory strategy.
        
        Args:
            factories: Dictionary of factory functions
        """
        self._factories = factories
    
    @property
    def name(self) -> str:
        return "factory"
    
    def create(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """Create optimizer via factory."""
        if not self._factories:
            return None
        
        try:
            if 'main' in self._factories:
                factory = self._factories['main']
                if hasattr(factory, 'create'):
                    result = factory.create(
                        optimizer_type,
                        learning_rate=learning_rate,
                        **kwargs
                    )
                    if result:
                        logger.debug(f"✅ Created {optimizer_type} via factory")
                        return result
            elif 'create_optimizer' in self._factories:
                create_func = self._factories['create_optimizer']
                result = create_func(
                    optimizer_type,
                    learning_rate=learning_rate,
                    **kwargs
                )
                if result:
                    logger.debug(f"✅ Created {optimizer_type} via create_optimizer")
                    return result
        except Exception as e:
            logger.debug(f"Factory creation failed: {e}")
        
        return None


class TensorFlowStrategy(OptimizerCreationStrategy):
    """Strategy for creating optimizers via TensorFlow-inspired optimizers."""
    
    def __init__(self, available: bool):
        """
        Initialize TensorFlow strategy.
        
        Args:
            available: Whether TensorFlow optimizers are available
        """
        self._available = available
    
    @property
    def name(self) -> str:
        return "tensorflow"
    
    def create(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """Create optimizer via TensorFlow-inspired optimizer."""
        if not self._available:
            return None
        
        try:
            from optimization_core.optimizers.tensorflow.tensorflow_inspired_optimizer import (
                TensorFlowInspiredOptimizer
            )
            supported_types = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamw']
            if optimizer_type in supported_types:
                result = TensorFlowInspiredOptimizer(
                    learning_rate=learning_rate,
                    optimizer_type=optimizer_type,
                    **kwargs
                )
                logger.debug(f"✅ Created {optimizer_type} via TensorFlow optimizer")
                return result
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"TensorFlow optimizer creation failed: {e}")
        
        return None


class UnifiedOptimizerStrategy(OptimizerCreationStrategy):
    """Strategy for creating optimizers via UnifiedOptimizer."""
    
    def __init__(self, available: bool):
        """
        Initialize unified optimizer strategy.
        
        Args:
            available: Whether core optimizers are available
        """
        self._available = available
    
    @property
    def name(self) -> str:
        return "unified"
    
    def create(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """Create optimizer via UnifiedOptimizer."""
        if not self._available:
            return None
        
        try:
            from optimization_core.optimizers.core.unified_optimizer import (
                UnifiedOptimizer
            )
            result = UnifiedOptimizer(
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                **kwargs
            )
            logger.debug(f"✅ Created {optimizer_type} via UnifiedOptimizer")
            return result
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"UnifiedOptimizer creation failed: {e}")
        
        return None


class GenericOptimizerStrategy(OptimizerCreationStrategy):
    """Strategy for creating optimizers via GenericOptimizer."""
    
    def __init__(self, available: bool):
        """
        Initialize generic optimizer strategy.
        
        Args:
            available: Whether core optimizers are available
        """
        self._available = available
    
    @property
    def name(self) -> str:
        return "generic"
    
    def create(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """Create optimizer via GenericOptimizer."""
        if not self._available:
            return None
        
        try:
            from optimization_core.optimizers.core.generic_optimizer import (
                GenericOptimizer
            )
            result = GenericOptimizer(
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                **kwargs
            )
            logger.debug(f"✅ Created {optimizer_type} via GenericOptimizer")
            return result
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"GenericOptimizer creation failed: {e}")
        
        return None


class SpecializedOptimizerStrategy(OptimizerCreationStrategy):
    """Strategy for creating specialized optimizers (quantum, kv_cache, etc.)."""
    
    def __init__(self, specialized_types: Dict[str, bool]):
        """
        Initialize specialized optimizer strategy.
        
        Args:
            specialized_types: Dictionary mapping specialized types to availability
        """
        self._specialized_types = specialized_types
    
    @property
    def name(self) -> str:
        return "specialized"
    
    def create(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """Create optimizer via specialized optimizers."""
        # Check for quantum optimizer
        if kwargs.get('use_quantum', False) and self._specialized_types.get('quantum', False):
            kwargs.pop('use_quantum', None)
            return self._create_quantum(optimizer_type, learning_rate, **kwargs)
        
        # Check for kv_cache optimizer
        if kwargs.get('use_kv_cache', False) and self._specialized_types.get('kv_cache', False):
            kwargs.pop('use_kv_cache', None)
            return self._create_kv_cache(optimizer_type, learning_rate, **kwargs)
        
        return None
    
    def _create_quantum(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """Create quantum optimizer."""
        try:
            from optimization_core.optimizers.quantum.quantum_truthgpt_optimizer import (
                QuantumTruthGPTOptimizer
            )
            result = QuantumTruthGPTOptimizer(
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                **kwargs
            )
            logger.debug(f"✅ Created {optimizer_type} via QuantumOptimizer")
            return result
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Quantum optimizer creation failed: {e}")
        
        return None
    
    def _create_kv_cache(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """Create KV cache optimizer."""
        try:
            from optimization_core.optimizers.kv_cache.kv_cache_optimizer import (
                KVCacheOptimizer
            )
            result = KVCacheOptimizer(
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                **kwargs
            )
            logger.debug(f"✅ Created {optimizer_type} via KVCacheOptimizer")
            return result
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"KV cache optimizer creation failed: {e}")
        
        return None


class OptimizerCreationStrategyChain:
    """
    Chain of responsibility for optimizer creation.
    Tries strategies in order until one succeeds.
    """
    
    def __init__(self, strategies: list[OptimizerCreationStrategy]):
        """
        Initialize strategy chain.
        
        Args:
            strategies: List of strategies to try in order
        """
        self._strategies = strategies
    
    def create(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """
        Try each strategy until one succeeds.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            **kwargs: Additional parameters
        
        Returns:
            Optimizer instance or None if all strategies fail
        """
        for strategy in self._strategies:
            result = strategy.create(optimizer_type, learning_rate, **kwargs)
            if result is not None:
                return result
        
        logger.warning(f"⚠️ Could not create {optimizer_type} from optimization_core")
        return None

