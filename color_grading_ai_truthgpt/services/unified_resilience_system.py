"""
Unified Resilience System for Color Grading AI
==============================================

Consolidates resilience services:
- CircuitBreaker (circuit breaker)
- RetryManager (retry logic)
- LoadBalancer (load balancing)
- ErrorRecoverySystem (error recovery)

Features:
- Unified resilience interface
- Circuit breaker protection
- Automatic retries
- Load balancing
- Error recovery
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from .retry_manager import RetryManager, RetryConfig
from .load_balancer import LoadBalancer, LoadBalanceStrategy, Worker
from .error_recovery import ErrorRecoverySystem, RecoveryStrategy

logger = logging.getLogger(__name__)


class ResilienceMode(Enum):
    """Resilience modes."""
    BASIC = "basic"  # Basic resilience
    STANDARD = "standard"  # Standard resilience
    AGGRESSIVE = "aggressive"  # Aggressive resilience
    FULL = "full"  # Full resilience with all features


@dataclass
class ResilienceResult:
    """Resilience operation result."""
    success: bool
    attempts: int = 0
    strategy_used: str = ""
    error: Optional[str] = None
    recovery_applied: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedResilienceSystem:
    """
    Unified resilience system.
    
    Consolidates:
    - CircuitBreaker: Circuit breaker protection
    - RetryManager: Retry logic
    - LoadBalancer: Load balancing
    - ErrorRecoverySystem: Error recovery
    
    Features:
    - Unified resilience interface
    - Multi-layer protection
    - Automatic recovery
    - Load distribution
    """
    
    def __init__(
        self,
        resilience_mode: ResilienceMode = ResilienceMode.STANDARD
    ):
        """
        Initialize unified resilience system.
        
        Args:
            resilience_mode: Resilience mode
        """
        self.resilience_mode = resilience_mode
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker("default")
        self.retry_manager = RetryManager()
        self.load_balancer = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOAD)
        self.error_recovery = ErrorRecoverySystem()
        
        logger.info(f"Initialized UnifiedResilienceSystem (mode={resilience_mode.value})")
    
    async def execute_with_resilience(
        self,
        operation: Callable,
        operation_name: str = "operation",
        *args,
        **kwargs
    ) -> ResilienceResult:
        """
        Execute operation with full resilience.
        
        Args:
            operation: Operation to execute
            operation_name: Operation name
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Resilience result
        """
        attempts = 0
        max_attempts = 3 if self.resilience_mode == ResilienceMode.BASIC else 5
        
        while attempts < max_attempts:
            attempts += 1
            
            try:
                # Check circuit breaker
                if self.circuit_breaker.state == CircuitState.OPEN:
                    # Try recovery
                    recovery_result = await self.error_recovery.recover(
                        Exception("Circuit breaker open"),
                        operation_name
                    )
                    
                    if not recovery_result.success:
                        return ResilienceResult(
                            success=False,
                            attempts=attempts,
                            strategy_used="circuit_breaker",
                            error="Circuit breaker is open and recovery failed"
                        )
                
                # Execute with circuit breaker protection
                result = await self.circuit_breaker.call(operation, *args, **kwargs)
                
                return ResilienceResult(
                    success=True,
                    attempts=attempts,
                    strategy_used="circuit_breaker",
                    recovery_applied=False
                )
            
            except Exception as e:
                # Try error recovery
                recovery_result = await self.error_recovery.recover(e, operation_name)
                
                if recovery_result.success:
                    return ResilienceResult(
                        success=True,
                        attempts=attempts,
                        strategy_used=recovery_result.strategy.value,
                        recovery_applied=True
                    )
                
                # Use retry manager for retry strategy
                if attempts < max_attempts:
                    retry_config = RetryConfig(
                        max_attempts=1,
                        backoff_factor=2.0
                    )
                    await asyncio.sleep(retry_config.backoff_factor ** (attempts - 1))
                    continue
                
                return ResilienceResult(
                    success=False,
                    attempts=attempts,
                    strategy_used="retry",
                    error=str(e),
                    recovery_applied=False
                )
        
        return ResilienceResult(
            success=False,
            attempts=attempts,
            error="Max attempts reached"
        )
    
    def get_worker(self, service_name: str) -> Optional[Worker]:
        """
        Get a worker from load balancer.
        
        Args:
            service_name: Service name
            
        Returns:
            Worker or None
        """
        return self.load_balancer.get_worker(service_name)
    
    def register_worker(
        self,
        service_name: str,
        worker: Worker
    ):
        """
        Register a worker with load balancer.
        
        Args:
            service_name: Service name
            worker: Worker instance
        """
        self.load_balancer.add_worker(service_name, worker)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        return {
            "resilience_mode": self.resilience_mode.value,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failures": self.circuit_breaker.failure_count,
            "load_balancer_workers": len(self.load_balancer._workers),
            "error_recovery_stats": self.error_recovery.get_error_statistics(),
        }


