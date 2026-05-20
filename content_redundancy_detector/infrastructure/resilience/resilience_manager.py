"""
Resilience Manager - Unified resilience orchestration
Production-ready resilience management system
"""

import asyncio
import logging
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
from .retry import retry_async, RetryPolicy, RetryStrategy
from .fallback import FallbackHandler, FallbackConfig, FallbackStrategy
from .timeout import TimeoutHandler, TimeoutError
from .bulkhead import Bulkhead, BulkheadConfig, BulkheadFullError, BulkheadIsolatedError
from .error_recovery import ErrorRecoverySystem, RecoveryStrategy, RecoveryAction

logger = logging.getLogger(__name__)

@dataclass
class ResilienceConfig:
    """Unified resilience configuration"""
    # Circuit breaker
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    
    # Retry
    retry_policy: Optional[RetryPolicy] = None
    
    # Fallback
    fallback_config: Optional[FallbackConfig] = None
    
    # Timeout
    timeout: Optional[float] = None
    
    # Bulkhead
    bulkhead_config: Optional[BulkheadConfig] = None
    
    # Error recovery
    enable_recovery: bool = True

class ResilienceManager:
    """Unified resilience management system"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.timeout_handlers: Dict[str, TimeoutHandler] = {}
        self.recovery_system = ErrorRecoverySystem()
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

    def register_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig = None
    ) -> CircuitBreaker:
        """Register a circuit breaker"""
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker

    def register_bulkhead(
        self,
        name: str,
        config: BulkheadConfig = None
    ) -> Bulkhead:
        """Register a bulkhead"""
        bulkhead = Bulkhead(name, config)
        self.bulkheads[name] = bulkhead
        return bulkhead

    def register_timeout_handler(
        self,
        name: str,
        timeout: float = 30.0
    ) -> TimeoutHandler:
        """Register a timeout handler"""
        handler = TimeoutHandler(timeout)
        self.timeout_handlers[name] = handler
        return handler

    async def execute_with_resilience(
        self,
        func: Callable,
        resilience_config: ResilienceConfig,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with full resilience protection"""
        self.total_executions += 1
        
        # Wrap function with resilience layers (inside-out)
        protected_func = func
        
        # 1. Timeout protection
        if resilience_config.timeout:
            timeout_handler = TimeoutHandler(resilience_config.timeout)
            original_func = protected_func
            protected_func = lambda *a, **kw: timeout_handler.with_timeout(
                original_func, resilience_config.timeout, *a, **kw
            )
        
        # 2. Bulkhead protection
        if resilience_config.bulkhead_config:
            bulkhead_name = kwargs.get("bulkhead_name", "default")
            if bulkhead_name not in self.bulkheads:
                self.register_bulkhead(bulkhead_name, resilience_config.bulkhead_config)
            bulkhead = self.bulkheads[bulkhead_name]
            original_func = protected_func
            protected_func = lambda *a, **kw: bulkhead.execute(original_func, *a, **kw)
        
        # 3. Retry
        if resilience_config.retry_policy:
            original_func = protected_func
            protected_func = lambda *a, **kw: retry_async(
                original_func,
                *a,
                policy=resilience_config.retry_policy,
                **kw
            )
        
        # 4. Circuit breaker
        if resilience_config.circuit_breaker:
            circuit_name = kwargs.get("circuit_name", "default")
            if circuit_name not in self.circuit_breakers:
                self.register_circuit_breaker(circuit_name, resilience_config.circuit_breaker)
            breaker = self.circuit_breakers[circuit_name]
            
            try:
                result = await breaker.call(protected_func, *args, **kwargs)
                self.successful_executions += 1
                return result
            except CircuitBreakerOpenError:
                # Try fallback if circuit is open
                if resilience_config.fallback_config:
                    fallback = FallbackHandler(resilience_config.fallback_config)
                    result = await fallback.execute_with_fallback(protected_func, *args, **kwargs)
                    self.successful_executions += 1
                    return result
                raise
        
        # 5. Fallback (if no circuit breaker)
        if resilience_config.fallback_config:
            fallback = FallbackHandler(resilience_config.fallback_config)
            protected_func = lambda *a, **kw: fallback.execute_with_fallback(
                protected_func, *a, **kw
            )
        
        # Execute protected function
        try:
            if asyncio.iscoroutinefunction(protected_func):
                result = await protected_func(*args, **kwargs)
            else:
                result = protected_func(*args, **kwargs)
            
            self.successful_executions += 1
            return result
            
        except Exception as e:
            self.failed_executions += 1
            
            # Attempt error recovery
            if resilience_config.enable_recovery:
                try:
                    recovery_result = await self.recovery_system.recover_from_error(
                        e, {"function": str(func), "args": str(args), "kwargs": str(kwargs)}
                    )
                    if recovery_result is not None:
                        self.successful_executions += 1
                        return recovery_result
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
            
            raise

    def get_resilience_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resilience statistics"""
        stats = {
            "executions": {
                "total": self.total_executions,
                "successful": self.successful_executions,
                "failed": self.failed_executions,
                "success_rate": (
                    self.successful_executions / max(self.total_executions, 1)
                )
            },
            "circuit_breakers": {
                name: breaker.get_stats()
                for name, breaker in self.circuit_breakers.items()
            },
            "bulkheads": {
                name: bulkhead.get_stats()
                for name, bulkhead in self.bulkheads.items()
            },
            "timeout_handlers": {
                name: handler.get_stats()
                for name, handler in self.timeout_handlers.items()
            },
            "error_recovery": self.recovery_system.get_recovery_statistics()
        }
        
        return stats

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall resilience health status"""
        # Check circuit breakers
        open_circuits = [
            name for name, breaker in self.circuit_breakers.items()
            if breaker.get_state().value == "open"
        ]
        
        # Check bulkheads
        full_bulkheads = [
            name for name, bulkhead in self.bulkheads.items()
            if bulkhead.get_state().value == "full"
        ]
        
        # Overall status
        if open_circuits or full_bulkheads:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "open_circuit_breakers": open_circuits,
            "full_bulkheads": full_bulkheads,
            "total_circuit_breakers": len(self.circuit_breakers),
            "total_bulkheads": len(self.bulkheads),
            "resilience_stats": self.get_resilience_statistics()
        }

# Global resilience manager instance
_global_resilience_manager: Optional[ResilienceManager] = None

def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance"""
    global _global_resilience_manager
    if _global_resilience_manager is None:
        _global_resilience_manager = ResilienceManager()
    return _global_resilience_manager






