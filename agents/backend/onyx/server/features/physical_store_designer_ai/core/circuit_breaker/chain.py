"""
Circuit Breaker Chain

Chain of circuit breakers for sequential operations.
"""

from typing import Callable, List, Any, Dict
import asyncio

from .breaker import CircuitBreaker
from .circuit_types import CircuitBreakerEventType

# Import ServiceError
try:
    from ...exceptions import ServiceError
except ImportError:
    class ServiceError(Exception):
        def __init__(self, service: str, message: str, details: Dict = None):
            self.service = service
            self.message = message
            self.details = details or {}
            super().__init__(f"{service}: {message}")


class CircuitBreakerChain:
    """
    Chain of circuit breakers for sequential operations.
    
    Useful when you need to call multiple services in sequence,
    where failure in any step should stop the chain.
    """
    
    def __init__(self, *breakers: CircuitBreaker):
        """
        Initialize circuit breaker chain.
        
        Args:
            *breakers: Circuit breakers in order
        """
        self.breakers = list(breakers)
        self._lock = asyncio.Lock()
    
    async def call(
        self,
        funcs: List[Callable],
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Execute functions through chain of circuit breakers.
        
        Args:
            funcs: List of functions to call (one per breaker)
            *args: Common arguments for all functions
            **kwargs: Common keyword arguments for all functions
            
        Returns:
            List of results from each function
            
        Raises:
            ServiceError: If any circuit breaker rejects the call
        """
        if len(funcs) != len(self.breakers):
            raise ValueError(
                f"Number of functions ({len(funcs)}) must match "
                f"number of breakers ({len(self.breakers)})"
            )
        
        results = []
        current_args = args
        current_kwargs = kwargs
        
        for i, (breaker, func) in enumerate(zip(self.breakers, funcs)):
            try:
                # Check if breaker is ready
                if not breaker.is_ready():
                    raise ServiceError(
                        service=breaker.name,
                        message=f"Circuit breaker {breaker.name} in chain is not ready",
                        details={"position": i, "state": breaker.state.value}
                    )
                
                # Call through circuit breaker
                result = await breaker.call(func, *current_args, **current_kwargs)
                results.append(result)
                
                # Use result as input for next function if it's a tuple/dict
                if isinstance(result, tuple) and len(result) > 0:
                    current_args = result
                elif isinstance(result, dict):
                    current_kwargs.update(result)
                
            except Exception as e:
                # Emit event for chain failure
                for cb in self.breakers:
                    cb._emit_event(
                        CircuitBreakerEventType.FAILURE_RECORDED,
                        error_type="ChainFailure",
                        error_message=str(e),
                        position=i,
                        chain_length=len(self.breakers)
                    )
                raise
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all breakers in chain"""
        return {
            "chain_length": len(self.breakers),
            "all_ready": all(b.is_ready() for b in self.breakers),
            "all_healthy": all(b.is_healthy() for b in self.breakers),
            "breakers": [
                {
                    "name": b.name,
                    "state": b.state.value,
                    "healthy": b.is_healthy(),
                    "ready": b.is_ready()
                }
                for b in self.breakers
            ]
        }




