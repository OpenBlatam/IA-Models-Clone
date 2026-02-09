"""
Circuit Breaker Groups

Group management for multiple circuit breakers with shared configuration.
"""

from typing import Optional, Dict, Any
import asyncio

from .breaker import CircuitBreaker
from .config import CircuitBreakerConfig
from .circuit_types import CircuitState


class CircuitBreakerGroup:
    """
    Group of circuit breakers with shared configuration.
    
    Useful for managing multiple related circuit breakers together.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker group.
        
        Args:
            name: Name of the group
            config: Shared configuration for breakers in the group
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    def get_or_create(self, service_name: str, **overrides) -> CircuitBreaker:
        """
        Get or create circuit breaker in group.
        
        Args:
            service_name: Name of the service/circuit breaker
            **overrides: Configuration overrides for this specific breaker
            
        Returns:
            CircuitBreaker instance
        """
        if service_name not in self.breakers:
            # Merge config with overrides
            breaker_config = CircuitBreakerConfig(
                failure_threshold=overrides.get('failure_threshold', self.config.failure_threshold),
                recovery_timeout=overrides.get('recovery_timeout', self.config.recovery_timeout),
                expected_exception=overrides.get('expected_exception', self.config.expected_exception),
                success_threshold=overrides.get('success_threshold', self.config.success_threshold),
                monitoring_window=overrides.get('monitoring_window', self.config.monitoring_window),
                call_timeout=overrides.get('call_timeout', self.config.call_timeout),
                enable_adaptive_timeout=overrides.get('enable_adaptive_timeout', self.config.enable_adaptive_timeout),
                retry_enabled=overrides.get('retry_enabled', self.config.retry_enabled),
                fallback_enabled=overrides.get('fallback_enabled', self.config.fallback_enabled),
                half_open_max_concurrent=overrides.get('half_open_max_concurrent', self.config.half_open_max_concurrent),
            )
            
            self.breakers[service_name] = CircuitBreaker(
                config=breaker_config,
                name=f"{self.name}.{service_name}"
            )
        
        return self.breakers[service_name]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all breakers in group"""
        status = {
            "group_name": self.name,
            "total_breakers": len(self.breakers),
            "healthy_count": 0,
            "degraded_count": 0,
            "critical_count": 0,
            "open_count": 0,
            "breakers": {}
        }
        
        for name, breaker in self.breakers.items():
            health = breaker.get_health_status()
            status["breakers"][name] = health
            
            if breaker.state == CircuitState.OPEN:
                status["open_count"] += 1
            elif health["critical"]:
                status["critical_count"] += 1
            elif health["degraded"]:
                status["degraded_count"] += 1
            else:
                status["healthy_count"] += 1
        
        return status
    
    async def reset_all(self):
        """Reset all circuit breakers in group"""
        for breaker in self.breakers.values():
            await breaker.reset()




