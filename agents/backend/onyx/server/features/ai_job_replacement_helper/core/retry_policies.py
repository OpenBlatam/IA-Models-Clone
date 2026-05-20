"""
Retry Policies Service - Políticas de reintento
================================================

Sistema de políticas de reintento avanzadas.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Estrategias de reintento"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CUSTOM = "custom"


@dataclass
class RetryPolicy:
    """Política de reintento"""
    name: str
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[type] = field(default_factory=list)


@dataclass
class RetryResult:
    """Resultado de reintento"""
    success: bool
    attempts: int
    total_time_seconds: float
    last_exception: Optional[Exception] = None
    result: Any = None


class RetryPoliciesService:
    """Servicio de políticas de reintento"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.policies: Dict[str, RetryPolicy] = {}
        logger.info("RetryPoliciesService initialized")
    
    def create_policy(
        self,
        name: str,
        max_attempts: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True
    ) -> RetryPolicy:
        """Crear política de reintento"""
        policy = RetryPolicy(
            name=name,
            max_attempts=max_attempts,
            strategy=strategy,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            backoff_multiplier=backoff_multiplier,
            jitter=jitter,
        )
        
        self.policies[name] = policy
        
        logger.info(f"Retry policy created: {name}")
        return policy
    
    async def execute_with_retry(
        self,
        policy_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> RetryResult:
        """Ejecutar función con política de reintento"""
        import time
        import random
        
        policy = self.policies.get(policy_name)
        if not policy:
            policy = self.create_policy(policy_name)
        
        start_time = time.time()
        last_exception = None
        
        for attempt in range(1, policy.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                total_time = time.time() - start_time
                
                return RetryResult(
                    success=True,
                    attempts=attempt,
                    total_time_seconds=total_time,
                    result=result,
                )
            
            except Exception as e:
                last_exception = e
                
                # Verificar si es excepción retryable
                if policy.retryable_exceptions:
                    if not isinstance(e, tuple(policy.retryable_exceptions)):
                        # No es retryable
                        break
                
                # Si es el último intento, no esperar
                if attempt >= policy.max_attempts:
                    break
                
                # Calcular delay
                delay = self._calculate_delay(policy, attempt)
                
                # Aplicar jitter si está habilitado
                if policy.jitter:
                    jitter_amount = delay * 0.1
                    delay += random.uniform(-jitter_amount, jitter_amount)
                    delay = max(0, delay)
                
                logger.warning(f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        total_time = time.time() - start_time
        
        return RetryResult(
            success=False,
            attempts=policy.max_attempts,
            total_time_seconds=total_time,
            last_exception=last_exception,
        )
    
    def _calculate_delay(self, policy: RetryPolicy, attempt: int) -> float:
        """Calcular delay según estrategia"""
        if policy.strategy == RetryStrategy.FIXED:
            return policy.base_delay_seconds
        
        elif policy.strategy == RetryStrategy.EXPONENTIAL:
            delay = policy.base_delay_seconds * (policy.backoff_multiplier ** (attempt - 1))
            return min(delay, policy.max_delay_seconds)
        
        elif policy.strategy == RetryStrategy.LINEAR:
            delay = policy.base_delay_seconds * attempt
            return min(delay, policy.max_delay_seconds)
        
        else:
            return policy.base_delay_seconds




