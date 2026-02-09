"""
Retry Policies - Políticas avanzadas de reintento
==================================================
"""

import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Estrategias de reintento"""
    FIXED = "fixed"  # Intervalo fijo
    EXPONENTIAL = "exponential"  # Backoff exponencial
    LINEAR = "linear"  # Backoff lineal
    JITTER = "jitter"  # Con variación aleatoria


@dataclass
class RetryPolicy:
    """Política de reintento"""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0  # Segundos
    max_delay: float = 60.0  # Segundos máximo
    jitter: bool = True
    retryable_exceptions: List[type] = field(default_factory=lambda: [Exception])
    on_retry: Optional[Callable] = None
    on_failure: Optional[Callable] = None
    
    def calculate_delay(self, attempt: int) -> float:
        """Calcula el delay para un intento"""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = min(self.base_delay * attempt, self.max_delay)
        elif self.strategy == RetryStrategy.JITTER:
            base_delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
            delay = base_delay + random.uniform(0, base_delay * 0.1)
        else:
            delay = self.base_delay
        
        # Aplicar jitter si está habilitado
        if self.jitter and self.strategy != RetryStrategy.JITTER:
            delay = delay * (1 + random.uniform(-0.1, 0.1))
        
        return max(0, delay)


@dataclass
class RetryResult:
    """Resultado de un reintento"""
    success: bool
    attempts: int
    total_duration: float
    last_exception: Optional[Exception] = None
    result: Any = None


class RetryManager:
    """Gestor de políticas de reintento"""
    
    def __init__(self):
        self.policies: Dict[str, RetryPolicy] = {}
        self.default_policy = RetryPolicy()
    
    def register_policy(self, name: str, policy: RetryPolicy):
        """Registra una política de reintento"""
        self.policies[name] = policy
        logger.info(f"Política de reintento {name} registrada")
    
    def get_policy(self, name: Optional[str] = None) -> RetryPolicy:
        """Obtiene una política"""
        if name and name in self.policies:
            return self.policies[name]
        return self.default_policy
    
    async def execute_with_retry(
        self,
        func: Callable,
        policy: Optional[RetryPolicy] = None,
        *args,
        **kwargs
    ) -> RetryResult:
        """Ejecuta una función con reintentos"""
        if policy is None:
            policy = self.default_policy
        
        start_time = datetime.now()
        last_exception = None
        
        for attempt in range(1, policy.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Éxito
                total_duration = (datetime.now() - start_time).total_seconds()
                return RetryResult(
                    success=True,
                    attempts=attempt,
                    total_duration=total_duration,
                    result=result
                )
            
            except tuple(policy.retryable_exceptions) as e:
                last_exception = e
                
                # Si es el último intento, fallar
                if attempt >= policy.max_attempts:
                    total_duration = (datetime.now() - start_time).total_seconds()
                    if policy.on_failure:
                        policy.on_failure(attempt, e)
                    return RetryResult(
                        success=False,
                        attempts=attempt,
                        total_duration=total_duration,
                        last_exception=e
                    )
                
                # Calcular delay y esperar
                delay = policy.calculate_delay(attempt)
                logger.warning(
                    f"Intento {attempt}/{policy.max_attempts} falló: {e}. "
                    f"Reintentando en {delay:.2f}s"
                )
                
                if policy.on_retry:
                    policy.on_retry(attempt, e, delay)
                
                await asyncio.sleep(delay)
            
            except Exception as e:
                # Excepción no retryable
                total_duration = (datetime.now() - start_time).total_seconds()
                return RetryResult(
                    success=False,
                    attempts=attempt,
                    total_duration=total_duration,
                    last_exception=e
                )
        
        # No debería llegar aquí
        total_duration = (datetime.now() - start_time).total_seconds()
        return RetryResult(
            success=False,
            attempts=policy.max_attempts,
            total_duration=total_duration,
            last_exception=last_exception
        )
    
    def execute_sync_with_retry(
        self,
        func: Callable,
        policy: Optional[RetryPolicy] = None,
        *args,
        **kwargs
    ) -> RetryResult:
        """Ejecuta una función síncrona con reintentos"""
        if policy is None:
            policy = self.default_policy
        
        import time
        start_time = time.time()
        last_exception = None
        
        for attempt in range(1, policy.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                
                # Éxito
                total_duration = time.time() - start_time
                return RetryResult(
                    success=True,
                    attempts=attempt,
                    total_duration=total_duration,
                    result=result
                )
            
            except tuple(policy.retryable_exceptions) as e:
                last_exception = e
                
                if attempt >= policy.max_attempts:
                    total_duration = time.time() - start_time
                    if policy.on_failure:
                        policy.on_failure(attempt, e)
                    return RetryResult(
                        success=False,
                        attempts=attempt,
                        total_duration=total_duration,
                        last_exception=e
                    )
                
                delay = policy.calculate_delay(attempt)
                logger.warning(
                    f"Intento {attempt}/{policy.max_attempts} falló: {e}. "
                    f"Reintentando en {delay:.2f}s"
                )
                
                if policy.on_retry:
                    policy.on_retry(attempt, e, delay)
                
                time.sleep(delay)
            
            except Exception as e:
                total_duration = time.time() - start_time
                return RetryResult(
                    success=False,
                    attempts=attempt,
                    total_duration=total_duration,
                    last_exception=e
                )
        
        total_duration = time.time() - start_time
        return RetryResult(
            success=False,
            attempts=policy.max_attempts,
            total_duration=total_duration,
            last_exception=last_exception
        )




