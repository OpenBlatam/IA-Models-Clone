"""
Smart Retry Manager - Gestor Inteligente de Reintentos
=======================================================

Sistema inteligente de gestión de reintentos con estrategias adaptativas y aprendizaje automático.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import random
import statistics

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Estrategia de reintento."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


class RetryStatus(Enum):
    """Estado de reintento."""
    PENDING = "pending"
    RETRYING = "retrying"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetryConfig:
    """Configuración de reintento."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryAttempt:
    """Intento de reintento."""
    attempt_number: int
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    response_time: Optional[float] = None
    delay_before: float = 0.0


@dataclass
class RetryOperation:
    """Operación con reintentos."""
    operation_id: str
    operation_type: str
    config: RetryConfig
    status: RetryStatus = RetryStatus.PENDING
    attempts: List[RetryAttempt] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartRetryManager:
    """Gestor inteligente de reintentos."""
    
    def __init__(self):
        self.operations: Dict[str, RetryOperation] = {}
        self.operation_history: deque = deque(maxlen=100000)
        self.success_patterns: Dict[str, Dict[str, Any]] = {}
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def create_retry_operation(
        self,
        operation_id: str,
        operation_type: str,
        config: Optional[RetryConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear operación con reintentos."""
        retry_config = config or RetryConfig()
        
        operation = RetryOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            config=retry_config,
            metadata=metadata or {},
        )
        
        async def save_operation():
            async with self._lock:
                self.operations[operation_id] = operation
        
        asyncio.create_task(save_operation())
        
        logger.info(f"Created retry operation: {operation_id}")
        return operation_id
    
    async def execute_with_retry(
        self,
        operation_id: str,
        operation_func: Callable[[], Awaitable[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Ejecutar operación con reintentos inteligentes."""
        operation = self.operations.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")
        
        operation.status = RetryStatus.RETRYING
        
        last_error = None
        result = None
        
        for attempt_num in range(1, operation.config.max_attempts + 1):
            # Calcular delay antes del intento
            if attempt_num > 1:
                delay = self._calculate_delay(
                    attempt_num,
                    operation.config,
                    operation_type=operation.operation_type,
                )
                await asyncio.sleep(delay)
            
            start_time = datetime.now()
            success = False
            error = None
            
            try:
                result = await operation_func()
                success = True
            except Exception as e:
                error = str(e)
                last_error = error
                
                # Verificar si es error reintentable
                if not self._is_retryable_error(error, operation.config):
                    logger.info(f"Non-retryable error for {operation_id}: {error}")
                    break
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Registrar intento
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                timestamp=datetime.now(),
                success=success,
                error=error,
                response_time=response_time,
                delay_before=delay if attempt_num > 1 else 0.0,
            )
            
            async with self._lock:
                operation.attempts.append(attempt)
            
            if success:
                operation.status = RetryStatus.SUCCESS
                operation.completed_at = datetime.now()
                
                # Aprender del éxito
                asyncio.create_task(self._learn_from_success(operation))
                
                logger.info(f"Operation {operation_id} succeeded on attempt {attempt_num}")
                return True, result, None
            else:
                logger.warning(f"Operation {operation_id} failed on attempt {attempt_num}: {error}")
        
        # Todos los intentos fallaron
        operation.status = RetryStatus.FAILED
        operation.completed_at = datetime.now()
        
        # Aprender del fracaso
        asyncio.create_task(self._learn_from_failure(operation))
        
        async with self._lock:
            self.operation_history.append(operation)
        
        return False, None, last_error
    
    def _calculate_delay(
        self,
        attempt_num: int,
        config: RetryConfig,
        operation_type: Optional[str] = None,
    ) -> float:
        """Calcular delay antes del siguiente intento."""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.initial_delay * (config.backoff_multiplier ** (attempt_num - 2))
        
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.initial_delay * (attempt_num - 1)
        
        elif config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = config.initial_delay
        
        elif config.strategy == RetryStrategy.ADAPTIVE:
            # Usar aprendizaje para determinar mejor delay
            delay = self._adaptive_delay(attempt_num, config, operation_type)
        
        else:
            delay = config.initial_delay
        
        # Aplicar jitter aleatorio (±20%)
        jitter = delay * 0.2 * (random.random() * 2 - 1)
        delay = delay + jitter
        
        # Limitar a max_delay
        delay = min(delay, config.max_delay)
        
        return max(0.0, delay)
    
    def _adaptive_delay(
        self,
        attempt_num: int,
        config: RetryConfig,
        operation_type: Optional[str] = None,
    ) -> float:
        """Calcular delay adaptativo basado en aprendizaje."""
        # Buscar patrones de éxito para esta operación
        if operation_type and operation_type in self.success_patterns:
            pattern = self.success_patterns[operation_type]
            avg_delay = pattern.get("avg_delay_before_success", config.initial_delay)
            return avg_delay * (config.backoff_multiplier ** (attempt_num - 2))
        
        # Fallback a exponential backoff
        return config.initial_delay * (config.backoff_multiplier ** (attempt_num - 2))
    
    def _is_retryable_error(self, error: str, config: RetryConfig) -> bool:
        """Verificar si error es reintentable."""
        if not config.retryable_errors:
            return True  # Todos los errores son reintentables por defecto
        
        # Verificar si el error coincide con algún patrón retryable
        error_lower = error.lower()
        for retryable in config.retryable_errors:
            if retryable.lower() in error_lower:
                return True
        
        return False
    
    async def _learn_from_success(self, operation: RetryOperation):
        """Aprender de operación exitosa."""
        if not operation.attempts:
            return
        
        successful_attempt = operation.attempts[-1]
        
        async with self._lock:
            if operation.operation_type not in self.success_patterns:
                self.success_patterns[operation.operation_type] = {
                    "total_successes": 0,
                    "avg_attempts": 0.0,
                    "avg_delay_before_success": 0.0,
                    "avg_response_time": 0.0,
                }
            
            pattern = self.success_patterns[operation.operation_type]
            pattern["total_successes"] += 1
            
            # Actualizar promedios
            attempts_count = len(operation.attempts)
            pattern["avg_attempts"] = (
                (pattern["avg_attempts"] * (pattern["total_successes"] - 1) + attempts_count) /
                pattern["total_successes"]
            )
            
            total_delay = sum(att.delay_before for att in operation.attempts)
            pattern["avg_delay_before_success"] = (
                (pattern["avg_delay_before_success"] * (pattern["total_successes"] - 1) + total_delay) /
                pattern["total_successes"]
            )
            
            if successful_attempt.response_time:
                pattern["avg_response_time"] = (
                    (pattern["avg_response_time"] * (pattern["total_successes"] - 1) + successful_attempt.response_time) /
                    pattern["total_successes"]
                )
    
    async def _learn_from_failure(self, operation: RetryOperation):
        """Aprender de operación fallida."""
        if not operation.attempts:
            return
        
        last_attempt = operation.attempts[-1]
        
        async with self._lock:
            if operation.operation_type not in self.failure_patterns:
                self.failure_patterns[operation.operation_type] = {
                    "total_failures": 0,
                    "common_errors": defaultdict(int),
                    "avg_attempts_before_failure": 0.0,
                }
            
            pattern = self.failure_patterns[operation.operation_type]
            pattern["total_failures"] += 1
            
            if last_attempt.error:
                pattern["common_errors"][last_attempt.error] += 1
            
            attempts_count = len(operation.attempts)
            pattern["avg_attempts_before_failure"] = (
                (pattern["avg_attempts_before_failure"] * (pattern["total_failures"] - 1) + attempts_count) /
                pattern["total_failures"]
            )
    
    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de operación."""
        operation = self.operations.get(operation_id)
        if not operation:
            return None
        
        return {
            "operation_id": operation.operation_id,
            "operation_type": operation.operation_type,
            "status": operation.status.value,
            "config": {
                "max_attempts": operation.config.max_attempts,
                "strategy": operation.config.strategy.value,
                "initial_delay": operation.config.initial_delay,
            },
            "attempts": [
                {
                    "attempt_number": att.attempt_number,
                    "success": att.success,
                    "error": att.error,
                    "response_time": att.response_time,
                    "timestamp": att.timestamp.isoformat(),
                }
                for att in operation.attempts
            ],
            "created_at": operation.created_at.isoformat(),
            "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
        }
    
    def get_learning_patterns(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Obtener patrones aprendidos."""
        if operation_type:
            return {
                "success_patterns": self.success_patterns.get(operation_type, {}),
                "failure_patterns": self.failure_patterns.get(operation_type, {}),
            }
        
        return {
            "success_patterns": self.success_patterns,
            "failure_patterns": self.failure_patterns,
        }
    
    def get_smart_retry_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for operation in self.operations.values():
            by_status[operation.status.value] += 1
        
        return {
            "total_operations": len(self.operations),
            "operations_by_status": dict(by_status),
            "total_history": len(self.operation_history),
            "learned_patterns": {
                "operation_types": len(self.success_patterns),
                "success_patterns": sum(p.get("total_successes", 0) for p in self.success_patterns.values()),
                "failure_patterns": sum(p.get("total_failures", 0) for p in self.failure_patterns.values()),
            },
        }

