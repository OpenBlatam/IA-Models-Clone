"""
Advanced Error Recovery - Recuperación Avanzada de Errores
==========================================================

Recuperación avanzada de errores:
- Automatic retry with backoff
- Circuit breaker integration
- Fallback strategies
- Error classification
- Recovery strategies
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Tipos de error"""
    TRANSIENT = "transient"  # Temporal, puede reintentarse
    PERMANENT = "permanent"  # Permanente, no reintentar
    TIMEOUT = "timeout"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"


class RecoveryStrategy(str, Enum):
    """Estrategias de recuperación"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEGRADE = "degrade"
    SKIP = "skip"


class ErrorRecoveryManager:
    """
    Gestor de recuperación de errores.
    """
    
    def __init__(self) -> None:
        self.error_classifiers: List[Callable[[Exception], ErrorType]] = []
        self.recovery_strategies: Dict[ErrorType, RecoveryStrategy] = {
            ErrorType.TRANSIENT: RecoveryStrategy.RETRY,
            ErrorType.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorType.NETWORK: RecoveryStrategy.RETRY,
            ErrorType.RATE_LIMIT: RecoveryStrategy.RETRY,
            ErrorType.PERMANENT: RecoveryStrategy.FALLBACK,
            ErrorType.SERVER_ERROR: RecoveryStrategy.CIRCUIT_BREAKER
        }
        self.fallback_handlers: Dict[str, Callable] = {}
        self.retry_configs: Dict[str, Dict[str, Any]] = {}
        self.error_history: List[Dict[str, Any]] = []
    
    def register_error_classifier(
        self,
        classifier: Callable[[Exception], ErrorType]
    ) -> None:
        """Registra clasificador de errores"""
        self.error_classifiers.append(classifier)
    
    def register_fallback(
        self,
        operation: str,
        fallback_handler: Callable
    ) -> None:
        """Registra handler de fallback"""
        self.fallback_handlers[operation] = fallback_handler
        logger.info(f"Fallback registered for {operation}")
    
    def classify_error(self, error: Exception) -> ErrorType:
        """Clasifica error"""
        # Usar clasificadores registrados
        for classifier in self.error_classifiers:
            error_type = classifier(error)
            if error_type:
                return error_type
        
        # Clasificación por defecto
        error_str = str(error).lower()
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "500" in error_str or "server error" in error_str:
            return ErrorType.SERVER_ERROR
        else:
            return ErrorType.TRANSIENT
    
    async def execute_with_recovery(
        self,
        operation: str,
        func: Callable,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Ejecuta operación con recuperación"""
        max_retries = self.retry_configs.get(operation, {}).get("max_retries", 3)
        backoff_factor = self.retry_configs.get(operation, {}).get("backoff_factor", 1.0)
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Éxito
                return result
                
            except Exception as e:
                last_error = e
                error_type = self.classify_error(e)
                strategy = self.recovery_strategies.get(error_type, RecoveryStrategy.FALLBACK)
                
                # Registrar error
                self.error_history.append({
                    "operation": operation,
                    "error": str(e),
                    "error_type": error_type.value,
                    "strategy": strategy.value,
                    "attempt": attempt + 1,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Aplicar estrategia
                if strategy == RecoveryStrategy.RETRY and attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Retrying {operation} after {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                elif strategy == RecoveryStrategy.FALLBACK:
                    if operation in self.fallback_handlers:
                        logger.info(f"Using fallback for {operation}")
                        fallback = self.fallback_handlers[operation]
                        if asyncio.iscoroutinefunction(fallback):
                            return await fallback(*args, **kwargs)
                        else:
                            return fallback(*args, **kwargs)
                    else:
                        logger.error(f"No fallback handler for {operation}")
                        raise
                elif strategy == RecoveryStrategy.SKIP:
                    logger.warning(f"Skipping {operation} due to {error_type.value} error")
                    return None
                else:
                    # No más intentos o estrategia no soportada
                    raise
        
        # Si llegamos aquí, todos los reintentos fallaron
        raise last_error
    
    def get_error_history(
        self,
        operation: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Obtiene historial de errores"""
        history = self.error_history
        
        if operation:
            history = [e for e in history if e["operation"] == operation]
        
        if start_date:
            history = [
                e for e in history
                if datetime.fromisoformat(e["timestamp"]) >= start_date
            ]
        
        return history


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Obtiene gestor de recuperación de errores"""
    return ErrorRecoveryManager()















