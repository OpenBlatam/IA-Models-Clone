from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import logging
import traceback
import sys
import asyncio
import functools
import time
from typing import (
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import json
import hashlib
from datetime import datetime, timedelta
import weakref
from contextlib import contextmanager, asynccontextmanager
import threading
from collections import defaultdict, deque
import gc
import psutil
import os
        import re
from typing import Any, List, Dict, Optional
"""
🚨 COMPREHENSIVE ERROR HANDLING & EDGE CASE MANAGEMENT
=====================================================

Sistema robusto de manejo de errores y casos edge para el AI Video System.
Incluye categorización de errores, estrategias de recuperación, logging detallado
y monitoreo de errores en tiempo real.
"""

    Any, Callable, Dict, List, Optional, Union, Type, 
    Tuple, Set, Protocol, runtime_checkable
)

# =============================================================================
# CORE ERROR TYPES & CATEGORIES
# =============================================================================

class ErrorSeverity(Enum):
    """Niveles de severidad de errores."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    FATAL = auto()

class ErrorCategory(Enum):
    """Categorías de errores del sistema."""
    # Errores de sistema
    SYSTEM = auto()
    MEMORY = auto()
    DISK = auto()
    NETWORK = auto()
    PERMISSION = auto()
    
    # Errores de modelos AI
    MODEL_LOADING = auto()
    MODEL_INFERENCE = auto()
    MODEL_TRAINING = auto()
    MODEL_QUANTIZATION = auto()
    MODEL_MEMORY = auto()
    
    # Errores de datos
    DATA_LOADING = auto()
    DATA_VALIDATION = auto()
    DATA_TRANSFORMATION = auto()
    DATA_STORAGE = auto()
    
    # Errores de video
    VIDEO_PROCESSING = auto()
    VIDEO_ENCODING = auto()
    VIDEO_DECODING = auto()
    VIDEO_FORMAT = auto()
    VIDEO_MEMORY = auto()
    
    # Errores de API
    API_REQUEST = auto()
    API_RESPONSE = auto()
    API_RATE_LIMIT = auto()
    API_AUTHENTICATION = auto()
    
    # Errores de configuración
    CONFIGURATION = auto()
    ENVIRONMENT = auto()
    DEPENDENCY = auto()
    
    # Errores de concurrencia
    CONCURRENCY = auto()
    DEADLOCK = auto()
    RACE_CONDITION = auto()
    
    # Errores de seguridad
    SECURITY = auto()
    VALIDATION = auto()
    SANITIZATION = auto()

class ErrorContext(Enum):
    """Contextos donde pueden ocurrir errores."""
    INITIALIZATION = auto()
    RUNTIME = auto()
    CLEANUP = auto()
    SHUTDOWN = auto()
    BACKGROUND = auto()
    USER_INTERACTION = auto()
    BATCH_PROCESSING = auto()
    STREAMING = auto()

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class AIVideoError(Exception):
    """Excepción base para todos los errores del AI Video System."""
    
    def __init__(
        self, 
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_count: int = 0
    ):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.details = details or {}
        self.recoverable = recoverable
        self.retry_count = retry_count
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
        
    def __str__(self) -> Any:
        return f"[{self.category.name}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir error a diccionario para logging/monitoring."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.name,
            "severity": self.severity.name,
            "context": self.context.name if self.context else None,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback
        }

# Errores específicos del sistema
class SystemError(AIVideoError):
    """Errores del sistema operativo."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.SYSTEM, **kwargs)

class MemoryError(AIVideoError):
    """Errores de memoria."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.MEMORY, **kwargs)

class DiskError(AIVideoError):
    """Errores de disco."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.DISK, **kwargs)

class NetworkError(AIVideoError):
    """Errores de red."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.NETWORK, **kwargs)

# Errores de modelos AI
class ModelLoadingError(AIVideoError):
    """Errores al cargar modelos."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.MODEL_LOADING, **kwargs)

class ModelInferenceError(AIVideoError):
    """Errores durante inferencia."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.MODEL_INFERENCE, **kwargs)

class ModelTrainingError(AIVideoError):
    """Errores durante entrenamiento."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.MODEL_TRAINING, **kwargs)

class ModelMemoryError(AIVideoError):
    """Errores de memoria en modelos."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.MODEL_MEMORY, **kwargs)

# Errores de datos
class DataLoadingError(AIVideoError):
    """Errores al cargar datos."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.DATA_LOADING, **kwargs)

class DataValidationError(AIVideoError):
    """Errores de validación de datos."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.DATA_VALIDATION, **kwargs)

class DataTransformationError(AIVideoError):
    """Errores en transformación de datos."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.DATA_TRANSFORMATION, **kwargs)

# Errores de video
class VideoProcessingError(AIVideoError):
    """Errores en procesamiento de video."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.VIDEO_PROCESSING, **kwargs)

class VideoEncodingError(AIVideoError):
    """Errores en codificación de video."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.VIDEO_ENCODING, **kwargs)

class VideoFormatError(AIVideoError):
    """Errores de formato de video."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.VIDEO_FORMAT, **kwargs)

# Errores de API
class APIError(AIVideoError):
    """Errores de API."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.API_REQUEST, **kwargs)

class RateLimitError(APIError):
    """Errores de rate limiting."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.API_RATE_LIMIT, **kwargs)

# Errores de configuración
class ConfigurationError(AIVideoError):
    """Errores de configuración."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.CONFIGURATION, **kwargs)

class DependencyError(AIVideoError):
    """Errores de dependencias."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.DEPENDENCY, **kwargs)

# Errores de concurrencia
class ConcurrencyError(AIVideoError):
    """Errores de concurrencia."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.CONCURRENCY, **kwargs)

class DeadlockError(ConcurrencyError):
    """Errores de deadlock."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.DEADLOCK, **kwargs)

# Errores de seguridad
class SecurityError(AIVideoError):
    """Errores de seguridad."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.SECURITY, **kwargs)

class ValidationError(AIVideoError):
    """Errores de validación."""
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, ErrorCategory.VALIDATION, **kwargs)

# =============================================================================
# ERROR RECOVERY STRATEGIES
# =============================================================================

@dataclass
class RecoveryStrategy:
    """Estrategia de recuperación para un tipo de error."""
    name: str
    description: str
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    recovery_actions: List[Callable] = field(default_factory=list)
    cleanup_actions: List[Callable] = field(default_factory=list)
    
    def should_retry(self, error: AIVideoError) -> bool:
        """Determinar si se debe reintentar."""
        return (
            error.recoverable and 
            error.retry_count < self.max_retries and
            error.severity not in [ErrorSeverity.FATAL, ErrorSeverity.CRITICAL]
        )
    
    def get_retry_delay(self, retry_count: int) -> float:
        """Calcular delay para reintento con backoff exponencial."""
        return self.retry_delay * (self.backoff_multiplier ** retry_count)

class RecoveryManager:
    """Gestor de estrategias de recuperación."""
    
    def __init__(self) -> Any:
        self.strategies: Dict[ErrorCategory, RecoveryStrategy] = {}
        self._setup_default_strategies()
    
    def _setup_default_strategies(self) -> Any:
        """Configurar estrategias por defecto."""
        
        # Estrategia para errores de red
        self.strategies[ErrorCategory.NETWORK] = RecoveryStrategy(
            name="Network Retry",
            description="Reintento con backoff exponencial para errores de red",
            max_retries=5,
            retry_delay=1.0,
            backoff_multiplier=2.0,
            recovery_actions=[
                lambda: time.sleep(0.1),  # Pequeña pausa
                lambda: gc.collect(),      # Limpiar memoria
            ]
        )
        
        # Estrategia para errores de memoria
        self.strategies[ErrorCategory.MEMORY] = RecoveryStrategy(
            name="Memory Recovery",
            description="Recuperación de memoria con limpieza agresiva",
            max_retries=2,
            retry_delay=2.0,
            recovery_actions=[
                lambda: gc.collect(),
                lambda: self._clear_caches(),
                lambda: self._reduce_batch_size(),
            ]
        )
        
        # Estrategia para errores de modelo
        self.strategies[ErrorCategory.MODEL_INFERENCE] = RecoveryStrategy(
            name="Model Recovery",
            description="Recuperación de modelos con reload si es necesario",
            max_retries=3,
            retry_delay=1.0,
            recovery_actions=[
                lambda: self._clear_model_cache(),
                lambda: self._reset_model_state(),
            ]
        )
        
        # Estrategia para errores de video
        self.strategies[ErrorCategory.VIDEO_PROCESSING] = RecoveryStrategy(
            name="Video Recovery",
            description="Recuperación de procesamiento de video",
            max_retries=2,
            retry_delay=1.0,
            recovery_actions=[
                lambda: self._clear_video_cache(),
                lambda: self._reset_video_state(),
            ]
        )
    
    def _clear_caches(self) -> Any:
        """Limpiar caches del sistema."""
        # Implementar limpieza de caches
        pass
    
    def _reduce_batch_size(self) -> Any:
        """Reducir tamaño de batch para ahorrar memoria."""
        # Implementar reducción de batch size
        pass
    
    def _clear_model_cache(self) -> Any:
        """Limpiar cache de modelos."""
        # Implementar limpieza de cache de modelos
        pass
    
    def _reset_model_state(self) -> Any:
        """Resetear estado de modelos."""
        # Implementar reset de estado
        pass
    
    def _clear_video_cache(self) -> Any:
        """Limpiar cache de video."""
        # Implementar limpieza de cache de video
        pass
    
    def _reset_video_state(self) -> Any:
        """Resetear estado de video."""
        # Implementar reset de estado de video
        pass
    
    def get_strategy(self, error: AIVideoError) -> Optional[RecoveryStrategy]:
        """Obtener estrategia para un error."""
        return self.strategies.get(error.category)
    
    def register_strategy(self, category: ErrorCategory, strategy: RecoveryStrategy):
        """Registrar nueva estrategia."""
        self.strategies[category] = strategy

# =============================================================================
# ERROR MONITORING & TRACKING
# =============================================================================

@dataclass
class ErrorMetrics:
    """Métricas de errores."""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    recovery_success_rate: float = 0.0
    avg_recovery_time: float = 0.0
    last_error_time: Optional[datetime] = None
    error_trend: List[Tuple[datetime, int]] = field(default_factory=list)

class ErrorMonitor:
    """Monitor de errores en tiempo real."""
    
    def __init__(self, max_history: int = 1000):
        
    """__init__ function."""
self.max_history = max_history
        self.errors: deque = deque(maxlen=max_history)
        self.metrics = ErrorMetrics()
        self.recovery_manager = RecoveryManager()
        self.logger = logging.getLogger(__name__)
        
        # Configurar logging
        self._setup_logging()
    
    def _setup_logging(self) -> Any:
        """Configurar logging para errores."""
        error_handler = logging.FileHandler('error_logs.log')
        error_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def record_error(self, error: AIVideoError, recovery_success: bool = False, recovery_time: float = 0.0):
        """Registrar un error."""
        self.errors.append({
            'error': error,
            'timestamp': datetime.now(),
            'recovery_success': recovery_success,
            'recovery_time': recovery_time
        })
        
        # Actualizar métricas
        self.metrics.total_errors += 1
        self.metrics.errors_by_category[error.category] += 1
        self.metrics.errors_by_severity[error.severity] += 1
        self.metrics.last_error_time = datetime.now()
        
        # Actualizar tendencia
        self.metrics.error_trend.append((datetime.now(), 1))
        if len(self.metrics.error_trend) > 100:
            self.metrics.error_trend.pop(0)
        
        # Logging
        self.logger.error(f"Error registrado: {error.to_dict()}")
        
        # Alertas para errores críticos
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            self._send_alert(error)
    
    def _send_alert(self, error: AIVideoError):
        """Enviar alerta para errores críticos."""
        alert_message = f"🚨 ERROR CRÍTICO: {error.category.name} - {error.message}"
        self.logger.critical(alert_message)
        # Aquí se podría integrar con sistemas de alerta externos
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Obtener resumen de errores."""
        return {
            'total_errors': self.metrics.total_errors,
            'errors_by_category': {cat.name: count for cat, count in self.metrics.errors_by_category.items()},
            'errors_by_severity': {sev.name: count for sev, count in self.metrics.errors_by_severity.items()},
            'recovery_success_rate': self.metrics.recovery_success_rate,
            'avg_recovery_time': self.metrics.avg_recovery_time,
            'last_error_time': self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
            'recent_errors': [error['error'].to_dict() for error in list(self.errors)[-10:]]
        }
    
    def get_error_rate(self, minutes: int = 5) -> float:
        """Calcular tasa de errores en los últimos N minutos."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_errors = [
            error for error in self.errors 
            if error['timestamp'] > cutoff_time
        ]
        return len(recent_errors) / minutes  # errores por minuto

# =============================================================================
# ERROR HANDLING DECORATORS & CONTEXT MANAGERS
# =============================================================================

def handle_errors(
    error_types: Optional[List[Type[Exception]]] = None,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
):
    """Decorador para manejo automático de errores."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(f"Error en {func.__name__}: {e}")
                    logging.error(traceback.format_exc())
                
                if reraise:
                    raise
                
                return default_return
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(f"Error en {func.__name__}: {e}")
                    logging.error(traceback.format_exc())
                
                if reraise:
                    raise
                
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Optional[List[Type[Exception]]] = None
):
    """Decorador para reintento automático en errores."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if exceptions and not any(isinstance(e, exc) for exc in exceptions):
                        raise
                    
                    if attempt < max_retries:
                        sleep_time = delay * (backoff ** attempt)
                        logging.warning(f"Reintento {attempt + 1}/{max_retries} en {func.__name__} después de {sleep_time}s")
                        time.sleep(sleep_time)
                    else:
                        logging.error(f"Error final en {func.__name__} después de {max_retries} reintentos")
                        raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if exceptions and not any(isinstance(e, exc) for exc in exceptions):
                        raise
                    
                    if attempt < max_retries:
                        sleep_time = delay * (backoff ** attempt)
                        logging.warning(f"Reintento {attempt + 1}/{max_retries} en {func.__name__} después de {sleep_time}s")
                        await asyncio.sleep(sleep_time)
                    else:
                        logging.error(f"Error final en {func.__name__} después de {max_retries} reintentos")
                        raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

@contextmanager
def error_context(
    context_name: str,
    error_category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.ERROR
):
    """Context manager para manejo de errores con contexto."""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        error = AIVideoError(
            message=str(e),
            category=error_category,
            severity=severity,
            context=ErrorContext.RUNTIME,
            details={'context': context_name, 'duration': time.time() - start_time}
        )
        logging.error(f"Error en contexto '{context_name}': {error.to_dict()}")
        raise error

@asynccontextmanager
async def async_error_context(
    context_name: str,
    error_category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.ERROR
):
    """Context manager asíncrono para manejo de errores."""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        error = AIVideoError(
            message=str(e),
            category=error_category,
            severity=severity,
            context=ErrorContext.RUNTIME,
            details={'context': context_name, 'duration': time.time() - start_time}
        )
        logging.error(f"Error en contexto '{context_name}': {error.to_dict()}")
        raise error

# =============================================================================
# VALIDATION & SANITIZATION
# =============================================================================

class InputValidator:
    """Validador de entradas con manejo de errores."""
    
    @staticmethod
    def validate_file_path(path: str, must_exist: bool = True) -> Path:
        """Validar ruta de archivo."""
        try:
            file_path = Path(path)
            if must_exist and not file_path.exists():
                raise ValidationError(f"Archivo no encontrado: {path}")
            return file_path
        except Exception as e:
            raise ValidationError(f"Ruta de archivo inválida: {path}") from e
    
    @staticmethod
    def validate_video_format(file_path: Path) -> str:
        """Validar formato de video."""
        valid_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        suffix = file_path.suffix.lower()
        
        if suffix not in valid_formats:
            raise VideoFormatError(f"Formato de video no soportado: {suffix}")
        
        return suffix
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validar configuración de modelo."""
        required_fields = ['model_type', 'model_path']
        
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(f"Campo requerido faltante: {field}")
        
        return config
    
    @staticmethod
    def validate_memory_usage(required_mb: float) -> bool:
        """Validar uso de memoria disponible."""
        try:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < required_mb:
                raise MemoryError(f"Memoria insuficiente: {available_mb:.1f}MB disponible, {required_mb:.1f}MB requerido")
            return True
        except Exception as e:
            raise MemoryError(f"Error al verificar memoria: {e}") from e

class DataSanitizer:
    """Sanitizador de datos para prevenir errores de seguridad."""
    
    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitizar ruta de archivo."""
        # Remover caracteres peligrosos
        dangerous_chars = ['..', '~', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            path = path.replace(char, '')
        
        # Normalizar ruta
        return str(Path(path).resolve())
    
    @staticmethod
    def sanitize_model_name(name: str) -> str:
        """Sanitizar nombre de modelo."""
        # Solo permitir caracteres seguros
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        return safe_name[:50]  # Limitar longitud
    
    @staticmethod
    def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizar configuración."""
        # Remover campos peligrosos
        dangerous_keys = {'__class__', '__dict__', '__module__', 'exec', 'eval'}
        sanitized = {}
        
        for key, value in config.items():
            if key not in dangerous_keys:
                if isinstance(value, dict):
                    sanitized[key] = DataSanitizer.sanitize_config(value)
                elif isinstance(value, (str, int, float, bool, list)):
                    sanitized[key] = value
        
        return sanitized

# =============================================================================
# ERROR RECOVERY PIPELINE
# =============================================================================

class ErrorRecoveryPipeline:
    """Pipeline de recuperación de errores."""
    
    def __init__(self, monitor: ErrorMonitor):
        
    """__init__ function."""
self.monitor = monitor
        self.recovery_manager = monitor.recovery_manager
    
    async def handle_error(
        self, 
        error: AIVideoError, 
        operation: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """Manejar error con recuperación automática."""
        start_time = time.time()
        recovery_success = False
        
        try:
            # Obtener estrategia de recuperación
            strategy = self.recovery_manager.get_strategy(error)
            
            if not strategy or not strategy.should_retry(error):
                self.monitor.record_error(error, recovery_success, 0.0)
                raise error
            
            # Intentar recuperación
            for attempt in range(strategy.max_retries):
                try:
                    # Ejecutar acciones de recuperación
                    for action in strategy.recovery_actions:
                        if asyncio.iscoroutinefunction(action):
                            await action()
                        else:
                            action()
                    
                    # Esperar antes del reintento
                    delay = strategy.get_retry_delay(attempt)
                    await asyncio.sleep(delay)
                    
                    # Reintentar operación
                    if asyncio.iscoroutinefunction(operation):
                        result = await operation(*args, **kwargs)
                    else:
                        result = operation(*args, **kwargs)
                    
                    recovery_success = True
                    recovery_time = time.time() - start_time
                    
                    self.monitor.record_error(error, recovery_success, recovery_time)
                    return result
                    
                except Exception as retry_error:
                    error.retry_count += 1
                    logging.warning(f"Reintento {attempt + 1} falló: {retry_error}")
                    
                    # Ejecutar acciones de limpieza
                    for action in strategy.cleanup_actions:
                        try:
                            if asyncio.iscoroutinefunction(action):
                                await action()
                            else:
                                action()
                        except Exception as cleanup_error:
                            logging.error(f"Error en limpieza: {cleanup_error}")
            
            # Si llegamos aquí, todos los reintentos fallaron
            recovery_time = time.time() - start_time
            self.monitor.record_error(error, recovery_success, recovery_time)
            raise error
            
        except Exception as final_error:
            recovery_time = time.time() - start_time
            self.monitor.record_error(error, recovery_success, recovery_time)
            raise final_error

# =============================================================================
# GLOBAL ERROR HANDLER
# =============================================================================

class GlobalErrorHandler:
    """Manejador global de errores del sistema."""
    
    def __init__(self) -> Any:
        self.monitor = ErrorMonitor()
        self.recovery_pipeline = ErrorRecoveryPipeline(self.monitor)
        self.validator = InputValidator()
        self.sanitizer = DataSanitizer()
        
        # Configurar manejo de excepciones no capturadas
        self._setup_global_exception_handling()
    
    def _setup_global_exception_handling(self) -> Any:
        """Configurar manejo de excepciones no capturadas."""
        def handle_uncaught_exception(exc_type, exc_value, exc_traceback) -> Any:
            if issubclass(exc_type, KeyboardInterrupt):
                # Permitir que KeyboardInterrupt se propague
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            error = AIVideoError(
                message=str(exc_value),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                context=ErrorContext.RUNTIME,
                details={'uncaught': True}
            )
            
            self.monitor.record_error(error)
            logging.critical(f"Excepción no capturada: {error.to_dict()}")
        
        sys.excepthook = handle_uncaught_exception
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Obtener resumen de errores del sistema."""
        return self.monitor.get_error_summary()
    
    def get_error_rate(self, minutes: int = 5) -> float:
        """Obtener tasa de errores."""
        return self.monitor.get_error_rate(minutes)
    
    def is_system_healthy(self) -> bool:
        """Verificar si el sistema está saludable."""
        error_rate = self.get_error_rate(5)
        return error_rate < 0.1  # Menos de 0.1 errores por minuto

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_context(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: Optional[ErrorContext] = None
) -> Callable:
    """Crear contexto de error personalizado."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with error_context(func.__name__, category, severity):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            async with async_error_context(func.__name__, category, severity):
                return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

def safe_execute(
    operation: Callable,
    *args,
    error_category: ErrorCategory = ErrorCategory.SYSTEM,
    default_return: Any = None,
    **kwargs
) -> Any:
    """Ejecutar operación de forma segura."""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        error = AIVideoError(
            message=str(e),
            category=error_category,
            severity=ErrorSeverity.ERROR
        )
        logging.error(f"Error en operación segura: {error.to_dict()}")
        return default_return

async def safe_execute_async(
    operation: Callable,
    *args,
    error_category: ErrorCategory = ErrorCategory.SYSTEM,
    default_return: Any = None,
    **kwargs
) -> Any:
    """Ejecutar operación asíncrona de forma segura."""
    try:
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            return operation(*args, **kwargs)
    except Exception as e:
        error = AIVideoError(
            message=str(e),
            category=error_category,
            severity=ErrorSeverity.ERROR
        )
        logging.error(f"Error en operación asíncrona segura: {error.to_dict()}")
        return default_return

# =============================================================================
# INITIALIZATION
# =============================================================================

# Instancia global del manejador de errores
global_error_handler = GlobalErrorHandler()

def get_error_handler() -> GlobalErrorHandler:
    """Obtener instancia global del manejador de errores."""
    return global_error_handler

def setup_error_handling(log_level: int = logging.INFO):
    """Configurar sistema de manejo de errores."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_video_errors.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚨 Sistema de manejo de errores inicializado")
    
    return global_error_handler

# Configuración automática al importar
setup_error_handling() 