"""
Error Handling System
=====================

Sistema mejorado de manejo de errores con jerarquía de excepciones
y manejo centralizado.
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Códigos de error del sistema."""
    # Domain Errors
    DOMAIN_VALIDATION_ERROR = "DOMAIN_VALIDATION_ERROR"
    DOMAIN_BUSINESS_RULE_VIOLATION = "DOMAIN_BUSINESS_RULE_VIOLATION"
    DOMAIN_INVARIANT_VIOLATION = "DOMAIN_INVARIANT_VIOLATION"
    
    # Application Errors
    APPLICATION_VALIDATION_ERROR = "APPLICATION_VALIDATION_ERROR"
    APPLICATION_NOT_FOUND = "APPLICATION_NOT_FOUND"
    APPLICATION_UNAUTHORIZED = "APPLICATION_UNAUTHORIZED"
    APPLICATION_FORBIDDEN = "APPLICATION_FORBIDDEN"
    
    # Infrastructure Errors
    INFRASTRUCTURE_DATABASE_ERROR = "INFRASTRUCTURE_DATABASE_ERROR"
    INFRASTRUCTURE_NETWORK_ERROR = "INFRASTRUCTURE_NETWORK_ERROR"
    INFRASTRUCTURE_TIMEOUT = "INFRASTRUCTURE_TIMEOUT"
    INFRASTRUCTURE_SERVICE_UNAVAILABLE = "INFRASTRUCTURE_SERVICE_UNAVAILABLE"
    
    # Robot Errors
    ROBOT_NOT_CONNECTED = "ROBOT_NOT_CONNECTED"
    ROBOT_MOVEMENT_ERROR = "ROBOT_MOVEMENT_ERROR"
    ROBOT_TRAJECTORY_ERROR = "ROBOT_TRAJECTORY_ERROR"
    ROBOT_SAFETY_ERROR = "ROBOT_SAFETY_ERROR"
    ROBOT_COLLISION_DETECTED = "ROBOT_COLLISION_DETECTED"
    
    # Generic Errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ErrorSeverity(Enum):
    """Severidad del error."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Contexto adicional del error."""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    robot_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorDetails:
    """Detalles del error."""
    code: ErrorCode
    message: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Optional[ErrorContext] = None
    original_error: Optional[Exception] = None
    stack_trace: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "context": {
                "user_id": self.context.user_id if self.context else None,
                "request_id": self.context.request_id if self.context else None,
                "robot_id": self.context.robot_id if self.context else None,
                "operation": self.context.operation if self.context else None,
                "metadata": self.context.metadata if self.context else {},
                "timestamp": self.context.timestamp.isoformat() if self.context else None,
            } if self.context else None,
            "suggestions": self.suggestions,
            "stack_trace": self.stack_trace if self.severity == ErrorSeverity.CRITICAL else None
        }


class BaseArchitectureError(Exception):
    """Excepción base para errores de arquitectura."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.code = code
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_error = original_error
        self.suggestions = suggestions or []
        self._capture_stack_trace()
    
    def _capture_stack_trace(self):
        """Capturar stack trace si es crítico."""
        if self.severity == ErrorSeverity.CRITICAL:
            self.stack_trace = traceback.format_exc()
        else:
            self.stack_trace = None
    
    def to_error_details(self) -> ErrorDetails:
        """Convertir a ErrorDetails."""
        return ErrorDetails(
            code=self.code,
            message=str(self),
            severity=self.severity,
            context=self.context,
            original_error=self.original_error,
            stack_trace=self.stack_trace,
            suggestions=self.suggestions
        )


class DomainError(BaseArchitectureError):
    """Error de dominio."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.DOMAIN_VALIDATION_ERROR,
        **kwargs
    ):
        super().__init__(
            message,
            code=code,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ApplicationError(BaseArchitectureError):
    """Error de aplicación."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.APPLICATION_VALIDATION_ERROR,
        **kwargs
    ):
        super().__init__(
            message,
            code=code,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class InfrastructureError(BaseArchitectureError):
    """Error de infraestructura."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INFRASTRUCTURE_DATABASE_ERROR,
        **kwargs
    ):
        super().__init__(
            message,
            code=code,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class RobotError(BaseArchitectureError):
    """Error relacionado con robots."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.ROBOT_MOVEMENT_ERROR,
        robot_id: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context') or ErrorContext()
        context.robot_id = robot_id
        kwargs['context'] = context
        super().__init__(
            message,
            code=code,
            severity=ErrorSeverity.CRITICAL if code == ErrorCode.ROBOT_SAFETY_ERROR else ErrorSeverity.HIGH,
            **kwargs
        )


class ErrorHandler:
    """Manejador centralizado de errores."""
    
    def __init__(self):
        """Inicializar manejador."""
        self.error_loggers: Dict[ErrorSeverity, List[callable]] = {
            ErrorSeverity.LOW: [],
            ErrorSeverity.MEDIUM: [],
            ErrorSeverity.HIGH: [],
            ErrorSeverity.CRITICAL: []
        }
    
    def register_logger(self, severity: ErrorSeverity, logger_func: callable):
        """Registrar logger para severidad específica."""
        self.error_loggers[severity].append(logger_func)
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> ErrorDetails:
        """
        Manejar error.
        
        Args:
            error: Excepción a manejar
            context: Contexto adicional
            
        Returns:
            Detalles del error
        """
        # Convertir a ErrorDetails
        if isinstance(error, BaseArchitectureError):
            error_details = error.to_error_details()
            if context:
                error_details.context = context
        else:
            error_details = ErrorDetails(
                code=ErrorCode.UNKNOWN_ERROR,
                message=str(error),
                severity=ErrorSeverity.MEDIUM,
                context=context or ErrorContext(),
                original_error=error,
                stack_trace=traceback.format_exc()
            )
        
        # Log según severidad
        self._log_error(error_details)
        
        return error_details
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error según severidad."""
        loggers = self.error_loggers.get(error_details.severity, [])
        
        for logger_func in loggers:
            try:
                logger_func(error_details)
            except Exception as e:
                logger.error(f"Error en logger: {e}")
        
        # Log estándar
        log_message = f"[{error_details.code.value}] {error_details.message}"
        if error_details.context:
            log_message += f" | Context: {error_details.context.operation}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=error_details.original_error)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=error_details.original_error)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def create_error_response(
        self,
        error_details: ErrorDetails,
        include_stack_trace: bool = False
    ) -> Dict[str, Any]:
        """
        Crear respuesta de error para API.
        
        Args:
            error_details: Detalles del error
            include_stack_trace: Incluir stack trace
            
        Returns:
            Diccionario con respuesta de error
        """
        response = {
            "error": {
                "code": error_details.code.value,
                "message": error_details.message,
                "severity": error_details.severity.value
            }
        }
        
        if error_details.context:
            response["error"]["context"] = {
                "request_id": error_details.context.request_id,
                "operation": error_details.context.operation
            }
        
        if error_details.suggestions:
            response["error"]["suggestions"] = error_details.suggestions
        
        if include_stack_trace and error_details.stack_trace:
            response["error"]["stack_trace"] = error_details.stack_trace
        
        return response


# Instancia global del manejador de errores
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Obtener instancia global del manejador de errores."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(error: Exception, context: Optional[ErrorContext] = None) -> ErrorDetails:
    """
    Función helper para manejar errores.
    
    Args:
        error: Excepción a manejar
        context: Contexto adicional
        
    Returns:
        Detalles del error
    """
    handler = get_error_handler()
    return handler.handle_error(error, context)




