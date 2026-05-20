"""
Custom Exceptions - Excepciones personalizadas
==============================================

Excepciones personalizadas para el sistema Cursor Agent 24/7.
Proporciona mejor manejo de errores y debugging.
"""

from typing import Optional, Dict, Any


class CursorAgentException(Exception):
    """Excepción base para todas las excepciones del agente"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir excepción a diccionario"""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class TaskExecutionException(CursorAgentException):
    """Excepción durante la ejecución de una tarea"""
    
    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        command: Optional[str] = None,
        sanitize: bool = True,
        **kwargs
    ):
        # Sanitizar mensaje si es necesario
        if sanitize:
            from .security import SecurityValidator
            validator = SecurityValidator()
            message = validator.sanitize_error_message(message)
        
        super().__init__(message, error_code="TASK_EXECUTION_ERROR", **kwargs)
        self.task_id = task_id
        self.command = command
        if task_id:
            self.details["task_id"] = task_id
        if command:
            # No incluir comando completo en detalles por seguridad
            self.details["command_length"] = len(command)
            self.details["command_preview"] = command[:50] + "..." if len(command) > 50 else command


class TaskTimeoutException(TaskExecutionException):
    """Excepción cuando una tarea excede su timeout"""
    
    def __init__(
        self,
        timeout: float,
        task_id: Optional[str] = None,
        command: Optional[str] = None
    ):
        message = f"Task timeout after {timeout}s"
        super().__init__(
            message,
            task_id=task_id,
            command=command,
            error_code="TASK_TIMEOUT"
        )
        self.timeout = timeout
        self.details["timeout"] = timeout


class TaskValidationException(CursorAgentException):
    """Excepción cuando una tarea no pasa la validación"""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[list] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="TASK_VALIDATION_ERROR",
            **kwargs
        )
        self.validation_errors = validation_errors or []
        if validation_errors:
            self.details["validation_errors"] = validation_errors


class AgentNotRunningException(CursorAgentException):
    """Excepción cuando se intenta operar con el agente detenido"""
    
    def __init__(self, operation: str):
        message = f"Cannot perform {operation}: agent is not running"
        super().__init__(message, error_code="AGENT_NOT_RUNNING")
        self.operation = operation
        self.details["operation"] = operation


class RateLimitExceededException(CursorAgentException):
    """Excepción cuando se excede el límite de rate"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        limit: Optional[int] = None
    ):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
        self.retry_after = retry_after
        self.limit = limit
        if retry_after:
            self.details["retry_after"] = retry_after
        if limit:
            self.details["limit"] = limit


class ConfigurationException(CursorAgentException):
    """Excepción relacionada con la configuración"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, error_code="CONFIGURATION_ERROR")
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


class StorageException(CursorAgentException):
    """Excepción relacionada con el almacenamiento persistente"""
    
    def __init__(
        self,
        message: str,
        storage_path: Optional[str] = None,
        operation: Optional[str] = None
    ):
        super().__init__(message, error_code="STORAGE_ERROR")
        self.storage_path = storage_path
        self.operation = operation
        if storage_path:
            self.details["storage_path"] = storage_path
        if operation:
            self.details["operation"] = operation

