"""
Excepciones personalizadas para el agente.

Todas las excepciones incluyen contexto adicional para mejor debugging.
"""

from typing import Optional, Dict, Any


class GitHubAgentError(Exception):
    """Excepción base para errores del agente."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Inicializar excepción.
        
        Args:
            message: Mensaje de error
            details: Detalles adicionales del error
            original_error: Error original que causó esta excepción
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        """Representación en string de la excepción."""
        base = self.message
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base += f" ({details_str})"
        return base
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir excepción a diccionario para serialización.
        
        Returns:
            Diccionario con información de la excepción
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details.copy() if self.details else {}
        }
        
        if self.original_error:
            result["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error)
            }
        
        return result
    
    def __repr__(self) -> str:
        """Representación detallada de la excepción."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details}, "
            f"original_error={type(self.original_error).__name__ if self.original_error else None}"
            f")"
        )


class GitHubClientError(GitHubAgentError):
    """Error en el cliente de GitHub."""
    
    def __init__(
        self,
        message: str,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Inicializar error de cliente de GitHub.
        
        Args:
            message: Mensaje de error
            owner: Propietario del repositorio (si aplica)
            repo: Nombre del repositorio (si aplica)
            details: Detalles adicionales
            original_error: Error original
        """
        details = details or {}
        if owner:
            details["owner"] = owner
        if repo:
            details["repo"] = repo
        super().__init__(message, details, original_error)


class TaskProcessingError(GitHubAgentError):
    """Error al procesar una tarea."""
    
    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Inicializar error de procesamiento de tarea.
        
        Args:
            message: Mensaje de error
            task_id: ID de la tarea (si aplica)
            details: Detalles adicionales
            original_error: Error original
        """
        details = details or {}
        if task_id:
            details["task_id"] = task_id
        super().__init__(message, details, original_error)


class StorageError(GitHubAgentError):
    """Error en el almacenamiento."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Inicializar error de almacenamiento.
        
        Args:
            message: Mensaje de error
            operation: Operación que falló (save, get, update, etc.)
            details: Detalles adicionales
            original_error: Error original
        """
        details = details or {}
        if operation:
            details["operation"] = operation
        super().__init__(message, details, original_error)


class InstructionParseError(TaskProcessingError):
    """Error al parsear una instrucción."""
    
    def __init__(
        self,
        message: str,
        instruction: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Inicializar error de parsing de instrucción.
        
        Args:
            message: Mensaje de error
            instruction: Instrucción que falló (si aplica)
            details: Detalles adicionales
            original_error: Error original
        """
        details = details or {}
        if instruction:
            details["instruction"] = instruction[:100]  # Limitar longitud
        super().__init__(message, details=details, original_error=original_error)

