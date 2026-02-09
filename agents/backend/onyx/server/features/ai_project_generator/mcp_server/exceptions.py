"""
MCP Exceptions - Excepciones personalizadas para MCP
====================================================

Jerarquía de excepciones específicas para el Model Context Protocol,
proporcionando información detallada sobre diferentes tipos de errores.
"""

from typing import Optional, Any, Dict


class MCPError(Exception):
    """
    Excepción base para todos los errores MCP.
    
    Todas las excepciones específicas de MCP heredan de esta clase,
    permitiendo capturar cualquier error relacionado con MCP de forma genérica.
    
    Attributes:
        message: Mensaje de error descriptivo.
        details: Diccionario con detalles adicionales del error (opcional).
        error_code: Código de error específico (opcional).
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar excepción MCP.
        
        Args:
            message: Mensaje de error descriptivo (debe ser no vacío).
            details: Diccionario con detalles adicionales del error (opcional).
            error_code: Código de error específico (opcional).
            
        Raises:
            ValueError: Si message es inválido
        """
        if not message or not isinstance(message, str):
            raise ValueError("message must be a non-empty string")
        if not message.strip():
            raise ValueError("message cannot be empty or whitespace only")
        
        super().__init__(message)
        self.message: str = message.strip()
        self.details: Optional[Dict[str, Any]] = details if details is None or isinstance(details, dict) else {}
        self.error_code: Optional[str] = error_code.strip() if error_code and isinstance(error_code, str) else error_code
    
    def __str__(self) -> str:
        """Retorna representación en string de la excepción"""
        result = self.message
        if self.error_code:
            result = f"[{self.error_code}] {result}"
        if self.details:
            result = f"{result} (details: {self.details})"
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la excepción a diccionario para serialización.
        
        Returns:
            Diccionario con información de la excepción, incluyendo
            nombre de la clase, mensaje, código de error y detalles.
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
        }
        
        if self.error_code:
            result["error_code"] = self.error_code
        
        if self.details:
            result["details"] = self.details
        
        return result
    
    def get_user_message(self) -> str:
        """
        Retorna un mensaje de error amigable para el usuario.
        
        Returns:
            Mensaje de error sin detalles técnicos internos.
        """
        return self.message
    
    def has_details(self) -> bool:
        """
        Verifica si la excepción tiene detalles adicionales.
        
        Returns:
            True si tiene detalles, False en caso contrario.
        """
        return self.details is not None and len(self.details) > 0


class MCPAuthenticationError(MCPError):
    """
    Error de autenticación.
    
    Se lanza cuando las credenciales proporcionadas son inválidas
    o cuando el usuario no está autenticado.
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar error de autenticación.
        
        Args:
            message: Mensaje de error (default: "Authentication failed").
            details: Detalles adicionales (opcional).
            error_code: Código de error (opcional).
        """
        super().__init__(
            message,
            details=details,
            error_code=error_code or "AUTH_ERROR"
        )


class MCPAuthorizationError(MCPError):
    """
    Error de autorización.
    
    Se lanza cuando el usuario autenticado no tiene permisos
    para realizar la operación solicitada.
    """
    
    def __init__(
        self,
        message: str = "Authorization failed",
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar error de autorización.
        
        Args:
            message: Mensaje de error (default: "Authorization failed").
            details: Detalles adicionales (opcional).
            error_code: Código de error (opcional).
        """
        super().__init__(
            message,
            details=details,
            error_code=error_code or "AUTHZ_ERROR"
        )


class MCPResourceNotFoundError(MCPError):
    """
    Recurso no encontrado.
    
    Se lanza cuando se intenta acceder a un recurso que no existe
    o no está disponible.
    """
    
    def __init__(
        self,
        resource_id: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar error de recurso no encontrado.
        
        Args:
            resource_id: ID del recurso que no se encontró.
            message: Mensaje de error personalizado (opcional).
            details: Detalles adicionales (opcional).
            error_code: Código de error (opcional).
        """
        if message is None:
            message = f"Resource not found: {resource_id}"
        
        if details is None:
            details = {}
        details["resource_id"] = resource_id
        
        super().__init__(
            message,
            details=details,
            error_code=error_code or "RESOURCE_NOT_FOUND"
        )


class MCPConnectorError(MCPError):
    """
    Error en conector.
    
    Se lanza cuando hay un error al comunicarse con el conector
    o cuando el conector falla al ejecutar una operación.
    """
    
    def __init__(
        self,
        connector_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar error de conector.
        
        Args:
            connector_name: Nombre del conector que falló.
            message: Mensaje de error personalizado (opcional).
            details: Detalles adicionales (opcional).
            error_code: Código de error (opcional).
        """
        if message is None:
            message = f"Connector error: {connector_name}"
        
        if details is None:
            details = {}
        details["connector_name"] = connector_name
        
        super().__init__(
            message,
            details=details,
            error_code=error_code or "CONNECTOR_ERROR"
        )


class MCPOperationError(MCPError):
    """
    Error en operación.
    
    Se lanza cuando una operación falla por razones no relacionadas
    con autenticación, autorización o recursos no encontrados.
    """
    
    def __init__(
        self,
        operation: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar error de operación.
        
        Args:
            operation: Nombre de la operación que falló.
            message: Mensaje de error personalizado (opcional).
            details: Detalles adicionales (opcional).
            error_code: Código de error (opcional).
        """
        if message is None:
            message = f"Operation failed: {operation}"
        
        if details is None:
            details = {}
        details["operation"] = operation
        
        super().__init__(
            message,
            details=details,
            error_code=error_code or "OPERATION_ERROR"
        )


class MCPValidationError(MCPError):
    """
    Error de validación.
    
    Se lanza cuando los datos proporcionados no pasan la validación
    o cuando los parámetros son inválidos.
    """
    
    def __init__(
        self,
        field: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar error de validación.
        
        Args:
            field: Campo que falló la validación (opcional).
            message: Mensaje de error personalizado (opcional).
            details: Detalles adicionales (opcional).
            error_code: Código de error (opcional).
        """
        if message is None:
            if field:
                message = f"Validation failed for field: {field}"
            else:
                message = "Validation failed"
        
        if details is None:
            details = {}
        if field:
            details["field"] = field
        
        super().__init__(
            message,
            details=details,
            error_code=error_code or "VALIDATION_ERROR"
        )


class MCPRateLimitError(MCPError):
    """
    Error de rate limit.
    
    Se lanza cuando se excede el límite de solicitudes permitidas
    en un período de tiempo determinado.
    """
    
    def __init__(
        self,
        limit: Optional[int] = None,
        window: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar error de rate limit.
        
        Args:
            limit: Límite de solicitudes (opcional).
            window: Ventana de tiempo (opcional, ej: "per minute").
            message: Mensaje de error personalizado (opcional).
            details: Detalles adicionales (opcional).
            error_code: Código de error (opcional).
        """
        if message is None:
            if limit and window:
                message = f"Rate limit exceeded: {limit} {window}"
            else:
                message = "Rate limit exceeded"
        
        if details is None:
            details = {}
        if limit:
            details["limit"] = limit
        if window:
            details["window"] = window
        
        super().__init__(
            message,
            details=details,
            error_code=error_code or "RATE_LIMIT_ERROR"
        )


class MCPContextLimitError(MCPError):
    """
    Error de límite de contexto.
    
    Se lanza cuando el tamaño del contexto excede los límites permitidos
    o cuando hay demasiados frames de contexto.
    """
    
    def __init__(
        self,
        limit: Optional[int] = None,
        current: Optional[int] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Inicializar error de límite de contexto.
        
        Args:
            limit: Límite permitido (opcional).
            current: Valor actual que excede el límite (opcional).
            message: Mensaje de error personalizado (opcional).
            details: Detalles adicionales (opcional).
            error_code: Código de error (opcional).
        """
        if message is None:
            if limit and current:
                message = f"Context limit exceeded: {current} > {limit}"
            else:
                message = "Context limit exceeded"
        
        if details is None:
            details = {}
        if limit:
            details["limit"] = limit
        if current:
            details["current"] = current
        
        super().__init__(
            message,
            details=details,
            error_code=error_code or "CONTEXT_LIMIT_ERROR"
        )
