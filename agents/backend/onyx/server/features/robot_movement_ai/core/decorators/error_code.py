"""
ErrorCode Decorator
===================

Decorador centralizado para anotar excepciones con códigos de error y descripciones.
Este módulo proporciona una implementación única y reutilizable del decorador @ErrorCode.
"""

from typing import Type, TypeVar

T = TypeVar('T', bound=Type[Exception])


def ErrorCode(description: str):
    """
    Decorador para anotar excepciones con códigos de error y descripciones.
    
    Este decorador almacena la descripción del error en la clase de excepción,
    permitiendo que los constructores usen automáticamente esta descripción
    en lugar de requerir argumentos de mensaje.
    
    Args:
        description: Descripción del error que se usará en el constructor.
    
    Returns:
        Decorador que modifica la clase de excepción.
    
    Usage:
        @ErrorCode(description="Invalid input provided")
        class MyException(Exception):
            def __init__(self):
                message = getattr(self.__class__, '_error_description', "Invalid input provided")
                super().__init__(message)
                self.message = message
    
    Example:
        >>> @ErrorCode(description="Route not found")
        ... class RouteNotFoundError(Exception):
        ...     def __init__(self):
        ...         message = getattr(self.__class__, '_error_description', "Route not found")
        ...         super().__init__(message)
        ...         self.message = message
        >>> 
        >>> error = RouteNotFoundError()
        >>> str(error)
        'Route not found'
        >>> error.message
        'Route not found'
    """
    def decorator(cls: T) -> T:
        """
        Decorador interno que almacena la descripción en la clase.
        
        Args:
            cls: Clase de excepción a decorar.
        
        Returns:
            Clase decorada con la descripción almacenada.
        """
        # Almacenar la descripción en la clase como atributo de clase
        cls._error_description = description
        
        # Agregar método helper para obtener la descripción
        if not hasattr(cls, 'get_error_description'):
            @classmethod
            def get_error_description(cls_inner):
                """Obtener la descripción del error desde el decorador."""
                return getattr(cls_inner, '_error_description', None)
            
            cls.get_error_description = get_error_description
        
        return cls
    
    return decorator


def get_error_description(exception_class: Type[Exception]) -> str:
    """
    Función helper para obtener la descripción de error de una clase de excepción.
    
    Args:
        exception_class: Clase de excepción.
    
    Returns:
        Descripción del error o None si no está disponible.
    
    Example:
        >>> @ErrorCode(description="Test error")
        ... class TestError(Exception):
        ...     pass
        >>> 
        >>> get_error_description(TestError)
        'Test error'
    """
    return getattr(exception_class, '_error_description', None)

