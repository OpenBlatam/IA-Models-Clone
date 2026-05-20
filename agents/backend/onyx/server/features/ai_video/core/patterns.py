from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
import asyncio
import functools
from typing import (
from dataclasses import dataclass, field
from enum import Enum, auto
from ..error_handling import (
from .validators import (
from .error_handlers import (
from typing import Any, List, Dict, Optional
"""
🎯 HAPPY PATH PATTERNS - CORE MODULE
====================================

Módulo principal que implementa el principio "happy path last" donde todas las 
condiciones de error se manejan primero y la lógica principal se coloca al final 
para mejorar la legibilidad del código.
"""

    Any, Optional, Union, Dict, List, Tuple, Callable, 
    TypeVar, Generic, Protocol, runtime_checkable
)

    AIVideoError, ErrorCategory, ErrorSeverity, ErrorContext,
    ValidationError, SystemError, ConfigurationError
)

# =============================================================================
# HAPPY PATH PATTERNS
# =============================================================================

class HappyPathPattern(Enum):
    """Patrones de happy path."""
    VALIDATION_FIRST = auto()
    ERROR_HANDLING_FIRST = auto()
    RESOURCE_CHECK_FIRST = auto()
    GUARD_CLAUSE_FIRST = auto()
    CLEANUP_LAST = auto()

@dataclass
class HappyPathResult:
    """Resultado del happy path."""
    success: bool
    value: Any
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# HAPPY PATH DECORATORS
# =============================================================================

def happy_path_last(
    validators: List[Callable] = None,
    error_handlers: List[Callable] = None,
    cleanup_handlers: List[Callable] = None
):
    """Decorador para implementar happy path last."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 1. Validaciones al inicio
            if validators:
                for validator in validators:
                    try:
                        result = validator(*args, **kwargs)
                        if not result:
                            return {"error": f"Validation failed in {func.__name__}"}
                    except Exception as e:
                        return {"error": f"Validation error: {e}"}
            
            # 2. Manejo de errores al inicio
            if error_handlers:
                for handler in error_handlers:
                    try:
                        result = handler(*args, **kwargs)
                        if result is not None:
                            return result
                    except Exception as e:
                        return {"error": f"Error handler failed: {e}"}
            
            # 3. Happy path al final
            try:
                result = func(*args, **kwargs)
                
                # 4. Cleanup al final
                if cleanup_handlers:
                    for handler in cleanup_handlers:
                        try:
                            handler(*args, **kwargs)
                        except Exception as e:
                            logging.warning(f"Cleanup handler failed: {e}")
                
                return result
            except Exception as e:
                return {"error": f"Function execution failed: {e}"}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # 1. Validaciones al inicio
            if validators:
                for validator in validators:
                    try:
                        if asyncio.iscoroutinefunction(validator):
                            result = await validator(*args, **kwargs)
                        else:
                            result = validator(*args, **kwargs)
                        
                        if not result:
                            return {"error": f"Validation failed in {func.__name__}"}
                    except Exception as e:
                        return {"error": f"Validation error: {e}"}
            
            # 2. Manejo de errores al inicio
            if error_handlers:
                for handler in error_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(*args, **kwargs)
                        else:
                            result = handler(*args, **kwargs)
                        
                        if result is not None:
                            return result
                    except Exception as e:
                        return {"error": f"Error handler failed: {e}"}
            
            # 3. Happy path al final
            try:
                result = await func(*args, **kwargs)
                
                # 4. Cleanup al final
                if cleanup_handlers:
                    for handler in cleanup_handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(*args, **kwargs)
                            else:
                                handler(*args, **kwargs)
                        except Exception as e:
                            logging.warning(f"Cleanup handler failed: {e}")
                
                return result
            except Exception as e:
                return {"error": f"Function execution failed: {e}"}
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

# =============================================================================
# HAPPY PATH PATTERNS
# =============================================================================

class HappyPathPatterns:
    """Patrones para implementar happy path last."""
    
    @staticmethod
    def validation_first_pattern(func: Callable) -> Callable:
        """Patrón: validaciones primero, happy path al final."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 1. Validaciones al inicio
            if not _validate_inputs(*args, **kwargs):
                return {"error": "Input validation failed"}
            
            if not _validate_resources(*args, **kwargs):
                return {"error": "Resource validation failed"}
            
            if not _validate_state(*args, **kwargs):
                return {"error": "State validation failed"}
            
            # 2. Happy path al final
            return func(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def error_handling_first_pattern(func: Callable) -> Callable:
        """Patrón: manejo de errores primero, happy path al final."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 1. Manejo de errores al inicio
            error_result = _handle_common_errors(*args, **kwargs)
            if error_result is not None:
                return error_result
            
            error_result = _handle_system_errors(*args, **kwargs)
            if error_result is not None:
                return error_result
            
            error_result = _handle_business_errors(*args, **kwargs)
            if error_result is not None:
                return error_result
            
            # 2. Happy path al final
            return func(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def guard_clause_first_pattern(func: Callable) -> Callable:
        """Patrón: guard clauses primero, happy path al final."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 1. Guard clauses al inicio
            if _is_none_or_empty(*args, **kwargs):
                return {"error": "Required parameters are missing"}
            
            if _is_invalid_format(*args, **kwargs):
                return {"error": "Invalid format"}
            
            if _is_insufficient_resources(*args, **kwargs):
                return {"error": "Insufficient resources"}
            
            # 2. Happy path al final
            return func(*args, **kwargs)
        
        return wrapper

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_happy_path_last(func: Callable) -> Callable:
    """Aplicar patrón happy path last a función existente."""
    return HappyPathPatterns.validation_first_pattern(func)

def create_happy_path_validator(conditions: List[Callable]) -> Callable:
    """Crear validador para happy path last."""
    def validator(*args, **kwargs) -> bool:
        for condition in conditions:
            try:
                if not condition(*args, **kwargs):
                    return False
            except Exception:
                return False
        return True
    
    return validator

# =============================================================================
# INITIALIZATION
# =============================================================================

def setup_happy_path_last():
    """Configurar sistema de happy path last."""
    logger = logging.getLogger(__name__)
    logger.info("🎯 Sistema de happy path last inicializado")
    
    return {
        "patterns": HappyPathPatterns,
        "decorators": {
            "happy_path_last": happy_path_last,
            "apply_happy_path_last": apply_happy_path_last
        }
    }

# Configuración automática al importar
happy_path_system = setup_happy_path_last()

# Importar funciones auxiliares
    _validate_inputs, _validate_resources, _validate_state
)

    _handle_common_errors, _handle_system_errors, _handle_business_errors,
    _is_none_or_empty, _is_invalid_format, _is_insufficient_resources
) 