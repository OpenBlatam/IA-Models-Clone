# Refactoring Analysis: Helper Functions for Code Optimization

## Executive Summary

This document identifies repetitive patterns in the codebase and proposes helper functions to improve maintainability, reduce duplication, and simplify future updates. The analysis covers API routes, service classes, caching mechanisms, error handling, and response formatting.

---

## 1. Cache Key Generation Helper

### Problem Identified

The codebase repeatedly uses `hashlib.md5()` to generate cache keys with identical patterns:

**Locations Found:**
- `api/routes.py` (lines 142-144, 395)
- `services/content_generator.py` (lines 173-175, 265-267, 324-326)
- `services/identity_analyzer.py` (lines 116, 238)
- `api/decorators.py` (line 120)
- Multiple utility files

**Current Pattern:**
```python
cache_key = hashlib.md5(f"prefix_{var1}_{var2}_{var3}".encode()).hexdigest()
```

### Proposed Helper Function

```python
# utils/cache_helpers.py
import hashlib
from typing import Any, Union, List

def generate_cache_key(*parts: Union[str, int, float, None]) -> str:
    """
    Genera una clave de caché consistente a partir de múltiples partes.
    
    Args:
        *parts: Componentes para la clave (strings, números, o None)
        
    Returns:
        Hash MD5 hexadecimal de la clave generada
        
    Examples:
        >>> generate_cache_key("extract_profile", "tiktok", "username123")
        'a1b2c3d4e5f6...'
        
        >>> generate_cache_key("identity", identity_id, "v2")
        'f6e5d4c3b2a1...'
    """
    # Filtrar None y convertir todo a string
    parts_str = [str(p) if p is not None else "" for p in parts]
    key_string = "_".join(parts_str)
    return hashlib.md5(key_string.encode()).hexdigest()

def generate_cache_key_from_dict(prefix: str, data: dict) -> str:
    """
    Genera clave de caché desde un diccionario ordenado.
    
    Args:
        prefix: Prefijo para la clave
        data: Diccionario con datos a incluir
        
    Returns:
        Hash MD5 hexadecimal
    """
    # Ordenar keys para consistencia
    sorted_items = sorted(data.items())
    parts = [prefix] + [f"{k}:{v}" for k, v in sorted_items]
    return generate_cache_key(*parts)
```

### Integration Example

**Before:**
```python
# api/routes.py
cache_key = hashlib.md5(
    f"extract_profile_{request.platform}_{request.username}".encode()
).hexdigest()
```

**After:**
```python
# api/routes.py
from ..utils.cache_helpers import generate_cache_key

cache_key = generate_cache_key("extract_profile", request.platform, request.username)
```

**Before:**
```python
# services/content_generator.py
cache_key = hashlib.md5(
    f"instagram_{self.identity.profile_id}_{topic}_{style}_{use_lora}".encode()
).hexdigest()
```

**After:**
```python
# services/content_generator.py
from ..utils.cache_helpers import generate_cache_key

cache_key = generate_cache_key(
    "instagram", 
    self.identity.profile_id, 
    topic, 
    style, 
    use_lora
)
```

### Benefits
- **Consistency**: Single source of truth for cache key generation
- **Maintainability**: Easy to change hashing algorithm (e.g., to SHA256)
- **Readability**: More explicit and self-documenting
- **Type Safety**: Handles None values gracefully

---

## 2. API Response Formatter Helper

### Problem Identified

Multiple endpoints construct similar response dictionaries with "success", data, and metadata fields:

**Locations Found:**
- `api/routes.py` (lines 164-174, 253-264, 357-368, 409-412, 447-452)

**Current Pattern:**
```python
return {
    "success": True,
    "identity_id": identity.profile_id,
    "identity": identity.model_dump(),
    "stats": {...}
}
```

### Proposed Helper Function

```python
# api/response_helpers.py
from typing import Any, Dict, Optional
from fastapi import status
from fastapi.responses import JSONResponse

def success_response(
    data: Any = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_200_OK
) -> Dict[str, Any]:
    """
    Crea una respuesta de éxito estandarizada.
    
    Args:
        data: Datos principales de la respuesta
        message: Mensaje opcional de éxito
        metadata: Metadatos adicionales (stats, pagination, etc.)
        status_code: Código de estado HTTP (default: 200)
        
    Returns:
        Diccionario con estructura de respuesta estándar
        
    Examples:
        >>> success_response(
        ...     data={"identity": identity.model_dump()},
        ...     metadata={"count": 1}
        ... )
        {
            "success": True,
            "data": {"identity": {...}},
            "metadata": {"count": 1}
        }
    """
    response = {"success": True}
    
    if data is not None:
        response["data"] = data
    elif message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response

def error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST
) -> Dict[str, Any]:
    """
    Crea una respuesta de error estandarizada.
    
    Args:
        message: Mensaje de error
        error_code: Código de error opcional
        details: Detalles adicionales del error
        status_code: Código de estado HTTP
        
    Returns:
        Diccionario con estructura de error estándar
    """
    response = {
        "success": False,
        "error": {
            "message": message
        }
    }
    
    if error_code:
        response["error"]["code"] = error_code
    
    if details:
        response["error"]["details"] = details
    
    return response

def paginated_response(
    items: list,
    page: int,
    page_size: int,
    total: Optional[int] = None,
    **additional_metadata
) -> Dict[str, Any]:
    """
    Crea una respuesta paginada estandarizada.
    
    Args:
        items: Lista de items en la página actual
        page: Número de página (1-indexed)
        page_size: Tamaño de página
        total: Total de items (opcional)
        **additional_metadata: Metadatos adicionales
        
    Returns:
        Diccionario con respuesta paginada
    """
    response = success_response(
        data=items,
        metadata={
            "pagination": {
                "page": page,
                "page_size": page_size,
                "count": len(items)
            },
            **additional_metadata
        }
    )
    
    if total is not None:
        response["metadata"]["pagination"]["total"] = total
        response["metadata"]["pagination"]["total_pages"] = (
            (total + page_size - 1) // page_size
        )
    
    return response
```

### Integration Example

**Before:**
```python
# api/routes.py
return {
    "success": True,
    "identity_id": identity.profile_id,
    "identity": identity.model_dump(),
    "stats": {
        "total_videos": identity.total_videos,
        "total_posts": identity.total_posts,
        "total_comments": identity.total_comments
    }
}
```

**After:**
```python
# api/routes.py
from .response_helpers import success_response

return success_response(
    data={
        "identity_id": identity.profile_id,
        "identity": identity.model_dump()
    },
    metadata={
        "stats": {
            "total_videos": identity.total_videos,
            "total_posts": identity.total_posts,
            "total_comments": identity.total_comments
        }
    }
)
```

**Before:**
```python
# api/routes.py
return {
    "success": True,
    "identity_id": identity_id,
    "count": len(content_list),
    "content": [c.model_dump() for c in content_list]
}
```

**After:**
```python
# api/routes.py
from .response_helpers import success_response

return success_response(
    data={
        "identity_id": identity_id,
        "content": [c.model_dump() for c in content_list]
    },
    metadata={"count": len(content_list)}
)
```

### Benefits
- **Consistency**: Todas las respuestas siguen el mismo formato
- **Maintainability**: Cambios en formato se hacen en un solo lugar
- **Extensibility**: Fácil agregar campos comunes (timestamps, version, etc.)
- **Type Safety**: Mejor documentación y validación

---

## 3. HTTP Exception Helper

### Problem Identified

Repeated patterns for raising HTTPExceptions with similar structure:

**Locations Found:**
- `api/routes.py` (18 occurrences)
- `api/decorators.py` (2 occurrences)
- `middleware/security.py` (2 occurrences)
- `middleware/rate_limiter.py` (1 occurrence)

**Current Pattern:**
```python
raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail=f"Identidad no encontrada: {identity_id}"
)
```

### Proposed Helper Function

```python
# api/exception_helpers.py
from fastapi import HTTPException, status
from typing import Optional, Dict, Any

class APIException(HTTPException):
    """Excepción HTTP personalizada con estructura consistente"""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.details = details or {}
        
        detail_dict = {"message": message}
        if error_code:
            detail_dict["error_code"] = error_code
        if details:
            detail_dict["details"] = details
        
        super().__init__(status_code=status_code, detail=detail_dict)

def not_found(resource: str, identifier: str) -> APIException:
    """
    Crea excepción 404 para recurso no encontrado.
    
    Args:
        resource: Tipo de recurso (ej: "Identidad", "Contenido")
        identifier: Identificador del recurso
        
    Returns:
        APIException con status 404
    """
    return APIException(
        message=f"{resource} no encontrado: {identifier}",
        status_code=status.HTTP_404_NOT_FOUND,
        error_code="RESOURCE_NOT_FOUND",
        details={"resource": resource, "identifier": identifier}
    )

def validation_error(message: str, field: Optional[str] = None) -> APIException:
    """
    Crea excepción 400 para error de validación.
    
    Args:
        message: Mensaje de error
        field: Campo que falló la validación (opcional)
        
    Returns:
        APIException con status 400
    """
    details = {}
    if field:
        details["field"] = field
    
    return APIException(
        message=message,
        status_code=status.HTTP_400_BAD_REQUEST,
        error_code="VALIDATION_ERROR",
        details=details
    )

def unauthorized(message: str = "No autorizado") -> APIException:
    """Crea excepción 401 para no autorizado."""
    return APIException(
        message=message,
        status_code=status.HTTP_401_UNAUTHORIZED,
        error_code="UNAUTHORIZED"
    )

def forbidden(message: str = "Acceso prohibido") -> APIException:
    """Crea excepción 403 para prohibido."""
    return APIException(
        message=message,
        status_code=status.HTTP_403_FORBIDDEN,
        error_code="FORBIDDEN"
    )

def internal_error(message: str = "Error interno del servidor") -> APIException:
    """Crea excepción 500 para error interno."""
    return APIException(
        message=message,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_ERROR"
    )
```

### Integration Example

**Before:**
```python
# api/routes.py
if not identity:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Identidad no encontrada: {identity_id}"
    )
```

**After:**
```python
# api/routes.py
from .exception_helpers import not_found

if not identity:
    raise not_found("Identidad", identity_id)
```

**Before:**
```python
# api/routes.py
if not any([request.tiktok_username, request.instagram_username, request.youtube_channel_id]):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Al menos un perfil debe ser proporcionado"
    )
```

**After:**
```python
# api/routes.py
from .exception_helpers import validation_error

if not any([request.tiktok_username, request.instagram_username, request.youtube_channel_id]):
    raise validation_error("Al menos un perfil debe ser proporcionado")
```

**Before:**
```python
# api/decorators.py
raise HTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail=f"Error interno: {str(e)}"
)
```

**After:**
```python
# api/decorators.py
from .exception_helpers import internal_error

raise internal_error(f"Error interno: {str(e)}")
```

### Benefits
- **Consistency**: Errores estructurados de forma uniforme
- **Maintainability**: Cambios en formato de errores centralizados
- **Debugging**: Códigos de error facilitan troubleshooting
- **API Documentation**: Mejor documentación automática de errores

---

## 4. Cache Management Helper

### Problem Identified

Repeated cache checking, storing, and size management logic:

**Locations Found:**
- `api/routes.py` (lines 141-147, 176-180, 394-417)
- `services/content_generator.py` (lines 172-181, 204-208, 264-272, 293-297, 322-331, 349-353)

**Current Pattern:**
```python
# Check cache
if use_cache:
    cache_key = hashlib.md5(...).hexdigest()
    if cache_key in _response_cache:
        return _response_cache[cache_key]

# ... do work ...

# Store in cache
if use_cache:
    if len(_response_cache) >= _cache_max_size:
        _response_cache.popitem(last=False)
    _response_cache[cache_key] = result
```

### Proposed Helper Function

```python
# utils/cache_manager.py
from typing import Any, Optional, Callable, TypeVar, OrderedDict
from collections import OrderedDict
import hashlib
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ResponseCache:
    """
    Gestor de caché para respuestas de API con tamaño limitado.
    Implementa política LRU (Least Recently Used).
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializa el caché.
        
        Args:
            max_size: Tamaño máximo del caché
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene valor del caché.
        
        Args:
            key: Clave de caché
            
        Returns:
            Valor en caché o None si no existe
        """
        if key in self._cache:
            # Mover al final (más reciente)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Guarda valor en caché.
        
        Args:
            key: Clave de caché
            value: Valor a guardar
        """
        # Si existe, mover al final
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            # Si está lleno, eliminar el más antiguo
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        
        self._cache[key] = value
    
    def clear(self) -> None:
        """Limpia todo el caché."""
        self._cache.clear()
    
    def size(self) -> int:
        """Retorna el tamaño actual del caché."""
        return len(self._cache)

# Instancia global
_global_cache = ResponseCache(max_size=1000)

def get_cache() -> ResponseCache:
    """Obtiene la instancia global del caché."""
    return _global_cache

def cached(
    cache_key_func: Optional[Callable[..., str]] = None,
    use_cache: bool = True
):
    """
    Decorador para cachear resultados de funciones.
    
    Args:
        cache_key_func: Función para generar clave de caché desde argumentos
        use_cache: Si usar caché (default: True)
        
    Usage:
        @cached(lambda identity_id: f"identity_{identity_id}")
        async def get_identity(identity_id: str):
            # ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args, **kwargs):
            if not use_cache:
                return await func(*args, **kwargs)
            
            # Generar clave
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default: usar nombre de función + argumentos
                key_parts = [func.__name__] + [str(a) for a in args] + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
                cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            
            # Verificar caché
            cache = get_cache()
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en caché
            cache.set(cache_key, result)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            if not use_cache:
                return func(*args, **kwargs)
            
            # Generar clave
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__] + [str(a) for a in args] + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
                cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            
            # Verificar caché
            cache = get_cache()
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en caché
            cache.set(cache_key, result)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        # Retornar wrapper apropiado
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
```

### Integration Example

**Before:**
```python
# api/routes.py
# Verificar caché de respuesta si use_cache está habilitado
if request.use_cache:
    cache_key = hashlib.md5(
        f"extract_profile_{request.platform}_{request.username}".encode()
    ).hexdigest()
    if cache_key in _response_cache:
        logger.debug(f"Respuesta obtenida de caché: {request.platform}/{request.username}")
        return _response_cache[cache_key]

# ... do work ...

# Guardar en caché si está habilitado
if request.use_cache:
    if len(_response_cache) >= _cache_max_size:
        _response_cache.popitem(last=False)
    _response_cache[cache_key] = response
```

**After:**
```python
# api/routes.py
from ..utils.cache_manager import get_cache
from ..utils.cache_helpers import generate_cache_key

cache = get_cache()
cache_key = generate_cache_key("extract_profile", request.platform, request.username)

# Verificar caché
if request.use_cache:
    cached_response = cache.get(cache_key)
    if cached_response:
        logger.debug(f"Respuesta obtenida de caché: {request.platform}/{request.username}")
        return cached_response

# ... do work ...

# Guardar en caché
if request.use_cache:
    cache.set(cache_key, response)
```

**Before:**
```python
# services/content_generator.py
# Verificar caché
if use_cache and self._cache is not None:
    cache_key = hashlib.md5(
        f"instagram_{self.identity.profile_id}_{topic}_{style}_{use_lora}".encode()
    ).hexdigest()
    if cache_key in self._cache:
        self.logger.debug("Post de Instagram obtenido de caché")
        cached = self._cache[cache_key]
        cached.generated_at = datetime.now()
        return cached

# ... do work ...

# Guardar en caché
if use_cache and self._cache is not None:
    if len(self._cache) > 500:
        oldest_key = next(iter(self._cache))
        del self._cache[oldest_key]
    self._cache[cache_key] = result
```

**After:**
```python
# services/content_generator.py
from ..utils.cache_manager import ResponseCache
from ..utils.cache_helpers import generate_cache_key

# En __init__:
self._cache = ResponseCache(max_size=500) if enable_cache else None

# En método:
if use_cache and self._cache:
    cache_key = generate_cache_key(
        "instagram", 
        self.identity.profile_id, 
        topic, 
        style, 
        use_lora
    )
    cached = self._cache.get(cache_key)
    if cached:
        self.logger.debug("Post de Instagram obtenido de caché")
        cached.generated_at = datetime.now()
        return cached

# ... do work ...

# Guardar en caché
if use_cache and self._cache:
    self._cache.set(cache_key, result)
```

### Benefits
- **DRY Principle**: Elimina duplicación de lógica de caché
- **Consistency**: Mismo comportamiento en todo el código
- **Maintainability**: Cambios en política de caché en un solo lugar
- **Performance**: Implementación LRU optimizada

---

## 5. Service Lazy Loading Helper

### Problem Identified

Repeated singleton/lazy loading pattern for services:

**Locations Found:**
- `api/routes.py` (lines 53-76)

**Current Pattern:**
```python
def get_analytics_service():
    if not hasattr(get_analytics_service, '_instance'):
        get_analytics_service._instance = AnalyticsService()
    return get_analytics_service._instance
```

### Proposed Helper Function

```python
# core/service_factory.py
from typing import TypeVar, Type, Callable, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceFactory:
    """
    Factory para crear y gestionar instancias singleton de servicios.
    """
    
    def __init__(self):
        self._instances: dict = {}
    
    def get_or_create(
        self,
        service_class: Type[T],
        *args,
        **kwargs
    ) -> T:
        """
        Obtiene instancia existente o crea una nueva.
        
        Args:
            service_class: Clase del servicio
            *args: Argumentos para el constructor
            **kwargs: Keyword arguments para el constructor
            
        Returns:
            Instancia del servicio
        """
        service_name = service_class.__name__
        
        if service_name not in self._instances:
            logger.info(f"Creating new instance of {service_name}")
            self._instances[service_name] = service_class(*args, **kwargs)
        else:
            logger.debug(f"Reusing existing instance of {service_name}")
        
        return self._instances[service_name]
    
    def create_factory_function(
        self,
        service_class: Type[T],
        *args,
        **kwargs
    ) -> Callable[[], T]:
        """
        Crea una función factory para un servicio específico.
        
        Args:
            service_class: Clase del servicio
            *args: Argumentos para el constructor
            **kwargs: Keyword arguments para el constructor
            
        Returns:
            Función que retorna instancia del servicio
        """
        def factory():
            return self.get_or_create(service_class, *args, **kwargs)
        
        factory.__name__ = f"get_{service_class.__name__.lower()}"
        return factory
    
    def reset(self, service_name: Optional[str] = None):
        """
        Resetea instancias (útil para testing).
        
        Args:
            service_name: Nombre del servicio a resetear, o None para todos
        """
        if service_name:
            if service_name in self._instances:
                del self._instances[service_name]
                logger.info(f"Reset instance of {service_name}")
        else:
            self._instances.clear()
            logger.info("Reset all service instances")

# Instancia global
_factory = ServiceFactory()

def get_service_factory() -> ServiceFactory:
    """Obtiene la instancia global del factory."""
    return _factory

def create_service_getter(service_class: Type[T], *args, **kwargs) -> Callable[[], T]:
    """
    Helper para crear función getter de servicio.
    
    Usage:
        get_analytics_service = create_service_getter(AnalyticsService)
    """
    return _factory.create_factory_function(service_class, *args, **kwargs)
```

### Integration Example

**Before:**
```python
# api/routes.py
def get_analytics_service():
    if not hasattr(get_analytics_service, '_instance'):
        get_analytics_service._instance = AnalyticsService()
    return get_analytics_service._instance

def get_export_service():
    if not hasattr(get_export_service, '_instance'):
        get_export_service._instance = ExportService()
    return get_export_service._instance

def get_versioning_service():
    if not hasattr(get_versioning_service, '_instance'):
        get_versioning_service._instance = VersioningService()
    return get_versioning_service._instance

# ... más funciones similares
```

**After:**
```python
# api/routes.py
from ..core.service_factory import create_service_getter

get_analytics_service = create_service_getter(AnalyticsService)
get_export_service = create_service_getter(ExportService)
get_versioning_service = create_service_getter(VersioningService)
get_batch_service = create_service_getter(BatchService)
get_search_service = create_service_getter(SearchService)
```

### Benefits
- **DRY**: Elimina código repetitivo
- **Consistency**: Mismo patrón para todos los servicios
- **Testability**: Fácil resetear instancias en tests
- **Maintainability**: Cambios en patrón singleton en un solo lugar

---

## 6. Validation Helper

### Problem Identified

Repeated validation patterns for checking None, empty values, and platform/content type validation:

**Locations Found:**
- `api/routes.py` (lines 96-101, 208-212, 300-307)
- `core/base_service.py` (line 51-54)

**Current Pattern:**
```python
if data is None:
    raise ValueError(f"{field_name} cannot be None")

if not any([request.tiktok_username, request.instagram_username, request.youtube_channel_id]):
    raise HTTPException(...)

try:
    platform = Platform(request.platform)
    content_type = ContentType(request.content_type)
except ValueError as e:
    raise HTTPException(...)
```

### Proposed Helper Function

```python
# utils/validation_helpers.py
from typing import Any, Optional, List, Type, Union
from enum import Enum
from fastapi import HTTPException, status
from .exception_helpers import validation_error

def validate_not_none(value: Any, field_name: str) -> None:
    """
    Valida que un valor no sea None.
    
    Args:
        value: Valor a validar
        field_name: Nombre del campo para mensaje de error
        
    Raises:
        ValueError: Si el valor es None
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be None")

def validate_not_empty(value: Union[str, List, dict], field_name: str) -> None:
    """
    Valida que un valor no esté vacío.
    
    Args:
        value: Valor a validar (string, lista, o dict)
        field_name: Nombre del campo
        
    Raises:
        ValueError: Si el valor está vacío
    """
    if not value:
        raise ValueError(f"{field_name} cannot be empty")

def validate_at_least_one(*values: Any, field_names: List[str], message: str = None) -> None:
    """
    Valida que al menos uno de los valores no sea None/vacío.
    
    Args:
        *values: Valores a validar
        field_names: Nombres de los campos
        message: Mensaje de error personalizado
        
    Raises:
        HTTPException: Si todos los valores son None/vacíos
    """
    if not any(v for v in values):
        if message:
            raise validation_error(message)
        else:
            field_list = ", ".join(field_names)
            raise validation_error(f"Al menos uno de los siguientes debe ser proporcionado: {field_list}")

def validate_enum(
    value: str,
    enum_class: Type[Enum],
    field_name: str = "value"
) -> Enum:
    """
    Valida y convierte un string a un valor de Enum.
    
    Args:
        value: String a validar
        enum_class: Clase Enum
        field_name: Nombre del campo para mensaje de error
        
    Returns:
        Valor del Enum
        
    Raises:
        HTTPException: Si el valor no es válido
    """
    try:
        return enum_class(value)
    except ValueError:
        valid_values = [e.value for e in enum_class]
        raise validation_error(
            f"{field_name} debe ser uno de: {', '.join(valid_values)}",
            field=field_name
        )

def validate_platform(platform_str: str) -> Any:  # Platform enum
    """
    Valida y convierte string a Platform enum.
    
    Args:
        platform_str: String de plataforma
        
    Returns:
        Platform enum
        
    Raises:
        HTTPException: Si la plataforma no es válida
    """
    from ..core.models import Platform
    return validate_enum(platform_str, Platform, "platform")

def validate_content_type(content_type_str: str) -> Any:  # ContentType enum
    """
    Valida y convierte string a ContentType enum.
    
    Args:
        content_type_str: String de tipo de contenido
        
    Returns:
        ContentType enum
        
    Raises:
        HTTPException: Si el tipo no es válido
    """
    from ..core.models import ContentType
    return validate_enum(content_type_str, ContentType, "content_type")
```

### Integration Example

**Before:**
```python
# api/routes.py
if not any([request.tiktok_username, request.instagram_username, request.youtube_channel_id]):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Al menos un perfil debe ser proporcionado"
    )
```

**After:**
```python
# api/routes.py
from ..utils.validation_helpers import validate_at_least_one

validate_at_least_one(
    request.tiktok_username,
    request.instagram_username,
    request.youtube_channel_id,
    field_names=["tiktok_username", "instagram_username", "youtube_channel_id"],
    message="Al menos un perfil debe ser proporcionado"
)
```

**Before:**
```python
# api/routes.py
try:
    platform = Platform(request.platform)
    content_type = ContentType(request.content_type)
except ValueError as e:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Plataforma o tipo de contenido inválido: {e}"
    )
```

**After:**
```python
# api/routes.py
from ..utils.validation_helpers import validate_platform, validate_content_type

platform = validate_platform(request.platform)
content_type = validate_content_type(request.content_type)
```

**Before:**
```python
# core/base_service.py
def _validate_input(self, data: Any, field_name: str) -> None:
    if data is None:
        raise ValueError(f"{field_name} cannot be None")
```

**After:**
```python
# core/base_service.py
from ..utils.validation_helpers import validate_not_none

def _validate_input(self, data: Any, field_name: str) -> None:
    validate_not_none(data, field_name)
```

### Benefits
- **Consistency**: Validaciones uniformes en todo el código
- **Reusability**: Funciones reutilizables para validaciones comunes
- **Error Messages**: Mensajes de error consistentes y claros
- **Type Safety**: Mejor manejo de tipos y Enums

---

## 7. Summary of Benefits

### Code Quality Improvements

1. **Reduced Duplication**: Elimina ~200+ lines of repetitive code
2. **Consistency**: Comportamiento uniforme en toda la aplicación
3. **Maintainability**: Cambios futuros más fáciles de implementar
4. **Testability**: Helpers más fáciles de testear individualmente
5. **Readability**: Código más claro y autodocumentado

### Performance Benefits

1. **Cache Management**: Implementación LRU optimizada
2. **Lazy Loading**: Servicios solo se crean cuando se necesitan
3. **Reduced Overhead**: Menos código duplicado = menos memoria

### Developer Experience

1. **Faster Development**: Menos código que escribir
2. **Fewer Bugs**: Lógica centralizada = menos errores
3. **Better Documentation**: Helpers bien documentados
4. **Easier Onboarding**: Patrones claros y consistentes

---

## 8. Implementation Plan

### Phase 1: Core Helpers (Week 1)
1. Create `utils/cache_helpers.py`
2. Create `api/exception_helpers.py`
3. Create `api/response_helpers.py`

### Phase 2: Cache Management (Week 1-2)
1. Create `utils/cache_manager.py`
2. Refactor `api/routes.py` to use new cache helpers
3. Refactor `services/content_generator.py` to use new cache helpers

### Phase 3: Service Factory (Week 2)
1. Create `core/service_factory.py`
2. Refactor service getters in `api/routes.py`

### Phase 4: Validation (Week 2-3)
1. Create `utils/validation_helpers.py`
2. Refactor validation code across codebase

### Phase 5: Testing & Documentation (Week 3)
1. Write unit tests for all helpers
2. Update API documentation
3. Create migration guide

---

## 9. Migration Checklist

- [ ] Create helper modules
- [ ] Write unit tests for helpers
- [ ] Refactor `api/routes.py`
- [ ] Refactor `services/content_generator.py`
- [ ] Refactor `services/profile_extractor.py`
- [ ] Refactor `services/identity_analyzer.py`
- [ ] Update `api/decorators.py`
- [ ] Update `core/base_service.py`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Code review
- [ ] Deploy to staging
- [ ] Monitor for issues
- [ ] Deploy to production

---

## Conclusion

These helper functions will significantly improve code maintainability, reduce duplication, and make future updates easier. The refactoring is incremental and can be done gradually without breaking existing functionality.








