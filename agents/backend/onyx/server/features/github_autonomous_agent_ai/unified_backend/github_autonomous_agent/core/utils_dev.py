"""
Utilidades de desarrollo y debugging.
"""

import json
import time
from typing import Any, Dict, Optional
from datetime import datetime
from functools import wraps
from config.logging_config import get_logger

logger = get_logger(__name__)


def timing_decorator(func):
    """
    Decorador para medir tiempo de ejecución con mejoras.
    
    Args:
        func: Función a decorar (puede ser sync o async)
        
    Returns:
        Función decorada con timing
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            logger.debug(f"⏱️  Iniciando {func.__name__} (async)")
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(
                f"✅ {func.__name__} completado en {duration:.3f}s "
                f"({format_duration(duration)})"
            )
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(
                f"❌ {func.__name__} falló después de {duration:.3f}s "
                f"({format_duration(duration)}): {type(e).__name__}: {e}",
                exc_info=True
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            logger.debug(f"⏱️  Iniciando {func.__name__} (sync)")
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(
                f"✅ {func.__name__} completado en {duration:.3f}s "
                f"({format_duration(duration)})"
            )
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(
                f"❌ {func.__name__} falló después de {duration:.3f}s "
                f"({format_duration(duration)}): {type(e).__name__}: {e}",
                exc_info=True
            )
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def log_request_details(
    request_data: Dict[str, Any],
    context: Optional[str] = None
) -> None:
    """
    Loggear detalles de un request para debugging con validaciones.
    
    Args:
        request_data: Datos del request (debe ser diccionario)
        context: Contexto adicional (opcional, debe ser string si se proporciona)
        
    Raises:
        ValueError: Si request_data no es un diccionario
    """
    # Validaciones
    if not isinstance(request_data, dict):
        raise ValueError(f"request_data debe ser un diccionario, recibido: {type(request_data)}")
    
    if context is not None:
        if not isinstance(context, str):
            raise ValueError(f"context debe ser un string si se proporciona, recibido: {type(context)}")
        context = context.strip()
    
    try:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "data": request_data
        }
        logger.debug(f"📋 Request details: {json.dumps(log_data, indent=2, default=str)}")
    except Exception as e:
        logger.warning(f"Error al loggear detalles de request: {e}", exc_info=True)


def format_error_for_logging(
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Formatear error para logging estructurado con validaciones.
    
    Args:
        error: Excepción (debe ser Exception o subclase)
        context: Contexto adicional (opcional, debe ser diccionario si se proporciona)
        
    Returns:
        Diccionario con información del error
        
    Raises:
        ValueError: Si error no es una excepción o context no es un diccionario
    """
    # Validaciones
    if not isinstance(error, Exception):
        raise ValueError(f"error debe ser una Exception, recibido: {type(error)}")
    
    if context is not None:
        if not isinstance(context, dict):
            raise ValueError(f"context debe ser un diccionario si se proporciona, recibido: {type(context)}")
    
    try:
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            error_info["context"] = context
        
        if hasattr(error, "__traceback__"):
            import traceback
            error_info["traceback"] = traceback.format_exception(
                type(error), error, error.__traceback__
            )
        
        # Agregar información adicional si está disponible
        if hasattr(error, "to_dict"):
            try:
                error_info["error_details"] = error.to_dict()
            except Exception:
                pass
        
        return error_info
    except Exception as e:
        logger.warning(f"Error al formatear error para logging: {e}", exc_info=True)
        return {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
            "formatting_error": str(e)
        }


def safe_json_parse(data: str, default: Any = None) -> Any:
    """
    Parsear JSON de forma segura con validaciones.
    
    Args:
        data: String JSON (debe ser string)
        default: Valor por defecto si falla
        
    Returns:
        Objeto parseado o default
        
    Raises:
        ValueError: Si data no es un string
    """
    # Validación
    if not isinstance(data, str):
        if data is None:
            logger.debug("safe_json_parse recibió None, retornando default")
            return default
        raise ValueError(f"data debe ser un string, recibido: {type(data)}")
    
    if not data.strip():
        logger.debug("safe_json_parse recibió string vacío, retornando default")
        return default
    
    try:
        result = json.loads(data)
        logger.debug("JSON parseado exitosamente")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Error parsing JSON (JSONDecodeError): {e} (data: {data[:100]}...)")
        return default
    except TypeError as e:
        logger.warning(f"Error parsing JSON (TypeError): {e}")
        return default
    except Exception as e:
        logger.warning(f"Error inesperado parsing JSON: {e}", exc_info=True)
        return default


def format_duration(seconds: float) -> str:
    """
    Formatear duración en formato legible con validaciones.
    
    Args:
        seconds: Segundos (debe ser número no negativo)
        
    Returns:
        String formateado (ej: "1h 23m 45s", "123ms", "45.2s")
        
    Raises:
        ValueError: Si seconds es inválido
    """
    # Validación
    if not isinstance(seconds, (int, float)):
        raise ValueError(f"seconds debe ser un número, recibido: {type(seconds)}")
    
    if seconds < 0:
        raise ValueError(f"seconds debe ser no negativo, recibido: {seconds}")
    
    try:
        if seconds < 0.001:  # Menos de 1ms
            return "<1ms"
        elif seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
    except Exception as e:
        logger.warning(f"Error al formatear duración: {e}", exc_info=True)
        return f"{seconds}s"


def get_memory_usage() -> Dict[str, Any]:
    """
    Obtener uso de memoria del proceso con mejor manejo de errores.
    
    Returns:
        Diccionario con información de memoria o error si no se puede obtener
    """
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        result = {
            "rss_bytes": memory_info.rss,  # Resident Set Size
            "vms_bytes": memory_info.vms,  # Virtual Memory Size
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug(
            f"Uso de memoria: RSS={result['rss_mb']}MB, "
            f"VMS={result['vms_mb']}MB, {result['percent']}%"
        )
        
        return result
    except ImportError:
        logger.debug("psutil no disponible para obtener uso de memoria")
        return {
            "error": "psutil not available",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.warning(f"Error getting memory usage: {e}", exc_info=True)
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def print_service_status(services: Dict[str, Any]) -> None:
    """
    Imprimir estado de servicios de forma legible con validaciones.
    
    Args:
        services: Diccionario con estado de servicios (debe ser diccionario)
        
    Raises:
        ValueError: Si services no es un diccionario
    """
    # Validación
    if not isinstance(services, dict):
        raise ValueError(f"services debe ser un diccionario, recibido: {type(services)}")
    
    if not services:
        logger.info("📊 Estado de Servicios: (ningún servicio registrado)")
        return
    
    logger.info(f"📊 Estado de Servicios ({len(services)} servicios):")
    
    ok_count = 0
    warning_count = 0
    error_count = 0
    
    for service_name, status in services.items():
        if not service_name or not isinstance(service_name, str):
            logger.warning(f"Nombre de servicio inválido: {service_name}")
            continue
        
        try:
            if isinstance(status, dict):
                status_str = status.get("status", "unknown")
                message = status.get("message", "")
                
                if status_str == "ok":
                    icon = "✅"
                    ok_count += 1
                elif status_str == "warning":
                    icon = "⚠️"
                    warning_count += 1
                else:
                    icon = "❌"
                    error_count += 1
                
                logger.info(f"  {icon} {service_name}: {status_str} - {message}")
            else:
                if status:
                    icon = "✅"
                    ok_count += 1
                    status_str = "ok"
                else:
                    icon = "❌"
                    error_count += 1
                    status_str = "error"
                
                logger.info(f"  {icon} {service_name}: {status_str}")
        except Exception as e:
            logger.warning(f"Error al procesar estado de servicio '{service_name}': {e}", exc_info=True)
            logger.info(f"  ❌ {service_name}: error al procesar estado")
            error_count += 1
    
    # Resumen
    logger.info(
        f"📊 Resumen: ✅ {ok_count} OK, ⚠️  {warning_count} Warning, ❌ {error_count} Error"
    )

