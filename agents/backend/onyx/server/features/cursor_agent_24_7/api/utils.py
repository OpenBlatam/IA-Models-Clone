"""
API Utilities - Utilidades para rutas de la API
===============================================

Utilidades para simplificar y hacer consistente el código de las rutas.
"""

import logging
from typing import Callable, Any, Optional
from functools import wraps
from fastapi import Request, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..core.agent import CursorAgent
from ..core.error_handling import safe_async_call

logger = logging.getLogger(__name__)


def get_agent(request: Request) -> CursorAgent:
    """
    Dependencia para obtener el agente desde el estado de la aplicación.
    
    Args:
        request: Request de FastAPI.
    
    Returns:
        Instancia del agente.
    
    Raises:
        HTTPException: Si el agente no está disponible.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not available"
        )
    return agent


AgentDep = Depends(get_agent)


def handle_route_errors(
    operation: str = "operation",
    default_status_code: int = 500
):
    """
    Decorador para manejo de errores en rutas de la API.
    
    Args:
        operation: Descripción de la operación (para logging).
        default_status_code: Código HTTP por defecto para errores.
    
    Returns:
        Decorador de función.
    
    Example:
        @router.post("/start")
        @handle_route_errors("starting agent")
        async def start_agent(agent: CursorAgent = AgentDep):
            await agent.start()
            return {"status": "started"}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ValueError as e:
                logger.warning(f"Validation error in {operation}: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in {operation}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=default_status_code,
                    detail=str(e)
                )
        return wrapper
    return decorator


def create_success_response(
    status: str,
    message: str,
    data: Optional[dict] = None
) -> dict:
    """
    Crear respuesta de éxito consistente.
    
    Args:
        status: Estado de la operación.
        message: Mensaje descriptivo.
        data: Datos adicionales (opcional).
    
    Returns:
        Diccionario con la respuesta.
    """
    response = {
        "status": status,
        "message": message
    }
    if data:
        response.update(data)
    return response


def create_error_response(
    detail: str,
    status_code: int = 500,
    error_type: Optional[str] = None
) -> JSONResponse:
    """
    Crear respuesta de error consistente.
    
    Args:
        detail: Mensaje de error.
        status_code: Código HTTP.
        error_type: Tipo de error (opcional).
    
    Returns:
        JSONResponse con el error.
    """
    content = {
        "error": detail,
        "status_code": status_code
    }
    if error_type:
        content["error_type"] = error_type
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )


async def safe_route_call(
    operation: str,
    func: Callable,
    *args,
    default_status_code: int = 500,
    **kwargs
) -> Any:
    """
    Ejecutar llamada de ruta de forma segura con manejo de errores.
    
    Args:
        operation: Descripción de la operación.
        func: Función async a ejecutar.
        *args: Argumentos posicionales.
        default_status_code: Código HTTP por defecto para errores.
        **kwargs: Argumentos con nombre.
    
    Returns:
        Resultado de la función.
    
    Raises:
        HTTPException: Si hay error en la operación.
    """
    try:
        return await func(*args, **kwargs)
    except ValueError as e:
        logger.warning(f"Validation error in {operation}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in {operation}: {e}", exc_info=True)
        raise HTTPException(
            status_code=default_status_code,
            detail=str(e)
        )




