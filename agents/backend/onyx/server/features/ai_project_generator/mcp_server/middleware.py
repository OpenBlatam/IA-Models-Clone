"""
MCP Middleware - Middleware para el servidor MCP
=================================================

Middleware para logging, CORS y otras funcionalidades transversales
del servidor MCP.
"""

import time
import logging
from typing import Callable, Optional, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class MCPLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para logging de requests y responses.
    
    Registra información sobre cada request incluyendo:
    - Método HTTP y path
    - Tiempo de procesamiento
    - Código de estado de respuesta
    - Errores si ocurren
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Callable]
    ) -> Response:
        """
        Procesa el request y registra información.
        
        Args:
            request: Request de FastAPI.
            call_next: Función para llamar al siguiente middleware/handler.
        
        Returns:
            Response de FastAPI.
        
        Raises:
            Exception: Re-lanza cualquier excepción que ocurra durante el procesamiento.
        """
        start_time = time.time()
        client_host = None
        
        try:
            if request.client:
                client_host = request.client.host
        except Exception:
            pass
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": client_host,
                "query_params": str(request.query_params),
            }
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} ({process_time:.3f}s)",
                extra={
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "path": request.url.path,
                    "method": request.method,
                }
            )
            
            # Agregar header de tiempo de procesamiento
            try:
                response.headers["X-Process-Time"] = f"{process_time:.3f}"
            except Exception:
                pass  # Si no se puede agregar header, continuar
            
            return response
        
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error processing request: {e} ({process_time:.3f}s)",
                exc_info=True,
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time": process_time,
                    "path": request.url.path,
                    "method": request.method,
                    "client": client_host,
                }
            )
            raise


class MCPCORSMiddleware(BaseHTTPMiddleware):
    """
    Middleware para CORS (Cross-Origin Resource Sharing).
    
    Permite configurar orígenes permitidos y headers CORS
    para habilitar requests desde diferentes dominios.
    """
    
    def __init__(
        self,
        app: Callable,
        allow_origins: Optional[List[str]] = None
    ) -> None:
        """
        Inicializar middleware CORS.
        
        Args:
            app: Aplicación FastAPI.
            allow_origins: Lista de orígenes permitidos (default: ["*"]).
                Si contiene "*", permite todos los orígenes.
        """
        super().__init__(app)
        self.allow_origins: List[str] = allow_origins or ["*"]
        logger.debug(f"CORS middleware initialized with origins: {self.allow_origins}")
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Callable]
    ) -> Response:
        """
        Procesa el request y agrega headers CORS.
        
        Args:
            request: Request de FastAPI.
            call_next: Función para llamar al siguiente middleware/handler.
        
        Returns:
            Response de FastAPI con headers CORS agregados.
        """
        # Manejar preflight OPTIONS request
        if request.method == "OPTIONS":
            response = Response()
        else:
            response = await call_next(request)
        
        # Obtener origen del request
        origin = request.headers.get("origin")
        
        # Verificar si el origen está permitido
        if origin and (origin in self.allow_origins or "*" in self.allow_origins):
            try:
                response.headers["Access-Control-Allow-Origin"] = (
                    origin if "*" not in self.allow_origins else "*"
                )
                response.headers["Access-Control-Allow-Methods"] = (
                    "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                )
                response.headers["Access-Control-Allow-Headers"] = (
                    "Authorization, Content-Type, X-Requested-With"
                )
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Max-Age"] = "3600"
            except Exception as e:
                logger.warning(f"Error setting CORS headers: {e}", exc_info=True)
        
        return response
