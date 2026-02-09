"""
Middleware adicionales para Robot Movement AI v2.0
Incluye request ID, timing, y CORS
"""

import time
import uuid
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False


class RequestIDMiddleware:
    """Middleware para agregar Request ID a cada petición"""
    
    def __init__(self, app, header_name: str = "X-Request-ID"):
        """
        Inicializar middleware
        
        Args:
            app: Aplicación FastAPI
            header_name: Nombre del header para Request ID
        """
        self.app = app
        self.header_name = header_name
    
    async def __call__(self, scope, receive, send):
        """Procesar petición"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Generar o obtener Request ID
        request = Request(scope, receive)
        request_id = request.headers.get(self.header_name.lower())
        
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Agregar a scope para acceso en handlers
        scope["request_id"] = request_id
        
        # Crear wrapper para agregar header en respuesta
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message["headers"].append([
                    self.header_name.encode(),
                    request_id.encode()
                ])
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


class TimingMiddleware:
    """Middleware para medir tiempo de procesamiento de peticiones"""
    
    def __init__(self, app, header_name: str = "X-Process-Time"):
        """
        Inicializar middleware
        
        Args:
            app: Aplicación FastAPI
            header_name: Nombre del header para tiempo de procesamiento
        """
        self.app = app
        self.header_name = header_name
    
    async def __call__(self, scope, receive, send):
        """Procesar petición"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        # Crear wrapper para medir tiempo
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                process_time = time.time() - start_time
                message["headers"].append([
                    self.header_name.encode(),
                    f"{process_time:.4f}".encode()
                ])
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


class CompressionMiddleware:
    """Middleware para compresión de respuestas (gzip)"""
    
    def __init__(self, app, minimum_size: int = 500):
        """
        Inicializar middleware
        
        Args:
            app: Aplicación FastAPI
            minimum_size: Tamaño mínimo para comprimir (bytes)
        """
        self.app = app
        self.minimum_size = minimum_size
    
    async def __call__(self, scope, receive, send):
        """Procesar petición"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Verificar si el cliente acepta gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        supports_gzip = "gzip" in accept_encoding.lower()
        
        if not supports_gzip:
            await self.app(scope, receive, send)
            return
        
        # Buffer para respuesta
        response_body = b""
        status_code = 200
        headers = []
        
        async def send_wrapper(message):
            nonlocal response_body, status_code, headers
            
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message["headers"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
                if not message.get("more_body", False):
                    # Comprimir si es necesario
                    if len(response_body) >= self.minimum_size:
                        import gzip
                        compressed = gzip.compress(response_body)
                        headers.append([b"content-encoding", b"gzip"])
                        headers.append([b"content-length", str(len(compressed)).encode()])
                        await send({
                            "type": "http.response.start",
                            "status": status_code,
                            "headers": headers
                        })
                        await send({
                            "type": "http.response.body",
                            "body": compressed
                        })
                    else:
                        await send({
                            "type": "http.response.start",
                            "status": status_code,
                            "headers": headers
                        })
                        await send({
                            "type": "http.response.body",
                            "body": response_body
                        })
            else:
                await send(message)
        
        await self.app(scope, receive, send_wrapper)


def setup_cors(app, **kwargs):
    """
    Configurar CORS middleware
    
    Args:
        app: Aplicación FastAPI
        **kwargs: Argumentos adicionales para CORSMiddleware
    """
    defaults = {
        "allow_origins": ["*"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    defaults.update(kwargs)
    
    app.add_middleware(CORSMiddleware, **defaults)


def setup_middleware(app, enable_request_id: bool = True, enable_timing: bool = True):
    """
    Configurar todos los middlewares
    
    Args:
        app: Aplicación FastAPI
        enable_request_id: Habilitar Request ID middleware
        enable_timing: Habilitar Timing middleware
    """
    if enable_request_id:
        app.add_middleware(RequestIDMiddleware)
    
    if enable_timing:
        app.add_middleware(TimingMiddleware)
    
    # CORS siempre habilitado
    setup_cors(app)
