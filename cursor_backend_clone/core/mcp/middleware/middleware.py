"""
MCP Middleware - Middleware avanzado para el servidor MCP
=========================================================

Middleware para logging estructurado, compresión, y tracking de requests.
"""

import uuid
import time
import logging
import gzip
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar Request ID a cada request"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            logger.info(
                f"Request {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4),
                    "client_ip": request.client.host if request.client else "unknown"
                }
            )
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "process_time": round(process_time, 4),
                    "error": str(e)
                },
                exc_info=True
            )
            raise


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware para comprimir respuestas grandes"""
    
    def __init__(self, app: ASGIApp, min_size: int = 1024):
        super().__init__(app)
        self.min_size = min_size
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        accept_encoding = request.headers.get("Accept-Encoding", "")
        
        if "gzip" not in accept_encoding.lower():
            return response
        
        if isinstance(response, StreamingResponse):
            return response
        
        if response.status_code >= 300:
            return response
        
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        if len(body) < self.min_size:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        compressed_body = gzip.compress(body, compresslevel=6)
        
        headers = dict(response.headers)
        headers["Content-Encoding"] = "gzip"
        headers["Content-Length"] = str(len(compressed_body))
        headers["Vary"] = "Accept-Encoding"
        
        return Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type
        )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar headers de seguridad"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

