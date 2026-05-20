"""
API Gateway Integration - Integración con AWS API Gateway
==========================================================

Utilidades para integrar con AWS API Gateway y otros gateways.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIGatewayMiddleware:
    """
    Middleware para AWS API Gateway.
    
    Maneja:
    - Request/Response transformation
    - CORS headers
    - Rate limiting headers
    - Error formatting
    """
    
    @staticmethod
    def is_api_gateway_request(request: Request) -> bool:
        """Verificar si el request viene de API Gateway."""
        return bool(
            request.headers.get("x-apigw-request-id") or
            request.headers.get("x-amzn-requestid") or
            os.getenv("AWS_API_GATEWAY_ID")
        )
    
    @staticmethod
    def get_api_gateway_context(request: Request) -> Dict[str, Any]:
        """Obtener contexto de API Gateway."""
        return {
            "request_id": request.headers.get("x-apigw-request-id") or request.headers.get("x-amzn-requestid"),
            "stage": request.headers.get("x-apigw-stage", os.getenv("API_GATEWAY_STAGE", "prod")),
            "api_id": os.getenv("AWS_API_GATEWAY_ID"),
            "account_id": os.getenv("AWS_ACCOUNT_ID"),
            "region": os.getenv("AWS_REGION", "us-east-1")
        }
    
    @staticmethod
    def format_api_gateway_response(data: Any, status_code: int = 200) -> Dict[str, Any]:
        """
        Formatear respuesta para API Gateway.
        
        Args:
            data: Datos de respuesta.
            status_code: Código de estado HTTP.
        
        Returns:
            Respuesta formateada para API Gateway.
        """
        return {
            "statusCode": status_code,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            },
            "body": data if isinstance(data, str) else str(data)
        }
    
    @staticmethod
    def format_error_response(error: Exception, status_code: int = 500) -> Dict[str, Any]:
        """
        Formatear error para API Gateway.
        
        Args:
            error: Excepción.
            status_code: Código de estado HTTP.
        
        Returns:
            Error formateado para API Gateway.
        """
        return {
            "statusCode": status_code,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "error": error.__class__.__name__,
                "message": str(error)
            }
        }


class RateLimitHeaders:
    """Utilidades para headers de rate limiting en API Gateway."""
    
    @staticmethod
    def add_rate_limit_headers(
        response: Response,
        limit: int,
        remaining: int,
        reset_time: int
    ) -> None:
        """
        Agregar headers de rate limiting.
        
        Args:
            response: Response de FastAPI.
            limit: Límite de requests.
            remaining: Requests restantes.
            reset_time: Tiempo de reset (timestamp).
        """
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)


class CORSHeaders:
    """Utilidades para headers CORS en API Gateway."""
    
    @staticmethod
    def add_cors_headers(response: Response, allowed_origins: Optional[List[str]] = None) -> None:
        """
        Agregar headers CORS.
        
        Args:
            response: Response de FastAPI.
            allowed_origins: Orígenes permitidos (None = todos).
        """
        if allowed_origins is None:
            allowed_origins = ["*"]
        
        origin = "*" if "*" in allowed_origins else ",".join(allowed_origins)
        
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "3600"


def setup_api_gateway_middleware(app) -> None:
    """
    Configurar middleware para API Gateway.
    
    Args:
        app: Aplicación FastAPI.
    """
    from fastapi.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp
    
    class APIGatewayMiddlewareClass(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Verificar si es request de API Gateway
            is_gateway = APIGatewayMiddleware.is_api_gateway_request(request)
            
            if is_gateway:
                # Agregar contexto a request state
                request.state.api_gateway = APIGatewayMiddleware.get_api_gateway_context(request)
            
            # Procesar request
            response = await call_next(request)
            
            # Agregar headers CORS si es API Gateway
            if is_gateway:
                CORSHeaders.add_cors_headers(response)
            
            return response
    
    app.add_middleware(APIGatewayMiddlewareClass)
    logger.info("API Gateway middleware configured")


# Utilidades para Kong API Gateway
class KongGateway:
    """Utilidades para Kong API Gateway."""
    
    @staticmethod
    def is_kong_request(request: Request) -> bool:
        """Verificar si el request viene de Kong."""
        return bool(request.headers.get("x-kong-request-id"))
    
    @staticmethod
    def get_kong_context(request: Request) -> Dict[str, Any]:
        """Obtener contexto de Kong."""
        return {
            "request_id": request.headers.get("x-kong-request-id"),
            "consumer_id": request.headers.get("x-consumer-id"),
            "consumer_username": request.headers.get("x-consumer-username"),
            "credential_identifier": request.headers.get("x-credential-identifier")
        }




