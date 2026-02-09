"""
API Gateway Middleware
Integración con API Gateways (Kong, AWS API Gateway, Traefik)
"""

import logging
from typing import Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIGatewayMiddleware(BaseHTTPMiddleware):
    """
    Middleware para integración con API Gateways
    Maneja rate limiting, request transformation, y routing
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.gateway_type = kwargs.get('gateway_type', 'aws')  # aws, kong, traefik
        self.enable_rate_limiting = kwargs.get('enable_rate_limiting', True)
        self.enable_request_transformation = kwargs.get('enable_request_transformation', False)
        self.api_key_header = kwargs.get('api_key_header', 'x-api-key')
        self.client_id_header = kwargs.get('client_id_header', 'x-client-id')
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request con API Gateway logic"""
        # Extraer información del API Gateway
        gateway_info = self._extract_gateway_info(request)
        request.state.gateway_info = gateway_info
        
        # Validar API key si está presente
        api_key = request.headers.get(self.api_key_header)
        if api_key:
            request.state.api_key = api_key
            # Aquí podrías validar el API key contra una base de datos
        
        # Extraer client ID
        client_id = request.headers.get(self.client_id_header)
        if client_id:
            request.state.client_id = client_id
        
        # Transformar request si es necesario
        if self.enable_request_transformation:
            request = await self._transform_request(request)
        
        # Procesar request
        response = await call_next(request)
        
        # Agregar headers de API Gateway
        response.headers["X-Gateway-Type"] = self.gateway_type
        if gateway_info.get("request_id"):
            response.headers["X-Request-ID"] = gateway_info["request_id"]
        if gateway_info.get("stage"):
            response.headers["X-API-Stage"] = gateway_info["stage"]
        
        return response
    
    def _extract_gateway_info(self, request: Request) -> Dict[str, Any]:
        """Extrae información del API Gateway desde headers"""
        info = {}
        
        if self.gateway_type == "aws":
            # AWS API Gateway headers
            info["request_id"] = request.headers.get("x-amzn-requestid")
            info["request_time"] = request.headers.get("x-amzn-request-time")
            info["stage"] = request.headers.get("x-amzn-stage")
            info["trace_id"] = request.headers.get("x-amzn-trace-id")
            info["api_key_id"] = request.headers.get("x-api-key-id")
            info["usage_plan_id"] = request.headers.get("x-usage-plan-id")
        
        elif self.gateway_type == "kong":
            # Kong headers
            info["request_id"] = request.headers.get("x-kong-request-id")
            info["upstream_latency"] = request.headers.get("x-kong-upstream-latency")
            info["proxy_latency"] = request.headers.get("x-kong-proxy-latency")
            info["consumer_id"] = request.headers.get("x-consumer-id")
            info["consumer_username"] = request.headers.get("x-consumer-username")
        
        elif self.gateway_type == "traefik":
            # Traefik headers
            info["request_id"] = request.headers.get("x-request-id")
            info["forwarded_for"] = request.headers.get("x-forwarded-for")
            info["forwarded_proto"] = request.headers.get("x-forwarded-proto")
            info["forwarded_host"] = request.headers.get("x-forwarded-host")
        
        return info
    
    async def _transform_request(self, request: Request) -> Request:
        """Transforma el request según reglas del API Gateway"""
        # Aquí puedes implementar transformaciones específicas
        # Por ejemplo, agregar headers, modificar body, etc.
        return request















