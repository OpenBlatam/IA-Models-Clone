"""
API Gateway Integration - Integración con API Gateways
======================================================

Integración con API Gateways como Kong, AWS API Gateway, Azure API Management,
Traefik, y NGINX para rate limiting, request transformation, y security filtering.
"""

import logging
import httpx
from typing import Optional, Dict, Any
from enum import Enum

from .microservices_config import get_microservices_config, APIGatewayType

logger = logging.getLogger(__name__)


class APIGatewayClient:
    """Cliente para interactuar con API Gateways"""
    
    def __init__(
        self,
        gateway_type: Optional[APIGatewayType] = None,
        gateway_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        config = get_microservices_config()
        self.gateway_type = gateway_type or config.api_gateway_type
        self.gateway_url = gateway_url or config.api_gateway_url
        self.api_key = api_key or config.api_gateway_key
        
        self.client = httpx.AsyncClient(timeout=30.0) if self.gateway_url else None
    
    async def register_service(
        self,
        service_name: str,
        service_url: str,
        routes: list,
        **kwargs
    ) -> bool:
        """
        Registra un servicio en el API Gateway.
        
        Args:
            service_name: Nombre del servicio
            service_url: URL del servicio
            routes: Lista de rutas a registrar
            **kwargs: Configuración adicional
        
        Returns:
            True si se registró exitosamente
        """
        if not self.gateway_url:
            logger.warning("API Gateway URL not configured")
            return False
        
        try:
            if self.gateway_type == APIGatewayType.KONG:
                return await self._register_kong_service(service_name, service_url, routes, **kwargs)
            elif self.gateway_type == APIGatewayType.AWS_API_GATEWAY:
                return await self._register_aws_service(service_name, service_url, routes, **kwargs)
            elif self.gateway_type == APIGatewayType.AZURE_API_MANAGEMENT:
                return await self._register_azure_service(service_name, service_url, routes, **kwargs)
            else:
                logger.warning(f"API Gateway type {self.gateway_type} not implemented")
                return False
        except Exception as e:
            logger.error(f"Failed to register service in API Gateway: {e}")
            return False
    
    async def _register_kong_service(
        self,
        service_name: str,
        service_url: str,
        routes: list,
        **kwargs
    ) -> bool:
        """Registra servicio en Kong"""
        try:
            # Crear servicio
            service_data = {
                "name": service_name,
                "url": service_url,
                **kwargs
            }
            
            response = await self.client.post(
                f"{self.gateway_url}/services",
                json=service_data,
                headers={"apikey": self.api_key} if self.api_key else {}
            )
            response.raise_for_status()
            
            service_id = response.json().get("id")
            
            # Crear rutas
            for route in routes:
                route_data = {
                    "service": {"id": service_id},
                    "paths": route.get("paths", []),
                    "methods": route.get("methods", ["GET", "POST", "PUT", "DELETE"]),
                    **route.get("config", {})
                }
                
                route_response = await self.client.post(
                    f"{self.gateway_url}/routes",
                    json=route_data,
                    headers={"apikey": self.api_key} if self.api_key else {}
                )
                route_response.raise_for_status()
            
            logger.info(f"Service {service_name} registered in Kong")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register Kong service: {e}")
            return False
    
    async def _register_aws_service(
        self,
        service_name: str,
        service_url: str,
        routes: list,
        **kwargs
    ) -> bool:
        """Registra servicio en AWS API Gateway"""
        # AWS API Gateway requiere AWS SDK (boto3)
        # Esta es una implementación simplificada
        logger.info(f"AWS API Gateway registration for {service_name} (requires boto3)")
        return False
    
    async def _register_azure_service(
        self,
        service_name: str,
        service_url: str,
        routes: list,
        **kwargs
    ) -> bool:
        """Registra servicio en Azure API Management"""
        # Azure API Management requiere Azure SDK
        # Esta es una implementación simplificada
        logger.info(f"Azure API Management registration for {service_name} (requires azure-mgmt-apimanagement)")
        return False
    
    async def configure_rate_limit(
        self,
        service_name: str,
        limit: int,
        window: int = 60
    ) -> bool:
        """Configura rate limiting para un servicio"""
        if not self.gateway_url:
            return False
        
        try:
            if self.gateway_type == APIGatewayType.KONG:
                # Configurar plugin de rate limiting en Kong
                plugin_data = {
                    "name": "rate-limiting",
                    "config": {
                        "minute": limit,
                        "hour": limit * 60,
                    }
                }
                
                response = await self.client.post(
                    f"{self.gateway_url}/services/{service_name}/plugins",
                    json=plugin_data,
                    headers={"apikey": self.api_key} if self.api_key else {}
                )
                response.raise_for_status()
                return True
            else:
                logger.warning(f"Rate limiting not implemented for {self.gateway_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to configure rate limit: {e}")
            return False
    
    async def close(self):
        """Cierra el cliente"""
        if self.client:
            await self.client.aclose()


def get_api_gateway_client() -> APIGatewayClient:
    """Obtiene cliente de API Gateway"""
    return APIGatewayClient()















