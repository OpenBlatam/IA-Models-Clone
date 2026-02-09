"""
Integrations - Sistema de Integraciones
========================================

Sistema de integraciones con servicios externos.
"""

import asyncio
import logging
import httpx
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Tipos de integración."""
    WEBHOOK = "webhook"
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"


@dataclass
class Integration:
    """Integración."""
    integration_id: str
    name: str
    integration_type: IntegrationType
    config: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntegrationManager:
    """Gestor de integraciones."""
    
    def __init__(self):
        self.integrations: Dict[str, Integration] = {}
        self.handlers: Dict[str, Callable] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Obtener cliente HTTP."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    def register_integration(self, integration: Integration):
        """Registrar integración."""
        self.integrations[integration.integration_id] = integration
        logger.info(f"Registered integration: {integration.integration_id}")
    
    def register_handler(
        self,
        integration_id: str,
        handler: Callable,
    ):
        """Registrar handler para integración."""
        self.handlers[integration_id] = handler
        logger.debug(f"Registered handler for {integration_id}")
    
    async def call_integration(
        self,
        integration_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Llamar integración.
        
        Args:
            integration_id: ID de la integración
            data: Datos a enviar
        
        Returns:
            Respuesta de la integración
        """
        integration = self.integrations.get(integration_id)
        if not integration:
            raise ValueError(f"Integration not found: {integration_id}")
        
        if not integration.enabled:
            raise ValueError(f"Integration {integration_id} is disabled")
        
        # Actualizar último uso
        integration.last_used = datetime.now()
        
        # Llamar handler personalizado si existe
        if integration_id in self.handlers:
            handler = self.handlers[integration_id]
            if asyncio.iscoroutinefunction(handler):
                return await handler(data, integration.config)
            else:
                return handler(data, integration.config)
        
        # Llamar según tipo
        if integration.integration_type == IntegrationType.WEBHOOK:
            return await self._call_webhook(integration, data)
        elif integration.integration_type == IntegrationType.REST_API:
            return await self._call_rest_api(integration, data)
        else:
            raise ValueError(f"Unsupported integration type: {integration.integration_type}")
    
    async def _call_webhook(
        self,
        integration: Integration,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Llamar webhook."""
        url = integration.config.get("url")
        if not url:
            raise ValueError("Webhook URL not configured")
        
        client = await self._get_http_client()
        
        try:
            response = await client.post(
                url,
                json=data,
                headers=integration.config.get("headers", {}),
            )
            response.raise_for_status()
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": response.json() if response.text else {},
            }
        except Exception as e:
            logger.error(f"Error calling webhook {integration.integration_id}: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _call_rest_api(
        self,
        integration: Integration,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Llamar REST API."""
        base_url = integration.config.get("base_url")
        endpoint = integration.config.get("endpoint", "/")
        method = integration.config.get("method", "POST").upper()
        
        if not base_url:
            raise ValueError("REST API base_url not configured")
        
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        client = await self._get_http_client()
        
        try:
            if method == "GET":
                response = await client.get(
                    url,
                    params=data,
                    headers=integration.config.get("headers", {}),
                )
            elif method == "POST":
                response = await client.post(
                    url,
                    json=data,
                    headers=integration.config.get("headers", {}),
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": response.json() if response.text else {},
            }
        except Exception as e:
            logger.error(f"Error calling REST API {integration.integration_id}: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_integration(self, integration_id: str) -> Optional[Integration]:
        """Obtener integración."""
        return self.integrations.get(integration_id)
    
    def list_integrations(self) -> List[Dict[str, Any]]:
        """Listar integraciones."""
        return [
            {
                "integration_id": i.integration_id,
                "name": i.name,
                "type": i.integration_type.value,
                "enabled": i.enabled,
                "last_used": i.last_used.isoformat() if i.last_used else None,
            }
            for i in self.integrations.values()
        ]
    
    async def close(self):
        """Cerrar recursos."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
































