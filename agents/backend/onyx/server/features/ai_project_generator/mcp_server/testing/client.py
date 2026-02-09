"""
MCP Test Client - Cliente de testing mejorado
==============================================

Cliente de testing para el servidor MCP con soporte síncrono y asíncrono.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Note: No importar MCPRequest/MCPResponse aquí para evitar imports circulares

logger = logging.getLogger(__name__)


class MCPTestClient:
    """
    Cliente de testing síncrono para MCP.
    
    Facilita testing de endpoints MCP con métodos helper
    y manejo automático de autenticación.
    """
    
    def __init__(self, app, base_url: str = "http://test", default_headers: Optional[Dict[str, str]] = None):
        """
        Inicializar cliente de testing.
        
        Args:
            app: Aplicación FastAPI
            base_url: URL base para testing
            default_headers: Headers por defecto (opcional)
        """
        self.client = TestClient(app, base_url=base_url)
        self.default_headers = default_headers or {}
        self._auth_token: Optional[str] = None
    
    def set_auth_token(self, token: str) -> None:
        """
        Establecer token de autenticación para requests.
        
        Args:
            token: Token JWT
        """
        self._auth_token = token
        self.default_headers["Authorization"] = f"Bearer {token}"
    
    def _get_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Combinar headers por defecto con headers proporcionados"""
        combined = self.default_headers.copy()
        if headers:
            combined.update(headers)
        return combined
    
    def list_resources(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Lista todos los recursos disponibles.
        
        Args:
            headers: Headers adicionales (opcional)
            
        Returns:
            Lista de recursos
        """
        response = self.client.get(
            "/mcp/v1/resources",
            headers=self._get_headers(headers)
        )
        response.raise_for_status()
        return response.json()
    
    def get_resource(self, resource_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Obtiene información de un recurso específico.
        
        Args:
            resource_id: ID del recurso
            headers: Headers adicionales (opcional)
            
        Returns:
            Información del recurso
        """
        response = self.client.get(
            f"/mcp/v1/resources/{resource_id}",
            headers=self._get_headers(headers)
        )
        response.raise_for_status()
        return response.json()
    
    def query_resource(
        self,
        resource_id: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta una operación sobre un recurso.
        
        Args:
            resource_id: ID del recurso
            operation: Operación a ejecutar
            parameters: Parámetros de la operación (opcional)
            context: Contexto adicional (opcional)
            headers: Headers adicionales (opcional)
            
        Returns:
            Resultado de la operación
        """
        request_data = {
            "resource_id": resource_id,
            "operation": operation,
            "parameters": parameters or {},
        }
        if context is not None:
            request_data["context"] = context
        
        response = self.client.post(
            f"/mcp/v1/resources/{resource_id}/query",
            json=request_data,
            headers=self._get_headers(headers)
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado de salud del servidor.
        
        Returns:
            Estado de salud
        """
        response = self.client.get("/mcp/v1/health")
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del servidor.
        
        Returns:
            Métricas del servidor
        """
        response = self.client.get("/mcp/v1/metrics")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del servidor.
        
        Returns:
            Estadísticas del servidor
        """
        response = self.client.get("/mcp/v1/stats")
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtiene información del servidor.
        
        Returns:
            Información del servidor
        """
        response = self.client.get("/mcp/v1/info")
        response.raise_for_status()
        return response.json()


class AsyncMCPTestClient:
    """
    Cliente de testing asíncrono para MCP.
    
    Similar a MCPTestClient pero para testing asíncrono.
    """
    
    def __init__(self, app, base_url: str = "http://test", default_headers: Optional[Dict[str, str]] = None):
        """
        Inicializar cliente de testing asíncrono.
        
        Args:
            app: Aplicación FastAPI
            base_url: URL base para testing
            default_headers: Headers por defecto (opcional)
        """
        self.app = app
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self._auth_token: Optional[str] = None
        self._client: Optional[AsyncClient] = None
    
    async def __aenter__(self):
        """Context manager entry"""
        self._client = AsyncClient(app=self.app, base_url=self.base_url)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._client:
            await self._client.aclose()
    
    def set_auth_token(self, token: str) -> None:
        """Establecer token de autenticación"""
        self._auth_token = token
        self.default_headers["Authorization"] = f"Bearer {token}"
    
    def _get_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Combinar headers"""
        combined = self.default_headers.copy()
        if headers:
            combined.update(headers)
        return combined
    
    async def list_resources(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Lista recursos (async)"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        response = await self._client.get(
            "/mcp/v1/resources",
            headers=self._get_headers(headers)
        )
        response.raise_for_status()
        return response.json()
    
    async def get_resource(self, resource_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Obtiene recurso (async)"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        response = await self._client.get(
            f"/mcp/v1/resources/{resource_id}",
            headers=self._get_headers(headers)
        )
        response.raise_for_status()
        return response.json()
    
    async def query_resource(
        self,
        resource_id: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Ejecuta operación (async)"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        request_data = {
            "resource_id": resource_id,
            "operation": operation,
            "parameters": parameters or {},
        }
        if context is not None:
            request_data["context"] = context
        
        response = await self._client.post(
            f"/mcp/v1/resources/{resource_id}/query",
            json=request_data,
            headers=self._get_headers(headers)
        )
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check (async)"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        response = await self._client.get("/mcp/v1/health")
        response.raise_for_status()
        return response.json()

