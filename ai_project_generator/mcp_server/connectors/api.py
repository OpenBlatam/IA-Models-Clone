"""
API Connector - Conector para APIs externas
============================================
"""

import httpx
import logging
from typing import Any, Dict, Optional, List

from .base import BaseConnector
from ..contracts import ContextFrame
from ..exceptions import MCPConnectorError, MCPOperationError

logger = logging.getLogger(__name__)


class APIConnector(BaseConnector):
    """
    Conector para acceso a APIs externas
    
    Operaciones soportadas:
    - get: GET request
    - post: POST request
    - put: PUT request
    - delete: DELETE request
    - patch: PATCH request
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        Inicializa el conector de API.
        
        Args:
            base_url: URL base para requests (opcional)
            timeout: Timeout en segundos (default: 30)
            
        Raises:
            ValueError: Si timeout es inválido
        """
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            raise ValueError("timeout must be a positive integer")
        
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._supported_operations = {
            "get", "post", "put", "delete", "patch", "request"
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea cliente HTTP"""
        if not self._client:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    async def execute(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        context: Optional[ContextFrame] = None,
    ) -> Any:
        """
        Ejecuta operación sobre API.
        
        Args:
            resource_id: ID del recurso
            operation: Operación HTTP a ejecutar
            parameters: Parámetros con url, headers, params, data, json
            context: Frame de contexto (opcional)
            
        Returns:
            Diccionario con status_code, headers, content y url
            
        Raises:
            MCPOperationError: Si la operación no está soportada
            MCPConnectorError: Si hay error HTTP o de conexión
            ValueError: Si los parámetros son inválidos
        """
        if not self.validate_operation(operation):
            raise MCPOperationError(f"Operation {operation} not supported by APIConnector")
        
        if not parameters or not isinstance(parameters, dict):
            raise ValueError("parameters must be a non-empty dictionary")
        
        try:
            client = await self._get_client()
            
            url = parameters.get("url", resource_id)
            if not url or not isinstance(url, str):
                raise ValueError("url must be a non-empty string")
            
            headers = parameters.get("headers", {})
            if not isinstance(headers, dict):
                headers = {}
            
            params = parameters.get("params", {})
            if not isinstance(params, dict):
                params = {}
            
            data = parameters.get("data")
            json_data = parameters.get("json")
            
            if operation == "get":
                response = await client.get(url, headers=headers, params=params)
            elif operation == "post":
                response = await client.post(url, headers=headers, params=params, data=data, json=json_data)
            elif operation == "put":
                response = await client.put(url, headers=headers, params=params, data=data, json=json_data)
            elif operation == "delete":
                response = await client.delete(url, headers=headers, params=params)
            elif operation == "patch":
                response = await client.patch(url, headers=headers, params=params, data=data, json=json_data)
            elif operation == "request":
                method = parameters.get("method", "GET")
                if not isinstance(method, str):
                    method = "GET"
                response = await client.request(method.upper(), url, headers=headers, params=params, data=data, json=json_data)
            else:
                raise MCPOperationError(f"Operation {operation} not implemented")
            
            response.raise_for_status()
            
            # Intentar parsear JSON, si falla retornar texto
            try:
                content = response.json()
            except Exception:
                content = response.text
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": content,
                "url": str(response.url),
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error in API connector: {e.response.status_code} - {e}")
            raise MCPConnectorError(
                f"API request failed with status {e.response.status_code}: {str(e)}",
                error_code=f"HTTP_{e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in API connector: {e}")
            raise MCPConnectorError(f"API request failed: {str(e)}") from e
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid parameters in API connector: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in API connector: {e}", exc_info=True)
            raise MCPConnectorError(f"Unexpected error: {e}") from e
    
    def validate_operation(self, operation: str) -> bool:
        """Valida si operación es soportada"""
        return operation.lower() in self._supported_operations
    
    def get_supported_operations(self) -> List[str]:
        """
        Retorna operaciones soportadas.
        
        Returns:
            Lista de nombres de operaciones HTTP soportadas
        """
        return list(self._supported_operations)
    
    async def close(self) -> None:
        """
        Cierra cliente HTTP y libera recursos.
        
        Debe llamarse cuando el conector ya no se use para evitar
        conexiones abiertas.
        """
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")
            finally:
                self._client = None

