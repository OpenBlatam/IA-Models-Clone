"""
OpenRouter API Client
=====================
Cliente de bajo nivel para OpenRouter API.
"""

import logging
from typing import Dict, Any, Optional, List, AsyncIterator, NoReturn
import httpx
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from ...config.app_config import get_config
from ...core.exceptions import OpenRouterException

logger = logging.getLogger(__name__)

# Constants
STREAM_DATA_PREFIX = "data: "
STREAM_DONE_MARKER = "[DONE]"
CONTENT_TYPE_JSON = "application/json"
AUTHORIZATION_PREFIX = "Bearer "
HTTP_REFERER = "https://blatam-academy.com"
APP_TITLE = "Dog Training Coaching AI"
STREAM_DATA_PREFIX_LENGTH = 6  # Length of "data: "


class APIClient:
    """Cliente HTTP para OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar cliente API.
        
        Args:
            api_key: API key de OpenRouter
        """
        self._config = get_config()
        self.api_key = api_key or self._config.openrouter_api_key
        self.base_url = self._config.openrouter_base_url
        self.timeout = self._config.request_timeout
        self.max_retries = self._config.max_retries
        
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
    
    def _get_headers(self, **extra_headers) -> Dict[str, str]:
        """
        Obtener headers estándar para requests.
        
        Args:
            **extra_headers: Headers adicionales
            
        Returns:
            Diccionario de headers
        """
        return {
            "Authorization": f"{AUTHORIZATION_PREFIX}{self.api_key}",
            "Content-Type": CONTENT_TYPE_JSON,
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": APP_TITLE,
            **extra_headers
        }
    
    def _check_api_key(self) -> None:
        """
        Verificar que la API key esté configurada.
        
        Raises:
            OpenRouterException: Si la API key no está configurada
        """
        if not self.api_key:
            raise OpenRouterException("OpenRouter API key not configured")
    
    def _handle_request_error(self, error: Exception) -> NoReturn:
        """
        Manejar errores de request de forma consistente.
        
        Args:
            error: Excepción capturada
            
        Raises:
            OpenRouterException: Siempre lanza una excepción con el error formateado
        """
        if isinstance(error, httpx.HTTPStatusError):
            logger.error(f"OpenRouter API error: {error.response.status_code}")
            raise OpenRouterException(f"API error: {error.response.status_code}")
        elif isinstance(error, httpx.RequestError):
            logger.error(f"Request error: {error}")
            raise OpenRouterException(f"Request failed: {str(error)}")
        else:
            logger.error(f"Unexpected error: {error}")
            raise OpenRouterException(f"Unexpected error: {str(error)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """
        Realizar request HTTP con retry.
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint de la API (sin el base_url)
            **kwargs: Argumentos adicionales para httpx (json, headers, etc.)
            
        Returns:
            Response de httpx
            
        Raises:
            OpenRouterException: Si el request falla después de los reintentos
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://blatam-academy.com",
            "X-Title": "Dog Training Coaching AI",
            **kwargs.pop("headers", {})
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs
            )
            response.raise_for_status()
            return response
    
    async def post_chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enviar request de chat completions.
        
        Args:
            messages: Lista de mensajes
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura
            **kwargs: Argumentos adicionales
            
        Returns:
            Respuesta de la API
        """
        self._check_api_key()
        
        try:
            response = await self._make_request(
                method="POST",
                endpoint="/chat/completions",
                json={
                    "model": model or self._config.openrouter_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
            )
            return response.json()
        except Exception as e:
            self._handle_request_error(e)
    
    async def post_chat_completions_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Enviar request de chat completions con streaming.
        
        Args:
            messages: Lista de mensajes
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura
            **kwargs: Argumentos adicionales
        
        Yields:
            Chunks de texto
        """
        self._check_api_key()
        
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        
        payload = {
            "model": model or config.openrouter_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    method="POST",
                    url=url,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith(STREAM_DATA_PREFIX):
                            data_str = line[STREAM_DATA_PREFIX_LENGTH:]
                            
                            if data_str.strip() == STREAM_DONE_MARKER:
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            self._handle_request_error(e)
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Obtener lista de modelos disponibles.
        
        Returns:
            Lista de modelos
        """
        try:
            response = await self._make_request(
                method="GET",
                endpoint="/models"
            )
            return response.json().get("data", [])
        except OpenRouterException:
            raise
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

