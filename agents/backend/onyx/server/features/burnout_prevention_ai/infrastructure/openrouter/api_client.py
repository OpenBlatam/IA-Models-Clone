"""
OpenRouter API Client
====================
Cliente HTTP para comunicación con OpenRouter API.
"""

import os
from typing import Dict, Any, Optional, List
import httpx
from ...core.types import JSONDict, MessageList, ModelList

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        retry_if,
        RetryError
    )
    _has_tenacity = True
except ImportError:
    _has_tenacity = False

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class APIClient:
    """Cliente HTTP para OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar cliente API.
        
        Args:
            api_key: API key de OpenRouter
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
        
        # Import timeout constants
        from ..core.constants import HTTP_TIMEOUT, HTTP_CONNECT_TIMEOUT
        
        # Optimize HTTP client with connection pooling and limits
        self.http_client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
                "X-Title": "Burnout Prevention AI"
            },
            timeout=httpx.Timeout(HTTP_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            http2=True  # Use HTTP/2 if available
        )
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> JSONDict:
        """Internal method to make HTTP request with retry logic."""
        import time as time_module
        from ..core.metrics import record_openrouter_request
        
        async def _request():
            """Internal request function with metrics and error handling."""
            start_time = time_module.time()
            
            # Extract model from kwargs for metrics (optimized)
            json_payload = kwargs.get("json", {})
            if isinstance(json_payload, dict):
                model = json_payload.get("model", "unknown")
            else:
                model = "unknown"
            
            try:
                response = await self.http_client.request(method, endpoint, **kwargs)
                response.raise_for_status()
                duration = time_module.time() - start_time
                
                # Record success metric
                record_openrouter_request(model=model, status="success", duration=duration)
                
                # Use centralized JSON parsing (handles orjson automatically)
                from ...core.utils import _json_loads, _has_orjson
                return _json_loads(response.content) if _has_orjson else response.json()
            except Exception as e:
                duration = time_module.time() - start_time
                status = "error"
                if isinstance(e, httpx.HTTPStatusError):
                    status = f"http_{e.response.status_code}"
                record_openrouter_request(model=model, status=status, duration=duration)
                raise
        
        if _has_tenacity:
            # Retry on network errors and 5xx status codes
            # Use constants but import at function level to avoid circular imports
            from ..core.constants import (
                MAX_RETRY_ATTEMPTS,
                RETRY_MIN_WAIT,
                RETRY_MAX_WAIT,
                RETRY_MULTIPLIER
            )
            
            # Helper function to check if error should be retried
            def should_retry_http_error(exception: Exception) -> bool:
                """Check if HTTP error should be retried (5xx only)."""
                if isinstance(exception, httpx.HTTPStatusError):
                    return exception.response.status_code >= 500
                return False
            
            # Only retry on network errors and 5xx server errors (not 4xx client errors)
            retry_decorator = retry(
                stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
                wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
                retry=retry_if_exception_type(httpx.RequestError) | retry_if(should_retry_http_error),
                reraise=True
            )
            try:
                return await retry_decorator(_request)()
            except httpx.HTTPStatusError as e:
                # Log and re-raise HTTP status errors (retry logic handles 5xx)
                from ..core.constants import MAX_ERROR_MESSAGE_LENGTH
                error_text = e.response.text[:MAX_ERROR_MESSAGE_LENGTH] if hasattr(e.response, 'text') else str(e)
                if 400 <= e.response.status_code < 500:
                    logger.error("OpenRouter API client error (4xx)", status=e.response.status_code, text=error_text)
                else:
                    logger.error("OpenRouter API server error (5xx)", status=e.response.status_code, text=error_text)
                raise
        else:
            # Fallback without retry
            return await _request()
    
    async def post_chat_completions(
        self,
        messages: MessageList,
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> JSONDict:
        """
        Enviar request de chat completion (with retry logic).
        
        Args:
            messages: Lista de mensajes
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura
            **kwargs: Parámetros adicionales
        
        Returns:
            Respuesta de la API
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            return await self._make_request("POST", "/chat/completions", json=payload)
        except Exception as e:
            self._handle_api_error(e, context="OpenRouter")
    
    async def get_models(self) -> ModelList:
        """
        Obtener lista de modelos disponibles (with retry logic).
        
        Returns:
            Lista de modelos
        """
        try:
            data = await self._make_request("GET", "/models")
            # Validate response structure
            if not isinstance(data, dict):
                from ..core.exceptions import APIError
                raise APIError("Invalid response format from OpenRouter models endpoint")
            return data.get("data", [])
        except Exception as e:
            self._handle_api_error(e, context="OpenRouter models")
    
    def _handle_api_error(self, e: Exception, context: str = "OpenRouter") -> None:
        """
        Centralized error handling for API errors.
        
        Args:
            e: Exception to handle
            context: Context string for error messages
            
        Raises:
            APIError: Always raises APIError with formatted message
        """
        from ..core.exceptions import APIError
        from ..core.constants import MAX_ERROR_MESSAGE_LENGTH
        
        from ..core.logging_helpers import truncate_error_message
        
        if isinstance(e, httpx.HTTPStatusError):
            error_text = truncate_error_message(e) if not hasattr(e.response, 'text') else e.response.text[:MAX_ERROR_MESSAGE_LENGTH]
            logger.error("API error", context=context, status=e.response.status_code, text=error_text)
            raise APIError(f"{context} API error: {e.response.status_code} - {error_text}") from e
        elif isinstance(e, httpx.RequestError):
            error_msg = truncate_error_message(e)
            logger.error("Request error", context=context, error=error_msg)
            raise APIError(f"{context} request error: {error_msg}") from e
        else:
            error_msg = truncate_error_message(e)
            logger.error("Unexpected error", context=context, error=error_msg)
            raise APIError(f"Unexpected {context} error: {error_msg}") from e
    
    async def close(self):
        """Cerrar cliente HTTP."""
        await self.http_client.aclose()

