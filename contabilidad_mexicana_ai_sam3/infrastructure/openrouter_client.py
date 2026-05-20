"""
OpenRouter Client for Contabilidad Mexicana AI SAM3
===================================================

Client for OpenRouter API with resilience patterns and connection pooling.
Based on SAM3 architecture but adapted for async operation.

Refactored to:
- Use centralized error handling
- Use response parser for consistent parsing
- Eliminate duplicate error handling logic
"""

import asyncio
import logging
import os
import base64
from typing import Dict, Any, Optional, List
import httpx

from .retry_helpers import (
    retry_with_exponential_backoff,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY
)
from .response_parser import OpenRouterResponseParser
from .error_handlers import handle_openrouter_error

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_CONNECT_TIMEOUT = 10.0


class OpenRouterClient:
    """Client for OpenRouter API with connection pooling and resilience"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
        
        self.base_url = OPENROUTER_API_URL
        self.timeout = DEFAULT_TIMEOUT
        self.max_retries = DEFAULT_MAX_RETRIES
        self.retry_delay = DEFAULT_RETRY_DELAY
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        logger.info("Initialized OpenRouterClient")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=100,
                        max_keepalive_connections=20,
                        keepalive_expiry=30.0
                    )
                    timeout = httpx.Timeout(self.timeout, connect=DEFAULT_CONNECT_TIMEOUT)
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=True
                    )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources"""
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing OpenRouter client: {e}")
            finally:
                self._client = None
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        http_referer: Optional[str] = None,
        app_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using OpenRouter.
        
        Args:
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            http_referer: HTTP referer header
            app_name: Application name header
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'response', 'tokens_used', and metadata
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": http_referer or "https://blatam-academy.com",
            "X-Title": app_name or "Contabilidad Mexicana AI SAM3"
        }
        
        payload = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        excluded = {"http_referer", "app_name"}
        payload.update({k: v for k, v in kwargs.items() if k not in excluded})
        
        client = await self._get_client()
        
        async def _make_request():
            """Make the actual HTTP request."""
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        
        # Use retry helper for HTTP errors and timeouts
        try:
            data = await retry_with_exponential_backoff(
                _make_request,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                retryable_exceptions=(httpx.HTTPStatusError, httpx.TimeoutException),
                operation_name="OpenRouter API request"
            )
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            raise handle_openrouter_error(e, timeout=self.timeout, operation_name="OpenRouter API request")
        
        # Parse response
        return OpenRouterResponseParser.parse_chat_completion(data, model)

