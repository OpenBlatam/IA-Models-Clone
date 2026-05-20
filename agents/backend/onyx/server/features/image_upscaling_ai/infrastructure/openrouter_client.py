"""
OpenRouter Client for Image Upscaling AI
==========================================

Client for OpenRouter API with resilience patterns and connection pooling.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from .base_http_client import BaseHTTPClient, HTTPClientConfig, HTTPMethod
from .response_parser import OpenRouterResponseParser
from .error_handlers import handle_openrouter_error

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_CONNECT_TIMEOUT = 10.0


class OpenRouterClient(BaseHTTPClient):
    """
    Client for OpenRouter API with connection pooling and resilience.
    
    Inherits from BaseHTTPClient for unified HTTP client functionality.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: float = 30.0,
        enable_http2: bool = True
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
            connect_timeout: Connection timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay in seconds
            max_connections: Maximum connections in pool
            max_keepalive_connections: Maximum keepalive connections
            keepalive_expiry: Keepalive expiry in seconds
            enable_http2: Enable HTTP/2 support
        """
        api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("OpenRouter API key not configured")
        
        # Configure default headers for OpenRouter
        default_headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://blatam-academy.com",
            "X-Title": "Image Upscaling AI"
        }
        
        # Initialize base client with configuration
        config = HTTPClientConfig(
            base_url=OPENROUTER_API_URL,
            api_key=api_key,
            timeout=timeout,
            connect_timeout=connect_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
            default_headers=default_headers,
            enable_http2=enable_http2
        )
        
        super().__init__(config)
        
        # Cache for models list (avoid repeated API calls)
        self._models_cache: Optional[List[Dict[str, Any]]] = None
        
        # Default model (can be set via parameter or config)
        self.default_model = model or "anthropic/claude-3.5-sonnet"
        
        logger.info("Initialized OpenRouterClient")
    
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
            http_referer: HTTP referer header (overrides default)
            app_name: Application name header (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'response', 'tokens_used', and metadata
        """
        if not self.config.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        # Build payload
        payload = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Add additional parameters (excluding header-related ones)
        excluded = {"http_referer", "app_name"}
        payload.update({k: v for k, v in kwargs.items() if k not in excluded})
        
        # Build headers (override defaults if provided)
        headers = {}
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_name:
            headers["X-Title"] = app_name
        
        try:
            # Use base client's post method which handles retries automatically
            data = await self.post(
                "/chat/completions",
                json=payload,
                headers=headers if headers else None
            )
        except Exception as e:
            # Handle OpenRouter-specific errors
            raise handle_openrouter_error(
                e,
                timeout=self.config.timeout,
                operation_name="OpenRouter API request"
            )
        
        # Parse response
        return OpenRouterResponseParser.parse_chat_completion(data, model)
    
    async def chat_completion_simple(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Simplified chat completion with just prompt text.
        
        Args:
            model: Model identifier
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Response text (just the content, not full response dict)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        result = await self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return result.get("response", "")
    
    def is_configured(self) -> bool:
        """
        Check if client is properly configured.
        
        Returns:
            True if API key is configured, False otherwise
        """
        return bool(self.config.api_key)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check by getting models list.
        
        Returns:
            Health check result with status and details
        """
        try:
            data = await self.get("/models")
            models = data.get("data", [])
            return {
                "status": "healthy",
                "models_available": len(models),
                "api_url": OPENROUTER_API_URL,
                "configured": self.is_configured()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "api_url": OPENROUTER_API_URL,
                "configured": self.is_configured()
            }

