"""
OpenRouter Client for Imagen Video Enhancer AI
===============================================

Client for OpenRouter API with resilience patterns and connection pooling.
"""

import logging
import os
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base_client import BaseHTTPClient
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
DEFAULT_TIMEOUT = 120.0
DEFAULT_CONNECT_TIMEOUT = 10.0


class OpenRouterClient(BaseHTTPClient):
    """Client for OpenRouter API with connection pooling and resilience"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            base_url: Base API URL (defaults to OpenRouter API URL)
        """
        api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("OpenRouter API key not configured")
        
        super().__init__(
            base_url=base_url or OPENROUTER_API_URL,
            api_key=api_key,
            timeout=DEFAULT_TIMEOUT,
            connect_timeout=DEFAULT_CONNECT_TIMEOUT
        )
        
        self.max_retries = DEFAULT_MAX_RETRIES
        self.retry_delay = DEFAULT_RETRY_DELAY
        
        logger.info("Initialized OpenRouterClient")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with OpenRouter-specific headers."""
        headers = super()._get_headers()
        # OpenRouter uses HTTP-Referer and X-Title headers
        return headers
    
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
        
        # Prepare headers
        headers = {
            "HTTP-Referer": http_referer or "https://blatam-academy.com",
            "X-Title": app_name or "Imagen Video Enhancer AI"
        }
        
        # Prepare payload
        payload = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        excluded = {"http_referer", "app_name"}
        payload.update({k: v for k, v in kwargs.items() if k not in excluded})
        
        async def _make_request():
            """Make the actual HTTP request."""
            return await self.post("chat/completions", data=payload, headers=headers)
        
        # Use retry helper for HTTP errors and timeouts
        try:
            import httpx
            data = await retry_with_exponential_backoff(
                _make_request,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                retryable_exceptions=(httpx.HTTPStatusError, httpx.TimeoutException),
                operation_name="OpenRouter API request"
            )
        except Exception as e:
            import httpx
            if isinstance(e, (httpx.HTTPStatusError, httpx.TimeoutException)):
                raise handle_openrouter_error(e, timeout=self.timeout, operation_name="OpenRouter API request")
            raise
        
        # Parse response
        return OpenRouterResponseParser.parse_chat_completion(data, model)
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_image_content(self, image_path: str, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Create image content for OpenRouter vision API."""
        image_data = self.encode_image_to_base64(image_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }
        }
    
    async def process_image(
        self,
        model: str,
        image_path: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        mime_type: str = "image/jpeg",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image with vision model.
        
        Args:
            model: Vision model identifier
            image_path: Path to image file
            prompt: User prompt describing what to do
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            mime_type: MIME type of image
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'response', 'tokens_used', and metadata
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Create multimodal message with image
        image_content = self._create_image_content(image_path, mime_type)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content
            ]
        })
        
        return await self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

