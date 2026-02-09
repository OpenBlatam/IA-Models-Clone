"""
OpenRouter Client for Autonomous Long-Term Agent
Enhanced with resilience patterns
"""

import asyncio
import logging
import os
import json
from typing import Dict, Any, Optional, List
import httpx

from ...core.resilience import ResilienceManager, RetryConfig, CircuitBreakerConfig

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Client for OpenRouter API with connection pooling and resilience"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = OPENROUTER_API_URL
        self.timeout = 60.0
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        # Resilience patterns
        self.resilience = ResilienceManager(
            retry_config=RetryConfig(),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=2
            )
        )
    
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
                    timeout = httpx.Timeout(self.timeout, connect=10.0)
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
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        http_referer: Optional[str] = None,
        app_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using OpenRouter
        
        Args:
            model: Model identifier
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
            "X-Title": app_name or "Autonomous Long-Term Agent"
        }
        
        payload = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        excluded = {"http_referer", "app_name"}
        payload.update({k: v for k, v in kwargs.items() if k not in excluded})
        
        client = await self._get_client()
        
        # Use resilience patterns for API calls
        async def _make_request():
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        
        try:
            data = await self.resilience.execute(
                _make_request,
                use_retry=True,
                use_circuit_breaker=True,
                context={"operation": "openrouter_chat_completion", "model": model}
            )
            
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            
            return {
                "response": content,
                "tokens_used": tokens_used,
                "model": data.get("model", model),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "id": data.get("id"),
                "created": data.get("created")
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"OpenRouter API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {})
                error_msg = error_detail.get("message", error_msg)
                error_type = error_detail.get("type", "api_error")
            except Exception:
                error_type = "http_error"
            
            logger.error(f"OpenRouter API error ({error_type}): {error_msg}")
            raise Exception(f"{error_type}: {error_msg}")
        except httpx.TimeoutException:
            logger.error(f"OpenRouter API timeout after {self.timeout}s")
            raise Exception(f"Request timeout after {self.timeout}s")
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")
            raise
    
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get resilience statistics"""
        return self.resilience.get_stats()


_openrouter_client: Optional[OpenRouterClient] = None


def get_openrouter_client() -> OpenRouterClient:
    """Get or create OpenRouter client instance"""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client

