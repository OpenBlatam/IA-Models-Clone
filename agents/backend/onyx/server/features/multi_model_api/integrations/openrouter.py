"""
OpenRouter integration for Multi-Model API

OpenRouter provides unified access to 1000+ AI models from various providers:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude 3, Claude 2, etc.)
- Google (Gemini, PaLM, etc.)
- Meta (Llama 2, Llama 3, etc.)
- Mistral AI
- Cohere
- And many more...

Features:
- Unified API compatible with OpenAI format
- Automatic routing to best model
- Cost optimization
- Detailed logging and analytics
- Streaming support
"""

import asyncio
import logging
import os
import json
import time
from typing import Dict, Any, Optional, AsyncIterator, List
import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class OpenRouterClient:
    """Client for OpenRouter API - optimized with connection pooling"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = OPENROUTER_API_URL
        self.timeout = 60.0
        self._models_cache: Optional[list] = None
        self._models_cache_time: Optional[float] = None
        self._cache_ttl = 3600
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
    
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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using OpenRouter
        
        Args:
            model: Model identifier (e.g., 'openai/gpt-4', 'anthropic/claude-3-opus')
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'response' and 'tokens_used'
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")
        
        # Optimized: build headers once, reuse base structure
        http_referer = kwargs.get("http_referer", "https://github.com/your-repo")
        app_name = kwargs.get("app_name", "Multi-Model API")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": http_referer,
            "X-Title": app_name
        }
        
        # Optimized: build payload with conditional updates
        payload = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Filter excluded keys more efficiently
        excluded = {"http_referer", "app_name"}
        payload.update({k: v for k, v in kwargs.items() if k not in excluded})
        
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
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
    
    async def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Create a streaming chat completion using OpenRouter
        
        Args:
            model: Model identifier
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Yields:
            Dict with streaming chunks containing 'content', 'finish_reason', etc.
        """
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY environment variable."
            )
        
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list")
        
        # Optimized: build headers once, reuse base structure
        http_referer = kwargs.get("http_referer", "https://github.com/your-repo")
        app_name = kwargs.get("app_name", "Multi-Model API")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": http_referer,
            "X-Title": app_name
        }
        
        # Optimized: build payload with conditional updates
        payload = {"model": model, "messages": messages, "stream": True}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Filter excluded keys more efficiently
        excluded = {"http_referer", "app_name"}
        payload.update({k: v for k, v in kwargs.items() if k not in excluded})
        
        client = await self._get_client()
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            choice = chunk_data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            
                            yield {
                                "content": delta.get("content", ""),
                                "finish_reason": choice.get("finish_reason"),
                                "model": chunk_data.get("model", model),
                                "id": chunk_data.get("id"),
                                "usage": chunk_data.get("usage")
                            }
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as e:
            error_msg = f"OpenRouter API error: {e.response.status_code}"
            try:
                error_data = await e.response.json()
                error_detail = error_data.get("error", {})
                error_msg = error_detail.get("message", error_msg)
                error_type = error_detail.get("type", "api_error")
            except Exception:
                error_type = "http_error"
            
            logger.error(f"OpenRouter streaming error ({error_type}): {error_msg}")
            raise Exception(f"{error_type}: {error_msg}")
        except httpx.TimeoutException:
            logger.error(f"OpenRouter streaming timeout after {self.timeout}s")
            raise Exception(f"Request timeout after {self.timeout}s")
        except Exception as e:
            logger.error(f"Error streaming from OpenRouter: {e}")
            raise
    
    async def list_models(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """List all available models from OpenRouter with caching"""
        if not self.api_key:
            return []
        
        current_time = time.time()
        
        if use_cache and self._models_cache and self._models_cache_time:
            if current_time - self._models_cache_time < self._cache_ttl:
                return self._models_cache
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.base_url}/models",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            
            if use_cache and models:
                self._models_cache = models
                self._models_cache_time = current_time
            
            return models or []
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error listing models: {e.response.status_code}")
            if self._models_cache:
                return self._models_cache
            return []
        except Exception as e:
            logger.error(f"Error listing OpenRouter models: {e}")
            if self._models_cache:
                return self._models_cache
            return []


_openrouter_client: Optional[OpenRouterClient] = None


def get_openrouter_client() -> OpenRouterClient:
    """Get or create OpenRouter client instance"""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client


async def openrouter_handler(
    prompt: str,
    model: str = "openai/gpt-3.5-turbo",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Handler function for OpenRouter models
    
    Args:
        prompt: Input prompt
        model: OpenRouter model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        **kwargs: Additional parameters
        
    Returns:
        Dict with 'response' and 'tokens_used'
    """
    client = get_openrouter_client()
    
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    result = await client.chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return result


POPULAR_OPENROUTER_MODELS = {
    "openai/gpt-4": "GPT-4 by OpenAI",
    "openai/gpt-4-turbo": "GPT-4 Turbo by OpenAI",
    "openai/gpt-3.5-turbo": "GPT-3.5 Turbo by OpenAI",
    "anthropic/claude-3-opus": "Claude 3 Opus by Anthropic",
    "anthropic/claude-3-sonnet": "Claude 3 Sonnet by Anthropic",
    "anthropic/claude-3-haiku": "Claude 3 Haiku by Anthropic",
    "google/gemini-pro": "Gemini Pro by Google",
    "google/gemini-pro-vision": "Gemini Pro Vision by Google",
    "meta-llama/llama-3-70b-instruct": "Llama 3 70B by Meta",
    "mistralai/mistral-large": "Mistral Large by Mistral AI",
    "mistralai/mixtral-8x7b-instruct": "Mixtral 8x7B by Mistral AI",
    "cohere/command-r-plus": "Command R+ by Cohere",
    "perplexity/llama-3-sonar-large-32k-online": "Llama 3 Sonar Large by Perplexity",
    "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B by Alibaba",
    "01-ai/yi-34b-chat": "Yi 34B Chat by 01.AI",
}

