"""
LLM Client for Unified AI Model
Supports DeepSeek API directly and OpenRouter for multiple models
Combines resilience patterns from autonomous_long_term_agent and github_autonomous_agent
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import httpx

from ..config import get_config, OpenRouterConfig, DeepSeekConfig, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    min_wait: float = 1.0
    max_wait: float = 10.0
    exponential_base: float = 2.0


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    Prevents cascading failures by temporarily blocking requests after failures.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.failure_threshold = config.failure_threshold
        self.recovery_timeout = config.recovery_timeout
        self.success_threshold = config.success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._close()
        else:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
    
    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self._open()
    
    def can_attempt(self) -> bool:
        """Check if a request can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False
        
        # Half-open state - allow limited requests
        return True
    
    def _open(self) -> None:
        """Open the circuit breaker."""
        self.state = CircuitState.OPEN
        logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _close(self) -> None:
        """Close the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker closed - service recovered")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, key: str = "default") -> bool:
        """Check if a request is allowed."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests
        self.requests[key] = [t for t in self.requests[key] if t > cutoff]
        
        # Check limit
        if len(self.requests[key]) >= self.requests_per_minute:
            return False
        
        # Record request
        self.requests[key].append(now)
        return True
    
    def get_retry_after(self, key: str = "default") -> float:
        """Get seconds until rate limit resets."""
        if key not in self.requests or not self.requests[key]:
            return 0.0
        
        oldest = min(self.requests[key])
        return max(0.0, self.window_seconds - (datetime.now() - oldest).total_seconds())


class OpenRouterClient:
    """
    Client for DeepSeek and OpenRouter APIs with connection pooling, retry logic, and resilience patterns.
    Automatically detects which API to use based on configured API keys.
    """
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        app_config = get_config()
        
        # Detect which API to use (DeepSeek takes priority)
        self.use_deepseek = app_config.use_deepseek
        
        if self.use_deepseek:
            # Use DeepSeek API directly
            self.api_key = app_config.deepseek.api_key
            self.base_url = app_config.deepseek.api_base
            self.timeout = app_config.deepseek.timeout
            self.provider = "deepseek"
            logger.info("Using DeepSeek API directly")
        else:
            # Use OpenRouter
            self.config = config or app_config.openrouter
            self.api_key = self.config.api_key
            self.base_url = self.config.api_base
            self.timeout = self.config.timeout
            self.provider = "openrouter"
            logger.info("Using OpenRouter API")
        
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        # Resilience patterns
        self.circuit_breaker = CircuitBreaker(app_config.circuit_breaker)
        self.rate_limiter = RateLimiter(
            requests_per_minute=app_config.rate_limit.requests_per_minute,
            window_seconds=app_config.rate_limit.window_seconds
        )
        
        retry_max = 3
        retry_min = 1.0
        retry_max_wait = 10.0
        if hasattr(self, 'config') and self.config:
            retry_max = self.config.max_retries
            retry_min = self.config.retry_min_wait
            retry_max_wait = self.config.retry_max_wait
        
        self.retry_config = RetryConfig(
            max_retries=retry_max,
            min_wait=retry_min,
            max_wait=retry_max_wait
        )
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
            "provider": self.provider
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=100,
                        max_keepalive_connections=20,
                        keepalive_expiry=30.0
                    )
                    timeout = httpx.Timeout(self.timeout, connect=10.0)
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                        
                        # Add OpenRouter-specific headers if not using DeepSeek
                        if not self.use_deepseek and hasattr(self, 'config') and self.config:
                            headers["HTTP-Referer"] = self.config.http_referer
                            headers["X-Title"] = self.config.app_name
                    
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=False,  # Disabled to avoid h2 dependency
                        headers=headers
                    )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing OpenRouter client: {e}")
            finally:
                self._client = None
    
    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for the current provider."""
        if self.use_deepseek:
            # For DeepSeek direct API, use simple model names
            if "/" in model:
                # Convert "deepseek/deepseek-chat" to "deepseek-chat"
                model = model.split("/")[-1]
            # Ensure it's a valid DeepSeek model
            if model not in ["deepseek-chat", "deepseek-coder"]:
                model = "deepseek-chat"  # Default
        return model
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using DeepSeek or OpenRouter.
        
        Args:
            model: Model identifier (e.g., "deepseek-chat", "deepseek/deepseek-chat")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'content', 'tokens_used', 'model', and metadata
        """
        if not self.api_key:
            raise ValueError(f"API key not configured for {self.provider}")
        
        # Normalize model name for the provider
        model = self._normalize_model_name(model)
        
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            raise Exception("Circuit breaker is open - service temporarily unavailable")
        
        # Check rate limit
        if not self.rate_limiter.is_allowed(model):
            retry_after = self.rate_limiter.get_retry_after(model)
            raise Exception(f"Rate limit exceeded. Retry after {retry_after:.1f} seconds")
        
        # Build payload
        payload = {
            "model": model,
            "messages": messages
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stream:
            payload["stream"] = stream
        
        # Add additional parameters
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        
        # Execute request with retry
        return await self._execute_with_retry(payload, stream=stream)
    
    async def _execute_with_retry(
        self,
        payload: Dict[str, Any],
        stream: bool = False
    ) -> Dict[str, Any]:
        """Execute request with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_config.max_retries):
            try:
                self.stats["total_requests"] += 1
                start_time = datetime.now()
                
                client = await self._get_client()
                
                if stream:
                    return await self._execute_stream(client, payload)
                
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Record success
                self.circuit_breaker.record_success()
                self.stats["successful_requests"] += 1
                self.stats["total_latency_ms"] += latency_ms
                
                # Extract response
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)
                self.stats["total_tokens"] += tokens_used
                
                return {
                    "content": content,
                    "tokens_used": tokens_used,
                    "model": data.get("model", payload.get("model")),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "id": data.get("id"),
                    "created": data.get("created")
                }
                
            except httpx.HTTPStatusError as e:
                last_error = e
                self.circuit_breaker.record_failure()
                self.stats["failed_requests"] += 1
                
                # Don't retry on auth errors
                if e.response.status_code in [401, 403]:
                    raise Exception(f"Authentication error: {e.response.text}")
                
                # Don't retry on 4xx client errors (except rate limit)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise Exception(f"Client error: {e.response.text}")
                
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
            except httpx.TimeoutException as e:
                last_error = e
                self.circuit_breaker.record_failure()
                self.stats["failed_requests"] += 1
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                
            except Exception as e:
                last_error = e
                self.circuit_breaker.record_failure()
                self.stats["failed_requests"] += 1
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < self.retry_config.max_retries - 1:
                wait_time = min(
                    self.retry_config.min_wait * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_wait
                )
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Request failed after {self.retry_config.max_retries} attempts: {last_error}")
    
    async def _execute_stream(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute streaming request."""
        collected_content = []
        
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                collected_content.append(content)
                    except json.JSONDecodeError:
                        continue
        
        self.circuit_breaker.record_success()
        self.stats["successful_requests"] += 1
        
        return {
            "content": "".join(collected_content),
            "tokens_used": 0,  # Not available in streaming
            "model": payload.get("model"),
            "finish_reason": "stop",
            "usage": {},
            "latency_ms": 0.0,
            "streamed": True
        }
    
    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream chat completion responses.
        
        Yields:
            Content chunks as they are generated
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        if not self.circuit_breaker.can_attempt():
            raise Exception("Circuit breaker is open")
        
        if not self.rate_limiter.is_allowed(model):
            raise Exception("Rate limit exceeded")
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        
        try:
            self.stats["total_requests"] += 1
            client = await self._get_client()
            
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
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
            
            self.circuit_breaker.record_success()
            self.stats["successful_requests"] += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.stats["failed_requests"] += 1
            raise
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter."""
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/models")
        response.raise_for_status()
        
        data = response.json()
        return data.get("data", [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self.stats.copy()
        stats["circuit_breaker"] = self.circuit_breaker.get_state()
        stats["average_latency_ms"] = (
            stats["total_latency_ms"] / stats["successful_requests"]
            if stats["successful_requests"] > 0 else 0.0
        )
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0
        }


# Singleton instance
_openrouter_client: Optional[OpenRouterClient] = None


def get_openrouter_client() -> OpenRouterClient:
    """Get or create OpenRouter client instance."""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client


async def close_openrouter_client() -> None:
    """Close the global OpenRouter client."""
    global _openrouter_client
    if _openrouter_client:
        await _openrouter_client.close()
        _openrouter_client = None



