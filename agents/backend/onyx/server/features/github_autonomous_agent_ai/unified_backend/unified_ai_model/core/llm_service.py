"""
LLM Service for Unified AI Model
High-level service combining caching, parallel generation, and intelligent routing
Based on patterns from github_autonomous_agent and autonomous_long_term_agent
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
from enum import Enum

from ..config import get_config, UnifiedAIConfig
from .llm_client import OpenRouterClient, get_openrouter_client

logger = logging.getLogger(__name__)


class FinishReason(str, Enum):
    """Reasons for generation completion."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


@dataclass
class LLMRequest:
    """Request to an LLM model."""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stream: bool = False
    cache_enabled: bool = True
    cache_ttl: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API payload."""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.top_p:
            payload["top_p"] = self.top_p
        if self.frequency_penalty:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            payload["presence_penalty"] = self.presence_penalty
        
        return payload


@dataclass
class LLMResponse:
    """Response from an LLM model."""
    model: str
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    cached: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.error is None and bool(self.content)


@dataclass
class LLMStats:
    """Statistics for LLM service."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_prompt: int = 0
    total_tokens_completion: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    requests_by_model: Dict[str, int] = None
    errors_by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.requests_by_model is None:
            self.requests_by_model = defaultdict(int)
        if self.errors_by_type is None:
            self.errors_by_type = defaultdict(int)
    
    @property
    def average_latency_ms(self) -> float:
        if self.successful_requests > 0:
            return self.total_latency_ms / self.successful_requests
        return 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total > 0:
            return (self.cache_hits / total) * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "total_tokens_prompt": self.total_tokens_prompt,
            "total_tokens_completion": self.total_tokens_completion,
            "total_tokens": self.total_tokens,
            "average_latency_ms": self.average_latency_ms,
            "requests_by_model": dict(self.requests_by_model),
            "errors_by_type": dict(self.errors_by_type),
        }


class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if datetime.now().timestamp() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            # Simple eviction - remove oldest entries
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]["created_at"]
            )[:len(self.cache) // 4]
            for k in oldest_keys:
                del self.cache[k]
        
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            "value": value,
            "created_at": datetime.now().timestamp(),
            "expires_at": datetime.now().timestamp() + ttl
        }
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size
        }


class LLMService:
    """
    High-level LLM service with caching, parallel generation, and intelligent routing.
    
    Features:
    - Response caching with TTL
    - Parallel model execution
    - Streaming support
    - Statistics and monitoring
    - Automatic model selection
    """
    
    def __init__(
        self,
        client: Optional[OpenRouterClient] = None,
        config: Optional[UnifiedAIConfig] = None
    ):
        self.client = client or get_openrouter_client()
        self.config = config or get_config()
        
        # Cache
        self.cache = SimpleCache(
            max_size=self.config.cache.max_size,
            default_ttl=self.config.cache.ttl
        )
        self.cache_enabled = self.config.cache.enabled
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        
        # Statistics
        self.stats = LLMStats()
        
        # Default models
        self.default_models = self.config.default_models
        self.default_model = self.config.default_model
        
        logger.info(f"LLM Service initialized with default model: {self.default_model}")
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key from request."""
        key_data = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return f"llm:{request.model}:{key_hash[:16]}"
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from an LLM.
        
        Args:
            prompt: User prompt
            model: Model to use (default: deepseek/deepseek-chat)
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            cache_enabled: Override cache setting
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.config.default_max_tokens
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Create request
        request = LLMRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_enabled=cache_enabled if cache_enabled is not None else self.cache_enabled,
            **kwargs
        )
        
        return await self._process_request(request)
    
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            cache_enabled: Override cache setting
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.config.default_max_tokens
        
        request = LLMRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_enabled=cache_enabled if cache_enabled is not None else self.cache_enabled,
            **kwargs
        )
        
        return await self._process_request(request)
    
    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate streaming response.
        
        Yields:
            Content chunks as they are generated
        """
        model = model or self.default_model
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        self.stats.total_requests += 1
        self.stats.requests_by_model[model] += 1
        
        try:
            async with self.semaphore:
                async for chunk in self.client.stream_chat_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                ):
                    yield chunk
            
            self.stats.successful_requests += 1
            
        except Exception as e:
            self.stats.failed_requests += 1
            self.stats.errors_by_type["stream_error"] += 1
            logger.error(f"Streaming error: {e}")
            raise
    
    async def generate_parallel(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, LLMResponse]:
        """
        Generate responses from multiple models in parallel.
        
        Args:
            prompt: User prompt
            models: List of models to use (default: all default models)
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Dict mapping model names to responses
        """
        models = models or self.default_models
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Create requests for all models
        requests = [
            LLMRequest(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            for model in models
        ]
        
        # Execute in parallel
        start_time = datetime.now()
        tasks = [self._process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {}
        for model, response in zip(models, responses):
            if isinstance(response, Exception):
                logger.error(f"Error with model {model}: {response}")
                results[model] = LLMResponse(
                    model=model,
                    content="",
                    error=str(response)
                )
            else:
                results[model] = response
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Generated {len(results)} parallel responses in {total_time:.2f}ms")
        
        return results
    
    async def _process_request(self, request: LLMRequest) -> LLMResponse:
        """Process a single LLM request with caching."""
        self.stats.total_requests += 1
        self.stats.requests_by_model[request.model] += 1
        
        # Check cache
        if request.cache_enabled and self.cache_enabled:
            cache_key = self._generate_cache_key(request)
            cached = self.cache.get(cache_key)
            
            if cached:
                self.stats.cache_hits += 1
                logger.debug(f"Cache hit for model {request.model}")
                return LLMResponse(**cached, cached=True)
            
            self.stats.cache_misses += 1
        
        # Execute request
        try:
            async with self.semaphore:
                start_time = datetime.now()
                
                result = await self.client.chat_completion(
                    model=request.model,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    frequency_penalty=request.frequency_penalty,
                    presence_penalty=request.presence_penalty
                )
                
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Build response
                response = LLMResponse(
                    model=result.get("model", request.model),
                    content=result.get("content", ""),
                    usage=result.get("usage"),
                    finish_reason=result.get("finish_reason"),
                    latency_ms=latency_ms
                )
                
                # Update statistics
                self.stats.successful_requests += 1
                self.stats.total_latency_ms += latency_ms
                
                if response.usage:
                    self.stats.total_tokens_prompt += response.usage.get("prompt_tokens", 0)
                    self.stats.total_tokens_completion += response.usage.get("completion_tokens", 0)
                    self.stats.total_tokens += response.usage.get("total_tokens", 0)
                
                # Cache successful response
                if request.cache_enabled and self.cache_enabled and response.success:
                    cache_key = self._generate_cache_key(request)
                    self.cache.set(cache_key, response.to_dict(), request.cache_ttl)
                
                return response
                
        except Exception as e:
            self.stats.failed_requests += 1
            self.stats.errors_by_type[type(e).__name__] += 1
            logger.error(f"LLM request failed: {e}")
            
            return LLMResponse(
                model=request.model,
                content="",
                error=str(e)
            )
    
    async def analyze_code(
        self,
        code: str,
        language: Optional[str] = None,
        analysis_type: str = "general",
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Analyze code using an LLM.
        
        Args:
            code: Code to analyze
            language: Programming language
            analysis_type: Type of analysis (general, bugs, performance, security)
            model: Model to use
            
        Returns:
            LLMResponse with analysis
        """
        system_prompts = {
            "general": "You are an expert code analyst. Analyze the provided code and provide constructive feedback on structure, quality, and best practices.",
            "bugs": "You are an expert bug detector. Analyze the code and identify potential bugs, errors, edge cases, or logical issues.",
            "performance": "You are a performance optimization expert. Analyze the code and suggest performance improvements and optimizations.",
            "security": "You are a security expert. Analyze the code and identify security vulnerabilities and suggest fixes."
        }
        
        system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        
        prompt = f"""Analyze the following code{' (' + language + ')' if language else ''}:

```{language or ''}
{code}
```

Provide a detailed analysis focused on: {analysis_type}. Include:
- Issues identified
- Suggestions for improvement  
- Examples of improved code if relevant"""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.3
        )
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter."""
        return await self.client.get_models()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = self.stats.to_dict()
        stats["cache"] = self.cache.get_stats()
        stats["client"] = self.client.get_stats()
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = LLMStats()
        self.client.reset_stats()
        logger.info("LLM Service statistics reset")
    
    async def close(self) -> None:
        """Close service and cleanup resources."""
        await self.client.close()
        self.cache.clear()
        logger.info("LLM Service closed")


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


async def close_llm_service() -> None:
    """Close the global LLM service."""
    global _llm_service
    if _llm_service:
        await _llm_service.close()
        _llm_service = None



