"""
LLM Client
==========

Advanced LLM client for TruthGPT system with multiple provider support.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import httpx
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs):
        """Generate streaming text from prompt."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": kwargs.get("model", "gpt-3.5-turbo"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs):
        """Generate streaming text using OpenAI API."""
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": kwargs.get("model", "gpt-3.5-turbo"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"OpenAI streaming generation failed: {str(e)}")
            raise

class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            timeout=30.0
        )
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API."""
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/messages",
                json={
                    "model": kwargs.get("model", "claude-3-sonnet-20240229"),
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["content"][0]["text"]
            else:
                raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Anthropic generation failed: {str(e)}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs):
        """Generate streaming text using Anthropic API."""
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json={
                    "model": kwargs.get("model", "claude-3-sonnet-20240229"),
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            chunk = json.loads(data)
                            if chunk.get("type") == "content_block_delta":
                                yield chunk.get("delta", {}).get("text", "")
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Anthropic streaming generation failed: {str(e)}")
            raise

class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://blatam-academy.com",
                "X-Title": "Bulk TruthGPT"
            },
            timeout=30.0
        )
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenRouter API."""
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": kwargs.get("model", "openai/gpt-3.5-turbo"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "temperature": kwargs.get("temperature", 0.7)
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {str(e)}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs):
        """Generate streaming text using OpenRouter API."""
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": kwargs.get("model", "openai/gpt-3.5-turbo"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"OpenRouter streaming generation failed: {str(e)}")
            raise

class LLMClient:
    """
    Advanced LLM client with multiple provider support.
    
    Features:
    - Multiple provider support (OpenAI, Anthropic, OpenRouter)
    - Automatic failover
    - Rate limiting
    - Caching
    - Streaming support
    - Cost tracking
    """
    
    def __init__(self):
        self.providers = {}
        self.primary_provider = None
        self.fallback_providers = []
        self.rate_limiter = {}
        self.cost_tracker = {}
        self.request_count = 0
        
    async def initialize(self):
        """Initialize the LLM client."""
        logger.info("Initializing LLM Client...")
        
        try:
            # Initialize providers based on available API keys
            await self._initialize_providers()
            
            # Setup rate limiting
            await self._setup_rate_limiting()
            
            logger.info("LLM Client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Client: {str(e)}")
            raise
    
    async def _initialize_providers(self):
        """Initialize available providers."""
        import os
        
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.providers["openai"] = OpenAIProvider(openai_key)
            if not self.primary_provider:
                self.primary_provider = "openai"
            else:
                self.fallback_providers.append("openai")
            logger.info("OpenAI provider initialized")
        
        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.providers["anthropic"] = AnthropicProvider(anthropic_key)
            if not self.primary_provider:
                self.primary_provider = "anthropic"
            else:
                self.fallback_providers.append("anthropic")
            logger.info("Anthropic provider initialized")
        
        # OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            self.providers["openrouter"] = OpenRouterProvider(openrouter_key)
            if not self.primary_provider:
                self.primary_provider = "openrouter"
            else:
                self.fallback_providers.append("openrouter")
            logger.info("OpenRouter provider initialized")
        
        if not self.providers:
            raise Exception("No LLM providers configured. Please set API keys.")
    
    async def _setup_rate_limiting(self):
        """Setup rate limiting for each provider."""
        # Simple rate limiting configuration
        self.rate_limiter = {
            "openai": {"requests_per_minute": 60, "tokens_per_minute": 150000},
            "anthropic": {"requests_per_minute": 50, "tokens_per_minute": 100000},
            "openrouter": {"requests_per_minute": 100, "tokens_per_minute": 200000}
        }
    
    async def generate(
        self, 
        prompt: str, 
        model: str = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        provider: str = None,
        **kwargs
    ) -> str:
        """
        Generate text using the best available provider.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            provider: Specific provider to use
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        try:
            # Determine provider to use
            if provider and provider in self.providers:
                target_provider = provider
            else:
                target_provider = self.primary_provider
            
            # Check rate limits
            if not await self._check_rate_limit(target_provider, max_tokens):
                # Try fallback providers
                for fallback in self.fallback_providers:
                    if await self._check_rate_limit(fallback, max_tokens):
                        target_provider = fallback
                        break
                else:
                    raise Exception("All providers rate limited")
            
            # Generate text
            start_time = datetime.utcnow()
            
            result = await self.providers[target_provider].generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            await self._update_metrics(target_provider, max_tokens, generation_time)
            
            logger.info(f"Generated text using {target_provider} in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            
            # Try fallback providers
            for fallback in self.fallback_providers:
                try:
                    logger.info(f"Trying fallback provider: {fallback}")
                    result = await self.providers[fallback].generate(
                        prompt=prompt,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                    logger.info(f"Fallback generation successful with {fallback}")
                    return result
                except Exception as fallback_error:
                    logger.warning(f"Fallback provider {fallback} failed: {str(fallback_error)}")
                    continue
            
            raise Exception(f"All providers failed. Last error: {str(e)}")
    
    async def generate_stream(
        self, 
        prompt: str, 
        model: str = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        provider: str = None,
        **kwargs
    ):
        """
        Generate streaming text.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            provider: Specific provider to use
            **kwargs: Additional parameters
            
        Yields:
            Text chunks
        """
        try:
            # Determine provider to use
            if provider and provider in self.providers:
                target_provider = provider
            else:
                target_provider = self.primary_provider
            
            # Check rate limits
            if not await self._check_rate_limit(target_provider, max_tokens):
                raise Exception("Provider rate limited")
            
            # Generate streaming text
            async for chunk in self.providers[target_provider].generate_stream(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise
    
    async def _check_rate_limit(self, provider: str, tokens: int) -> bool:
        """Check if provider is within rate limits."""
        try:
            if provider not in self.rate_limiter:
                return True
            
            limits = self.rate_limiter[provider]
            
            # Simple rate limiting check
            # In production, this would use a proper rate limiter like Redis
            current_time = datetime.utcnow()
            minute_key = f"{provider}_minute_{current_time.minute}"
            
            if minute_key not in self.rate_limiter:
                self.rate_limiter[minute_key] = {"requests": 0, "tokens": 0}
            
            current_usage = self.rate_limiter[minute_key]
            
            if (current_usage["requests"] >= limits["requests_per_minute"] or 
                current_usage["tokens"] >= limits["tokens_per_minute"]):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            return True  # Allow if check fails
    
    async def _update_metrics(self, provider: str, tokens: int, generation_time: float):
        """Update provider metrics."""
        try:
            if provider not in self.cost_tracker:
                self.cost_tracker[provider] = {
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_time": 0.0,
                    "average_time": 0.0
                }
            
            metrics = self.cost_tracker[provider]
            metrics["total_requests"] += 1
            metrics["total_tokens"] += tokens
            metrics["total_time"] += generation_time
            metrics["average_time"] = metrics["total_time"] / metrics["total_requests"]
            
            self.request_count += 1
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {str(e)}")
    
    async def get_provider_metrics(self) -> Dict[str, Any]:
        """Get metrics for all providers."""
        return {
            "providers": self.cost_tracker,
            "total_requests": self.request_count,
            "available_providers": list(self.providers.keys()),
            "primary_provider": self.primary_provider,
            "fallback_providers": self.fallback_providers
        }
    
    async def get_available_models(self, provider: str = None) -> List[str]:
        """Get available models for a provider."""
        try:
            if provider and provider in self.providers:
                # In a real implementation, this would query the provider's API
                models = {
                    "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                    "anthropic": ["claude-3-sonnet-20240229", "claude-3-opus-20240229"],
                    "openrouter": ["openai/gpt-3.5-turbo", "openai/gpt-4", "anthropic/claude-3-sonnet"]
                }
                return models.get(provider, [])
            else:
                # Return all available models
                all_models = []
                for prov in self.providers:
                    models = await self.get_available_models(prov)
                    all_models.extend(models)
                return list(set(all_models))
                
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            for provider in self.providers.values():
                if hasattr(provider, 'client'):
                    await provider.client.aclose()
            
            logger.info("LLM Client cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to cleanup LLM Client: {str(e)}")











