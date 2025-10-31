"""
External AI API Integration System
=================================

This module provides integration with external AI APIs including:
- OpenAI API integration
- Anthropic Claude API integration
- Google AI API integration
- Hugging Face API integration
- Custom AI service integration
- Multi-provider load balancing
- API key management and rotation
- Rate limiting and cost tracking
- Performance monitoring across providers
"""

import asyncio
import logging
import json
import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import hashlib
from enum import Enum

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """AI provider enumeration"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class APIKey:
    """API key configuration"""
    provider: AIProvider
    key_id: str
    api_key: str
    enabled: bool = True
    rate_limit_per_minute: int = 1000
    cost_per_1k_tokens: float = 0.0
    max_tokens_per_request: int = 4000
    priority: int = 1  # Lower number = higher priority
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class APIRequest:
    """API request configuration"""
    provider: AIProvider
    model: str
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_count: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class APIResponse:
    """API response data"""
    provider: AIProvider
    model: str
    response_text: str
    tokens_used: int
    cost: float
    response_time: float
    success: bool
    error_message: Optional[str] = None
    request_id: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProviderStats:
    """Provider statistics"""
    provider: AIProvider
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExternalAIIntegration:
    """External AI API integration system"""
    
    def __init__(self):
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        
        # API configuration
        self.api_keys: Dict[str, APIKey] = {}
        self.provider_stats: Dict[AIProvider, ProviderStats] = {}
        self.load_balancing_enabled = True
        self.rate_limiting_enabled = True
        
        # Request tracking
        self.request_history: List[APIResponse] = []
        self.rate_limit_tracker: Dict[str, List[datetime]] = {}
        
        # Cost tracking
        self.daily_costs: Dict[str, float] = {}
        self.monthly_costs: Dict[str, float] = {}
        
        # Initialize providers
        self._initialize_providers()
        self._load_api_keys()
    
    def _initialize_providers(self):
        """Initialize provider statistics"""
        for provider in AIProvider:
            self.provider_stats[provider] = ProviderStats(provider=provider)
    
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        try:
            # OpenAI API keys
            openai_keys = os.getenv("OPENAI_API_KEYS", "").split(",")
            for i, key in enumerate(openai_keys):
                if key.strip():
                    api_key = APIKey(
                        provider=AIProvider.OPENAI,
                        key_id=f"openai_{i}",
                        api_key=key.strip(),
                        rate_limit_per_minute=3000,
                        cost_per_1k_tokens=0.03,
                        max_tokens_per_request=4000,
                        priority=1
                    )
                    self.api_keys[api_key.key_id] = api_key
            
            # Anthropic API keys
            anthropic_keys = os.getenv("ANTHROPIC_API_KEYS", "").split(",")
            for i, key in enumerate(anthropic_keys):
                if key.strip():
                    api_key = APIKey(
                        provider=AIProvider.ANTHROPIC,
                        key_id=f"anthropic_{i}",
                        api_key=key.strip(),
                        rate_limit_per_minute=1000,
                        cost_per_1k_tokens=0.015,
                        max_tokens_per_request=4000,
                        priority=2
                    )
                    self.api_keys[api_key.key_id] = api_key
            
            # Google AI API keys
            google_keys = os.getenv("GOOGLE_AI_API_KEYS", "").split(",")
            for i, key in enumerate(google_keys):
                if key.strip():
                    api_key = APIKey(
                        provider=AIProvider.GOOGLE,
                        key_id=f"google_{i}",
                        api_key=key.strip(),
                        rate_limit_per_minute=1500,
                        cost_per_1k_tokens=0.00125,
                        max_tokens_per_request=4000,
                        priority=3
                    )
                    self.api_keys[api_key.key_id] = api_key
            
            # Hugging Face API keys
            hf_keys = os.getenv("HUGGINGFACE_API_KEYS", "").split(",")
            for i, key in enumerate(hf_keys):
                if key.strip():
                    api_key = APIKey(
                        provider=AIProvider.HUGGINGFACE,
                        key_id=f"huggingface_{i}",
                        api_key=key.strip(),
                        rate_limit_per_minute=1000,
                        cost_per_1k_tokens=0.0,  # Free tier
                        max_tokens_per_request=4000,
                        priority=4
                    )
                    self.api_keys[api_key.key_id] = api_key
            
            logger.info(f"Loaded {len(self.api_keys)} API keys")
        
        except Exception as e:
            logger.error(f"Error loading API keys: {str(e)}")
    
    async def make_request(self, request: APIRequest) -> APIResponse:
        """Make a request to external AI API"""
        try:
            start_time = time.time()
            
            # Select best API key for the request
            api_key = self._select_api_key(request.provider)
            if not api_key:
                return APIResponse(
                    provider=request.provider,
                    model=request.model,
                    response_text="",
                    tokens_used=0,
                    cost=0.0,
                    response_time=0.0,
                    success=False,
                    error_message=f"No API key available for provider {request.provider.value}"
                )
            
            # Check rate limiting
            if self.rate_limiting_enabled and not self._check_rate_limit(api_key):
                return APIResponse(
                    provider=request.provider,
                    model=request.model,
                    response_text="",
                    tokens_used=0,
                    cost=0.0,
                    response_time=0.0,
                    success=False,
                    error_message="Rate limit exceeded"
                )
            
            # Make the actual API request
            response = await self._make_provider_request(request, api_key)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update statistics
            self._update_provider_stats(request.provider, response, response_time)
            
            # Track request
            self.request_history.append(response)
            
            # Update cost tracking
            self._update_cost_tracking(api_key, response.cost)
            
            # Record performance in analyzer
            await self._record_performance_metrics(request, response, response_time)
            
            return response
        
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return APIResponse(
                provider=request.provider,
                model=request.model,
                response_text="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _select_api_key(self, provider: AIProvider) -> Optional[APIKey]:
        """Select the best API key for a provider"""
        try:
            # Get available keys for the provider
            available_keys = [
                key for key in self.api_keys.values()
                if key.provider == provider and key.enabled
            ]
            
            if not available_keys:
                return None
            
            # Sort by priority and select the best one
            available_keys.sort(key=lambda x: x.priority)
            return available_keys[0]
        
        except Exception as e:
            logger.error(f"Error selecting API key: {str(e)}")
            return None
    
    def _check_rate_limit(self, api_key: APIKey) -> bool:
        """Check if API key is within rate limits"""
        try:
            current_time = datetime.now()
            key_id = api_key.key_id
            
            # Get recent requests for this key
            if key_id not in self.rate_limit_tracker:
                self.rate_limit_tracker[key_id] = []
            
            recent_requests = self.rate_limit_tracker[key_id]
            
            # Remove requests older than 1 minute
            cutoff_time = current_time - timedelta(minutes=1)
            recent_requests = [req_time for req_time in recent_requests if req_time > cutoff_time]
            self.rate_limit_tracker[key_id] = recent_requests
            
            # Check if within rate limit
            if len(recent_requests) >= api_key.rate_limit_per_minute:
                return False
            
            # Add current request
            recent_requests.append(current_time)
            return True
        
        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            return True  # Allow request on error
    
    async def _make_provider_request(self, request: APIRequest, api_key: APIKey) -> APIResponse:
        """Make request to specific provider"""
        try:
            if request.provider == AIProvider.OPENAI:
                return await self._make_openai_request(request, api_key)
            elif request.provider == AIProvider.ANTHROPIC:
                return await self._make_anthropic_request(request, api_key)
            elif request.provider == AIProvider.GOOGLE:
                return await self._make_google_request(request, api_key)
            elif request.provider == AIProvider.HUGGINGFACE:
                return await self._make_huggingface_request(request, api_key)
            else:
                raise ValueError(f"Unsupported provider: {request.provider}")
        
        except Exception as e:
            logger.error(f"Error making provider request: {str(e)}")
            return APIResponse(
                provider=request.provider,
                model=request.model,
                response_text="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _make_openai_request(self, request: APIRequest, api_key: APIKey) -> APIResponse:
        """Make request to OpenAI API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": request.model,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": min(request.max_tokens, api_key.max_tokens_per_request),
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        response_text = data["choices"][0]["message"]["content"]
                        tokens_used = data["usage"]["total_tokens"]
                        cost = (tokens_used / 1000) * api_key.cost_per_1k_tokens
                        
                        return APIResponse(
                            provider=request.provider,
                            model=request.model,
                            response_text=response_text,
                            tokens_used=tokens_used,
                            cost=cost,
                            response_time=0.0,  # Will be set by caller
                            success=True,
                            request_id=data.get("id")
                        )
                    else:
                        error_text = await response.text()
                        return APIResponse(
                            provider=request.provider,
                            model=request.model,
                            response_text="",
                            tokens_used=0,
                            cost=0.0,
                            response_time=0.0,
                            success=False,
                            error_message=f"HTTP {response.status}: {error_text}"
                        )
        
        except Exception as e:
            logger.error(f"Error making OpenAI request: {str(e)}")
            return APIResponse(
                provider=request.provider,
                model=request.model,
                response_text="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _make_anthropic_request(self, request: APIRequest, api_key: APIKey) -> APIResponse:
        """Make request to Anthropic API"""
        try:
            headers = {
                "x-api-key": api_key.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": request.model,
                "max_tokens": min(request.max_tokens, api_key.max_tokens_per_request),
                "temperature": request.temperature,
                "messages": [{"role": "user", "content": request.prompt}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        response_text = data["content"][0]["text"]
                        tokens_used = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
                        cost = (tokens_used / 1000) * api_key.cost_per_1k_tokens
                        
                        return APIResponse(
                            provider=request.provider,
                            model=request.model,
                            response_text=response_text,
                            tokens_used=tokens_used,
                            cost=cost,
                            response_time=0.0,
                            success=True,
                            request_id=data.get("id")
                        )
                    else:
                        error_text = await response.text()
                        return APIResponse(
                            provider=request.provider,
                            model=request.model,
                            response_text="",
                            tokens_used=0,
                            cost=0.0,
                            response_time=0.0,
                            success=False,
                            error_message=f"HTTP {response.status}: {error_text}"
                        )
        
        except Exception as e:
            logger.error(f"Error making Anthropic request: {str(e)}")
            return APIResponse(
                provider=request.provider,
                model=request.model,
                response_text="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _make_google_request(self, request: APIRequest, api_key: APIKey) -> APIResponse:
        """Make request to Google AI API"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{"parts": [{"text": request.prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": min(request.max_tokens, api_key.max_tokens_per_request),
                    "temperature": request.temperature,
                    "topP": request.top_p
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent?key={api_key.api_key}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                        # Estimate tokens (Google doesn't provide exact count)
                        tokens_used = len(request.prompt.split()) + len(response_text.split())
                        cost = (tokens_used / 1000) * api_key.cost_per_1k_tokens
                        
                        return APIResponse(
                            provider=request.provider,
                            model=request.model,
                            response_text=response_text,
                            tokens_used=tokens_used,
                            cost=cost,
                            response_time=0.0,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        return APIResponse(
                            provider=request.provider,
                            model=request.model,
                            response_text="",
                            tokens_used=0,
                            cost=0.0,
                            response_time=0.0,
                            success=False,
                            error_message=f"HTTP {response.status}: {error_text}"
                        )
        
        except Exception as e:
            logger.error(f"Error making Google request: {str(e)}")
            return APIResponse(
                provider=request.provider,
                model=request.model,
                response_text="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _make_huggingface_request(self, request: APIRequest, api_key: APIKey) -> APIResponse:
        """Make request to Hugging Face API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": min(request.max_tokens, api_key.max_tokens_per_request),
                    "temperature": request.temperature,
                    "top_p": request.top_p
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://api-inference.huggingface.co/models/{request.model}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if isinstance(data, list) and len(data) > 0:
                            response_text = data[0].get("generated_text", "")
                        else:
                            response_text = str(data)
                        
                        # Estimate tokens
                        tokens_used = len(request.prompt.split()) + len(response_text.split())
                        cost = 0.0  # Hugging Face is free
                        
                        return APIResponse(
                            provider=request.provider,
                            model=request.model,
                            response_text=response_text,
                            tokens_used=tokens_used,
                            cost=cost,
                            response_time=0.0,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        return APIResponse(
                            provider=request.provider,
                            model=request.model,
                            response_text="",
                            tokens_used=0,
                            cost=0.0,
                            response_time=0.0,
                            success=False,
                            error_message=f"HTTP {response.status}: {error_text}"
                        )
        
        except Exception as e:
            logger.error(f"Error making Hugging Face request: {str(e)}")
            return APIResponse(
                provider=request.provider,
                model=request.model,
                response_text="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _update_provider_stats(self, provider: AIProvider, response: APIResponse, response_time: float):
        """Update provider statistics"""
        try:
            stats = self.provider_stats[provider]
            
            stats.total_requests += 1
            stats.total_tokens += response.tokens_used
            stats.total_cost += response.cost
            stats.last_request_time = datetime.now()
            
            if response.success:
                stats.successful_requests += 1
            else:
                stats.failed_requests += 1
            
            # Update average response time
            if stats.total_requests == 1:
                stats.average_response_time = response_time
            else:
                stats.average_response_time = (
                    (stats.average_response_time * (stats.total_requests - 1) + response_time) /
                    stats.total_requests
                )
            
            # Update error rate
            stats.error_rate = stats.failed_requests / stats.total_requests
        
        except Exception as e:
            logger.error(f"Error updating provider stats: {str(e)}")
    
    def _update_cost_tracking(self, api_key: APIKey, cost: float):
        """Update cost tracking"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            month = datetime.now().strftime("%Y-%m")
            
            # Daily costs
            if today not in self.daily_costs:
                self.daily_costs[today] = 0.0
            self.daily_costs[today] += cost
            
            # Monthly costs
            if month not in self.monthly_costs:
                self.monthly_costs[month] = 0.0
            self.monthly_costs[month] += cost
        
        except Exception as e:
            logger.error(f"Error updating cost tracking: {str(e)}")
    
    async def _record_performance_metrics(self, request: APIRequest, response: APIResponse, response_time: float):
        """Record performance metrics in analyzer"""
        try:
            # Map provider to model type
            model_type = ModelType.TEXT_GENERATION
            
            # Record response time
            self.analyzer.record_performance(
                model_name=f"{request.provider.value}_{request.model}",
                model_type=model_type,
                metric=PerformanceMetric.RESPONSE_TIME,
                value=response_time,
                context=f"External API request to {request.provider.value}",
                metadata={
                    "provider": request.provider.value,
                    "model": request.model,
                    "tokens_used": response.tokens_used,
                    "cost": response.cost,
                    "success": response.success
                }
            )
            
            # Record cost efficiency (if we can calculate it)
            if response.tokens_used > 0 and response.cost > 0:
                cost_efficiency = 1.0 / (response.cost / response.tokens_used * 1000)  # Inverse of cost per token
                self.analyzer.record_performance(
                    model_name=f"{request.provider.value}_{request.model}",
                    model_type=model_type,
                    metric=PerformanceMetric.COST_EFFICIENCY,
                    value=cost_efficiency,
                    context=f"External API cost efficiency for {request.provider.value}",
                    metadata={
                        "provider": request.provider.value,
                        "model": request.model,
                        "tokens_used": response.tokens_used,
                        "cost": response.cost
                    }
                )
        
        except Exception as e:
            logger.error(f"Error recording performance metrics: {str(e)}")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            provider.value: {
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "total_tokens": stats.total_tokens,
                "total_cost": stats.total_cost,
                "average_response_time": stats.average_response_time,
                "error_rate": stats.error_rate,
                "last_request_time": stats.last_request_time.isoformat() if stats.last_request_time else None
            }
            for provider, stats in self.provider_stats.items()
        }
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        return {
            "daily_costs": self.daily_costs,
            "monthly_costs": self.monthly_costs,
            "total_daily_cost": sum(self.daily_costs.values()),
            "total_monthly_cost": sum(self.monthly_costs.values())
        }
    
    def get_request_history(self, limit: int = 100) -> List[APIResponse]:
        """Get recent request history"""
        return self.request_history[-limit:]
    
    def add_api_key(self, api_key: APIKey):
        """Add a new API key"""
        self.api_keys[api_key.key_id] = api_key
        logger.info(f"Added API key: {api_key.key_id}")
    
    def remove_api_key(self, key_id: str):
        """Remove an API key"""
        if key_id in self.api_keys:
            del self.api_keys[key_id]
            logger.info(f"Removed API key: {key_id}")
    
    def enable_provider(self, provider: AIProvider):
        """Enable a provider"""
        for api_key in self.api_keys.values():
            if api_key.provider == provider:
                api_key.enabled = True
        logger.info(f"Enabled provider: {provider.value}")
    
    def disable_provider(self, provider: AIProvider):
        """Disable a provider"""
        for api_key in self.api_keys.values():
            if api_key.provider == provider:
                api_key.enabled = False
        logger.info(f"Disabled provider: {provider.value}")


# Global external AI integration instance
_external_ai: Optional[ExternalAIIntegration] = None


def get_external_ai_integration() -> ExternalAIIntegration:
    """Get or create global external AI integration"""
    global _external_ai
    if _external_ai is None:
        _external_ai = ExternalAIIntegration()
    return _external_ai


# Example usage
async def main():
    """Example usage of external AI integration"""
    external_ai = get_external_ai_integration()
    
    # Make a request to OpenAI
    openai_request = APIRequest(
        provider=AIProvider.OPENAI,
        model="gpt-3.5-turbo",
        prompt="What is artificial intelligence?",
        max_tokens=100,
        temperature=0.7
    )
    
    response = await external_ai.make_request(openai_request)
    print(f"OpenAI Response: {response.response_text[:100]}...")
    print(f"Tokens used: {response.tokens_used}, Cost: ${response.cost:.4f}")
    
    # Make a request to Anthropic
    anthropic_request = APIRequest(
        provider=AIProvider.ANTHROPIC,
        model="claude-3-sonnet-20240229",
        prompt="Explain machine learning in simple terms.",
        max_tokens=150,
        temperature=0.5
    )
    
    response = await external_ai.make_request(anthropic_request)
    print(f"Anthropic Response: {response.response_text[:100]}...")
    print(f"Tokens used: {response.tokens_used}, Cost: ${response.cost:.4f}")
    
    # Get provider statistics
    stats = external_ai.get_provider_stats()
    print(f"Provider stats: {stats}")
    
    # Get cost summary
    costs = external_ai.get_cost_summary()
    print(f"Cost summary: {costs}")


if __name__ == "__main__":
    asyncio.run(main())

























