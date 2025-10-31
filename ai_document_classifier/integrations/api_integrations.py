"""
API Integrations System
=======================

Advanced API integration system with support for multiple services,
rate limiting, caching, and error handling.
"""

import logging
import asyncio
import aiohttp
import httpx
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import time
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    """Integration status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"

class APIProvider(Enum):
    """API providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    AMAZON = "amazon"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    ELEVENLABS = "elevenlabs"
    GRAMMARLY = "grammarly"
    COPYSCAPE = "copyscape"
    TRANSLATE = "translate"
    CUSTOM = "custom"

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    name: str
    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    rate_limit: int = 100  # requests per minute
    authentication: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIResponse:
    """API response wrapper"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: float = 0.0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    provider: APIProvider
    name: str
    base_url: str
    api_key: str
    endpoints: Dict[str, APIEndpoint] = field(default_factory=dict)
    status: IntegrationStatus = IntegrationStatus.ACTIVE
    priority: int = 1  # 1 = highest priority
    fallback_providers: List[str] = field(default_factory=list)
    cache_ttl: int = 3600  # seconds
    retry_delay: int = 1  # seconds
    max_retries: int = 3

class APIIntegrationManager:
    """
    Advanced API integration manager
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize API integration manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file) if config_file else Path(__file__).parent / "api_config.yaml"
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Load configurations
        self._load_configurations()
        
        # Initialize rate limiters
        self._initialize_rate_limiters()
    
    def _load_configurations(self):
        """Load API configurations from file"""
        if not self.config_file.exists():
            self._create_default_config()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for integration_name, config in config_data.get("integrations", {}).items():
                endpoints = {}
                for endpoint_name, endpoint_config in config.get("endpoints", {}).items():
                    endpoint = APIEndpoint(
                        name=endpoint_name,
                        url=endpoint_config["url"],
                        method=endpoint_config.get("method", "POST"),
                        headers=endpoint_config.get("headers", {}),
                        timeout=endpoint_config.get("timeout", 30),
                        retry_count=endpoint_config.get("retry_count", 3),
                        rate_limit=endpoint_config.get("rate_limit", 100),
                        authentication=endpoint_config.get("authentication", {})
                    )
                    endpoints[endpoint_name] = endpoint
                
                integration = IntegrationConfig(
                    provider=APIProvider(config["provider"]),
                    name=integration_name,
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                    endpoints=endpoints,
                    status=IntegrationStatus(config.get("status", "active")),
                    priority=config.get("priority", 1),
                    fallback_providers=config.get("fallback_providers", []),
                    cache_ttl=config.get("cache_ttl", 3600),
                    retry_delay=config.get("retry_delay", 1),
                    max_retries=config.get("max_retries", 3)
                )
                
                self.integrations[integration_name] = integration
                
        except Exception as e:
            logger.error(f"Error loading API configurations: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default API configuration"""
        default_config = {
            "integrations": {
                "openai": {
                    "provider": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "your_openai_api_key_here",
                    "status": "active",
                    "priority": 1,
                    "endpoints": {
                        "chat_completion": {
                            "url": "/chat/completions",
                            "method": "POST",
                            "rate_limit": 60,
                            "timeout": 30
                        },
                        "text_completion": {
                            "url": "/completions",
                            "method": "POST",
                            "rate_limit": 60,
                            "timeout": 30
                        },
                        "embeddings": {
                            "url": "/embeddings",
                            "method": "POST",
                            "rate_limit": 100,
                            "timeout": 30
                        }
                    }
                },
                "huggingface": {
                    "provider": "huggingface",
                    "base_url": "https://api-inference.huggingface.co",
                    "api_key": "your_huggingface_api_key_here",
                    "status": "active",
                    "priority": 2,
                    "endpoints": {
                        "text_generation": {
                            "url": "/models/gpt2",
                            "method": "POST",
                            "rate_limit": 30,
                            "timeout": 30
                        },
                        "text_classification": {
                            "url": "/models/distilbert-base-uncased-finetuned-sst-2-english",
                            "method": "POST",
                            "rate_limit": 30,
                            "timeout": 30
                        }
                    }
                },
                "google_translate": {
                    "provider": "google",
                    "base_url": "https://translation.googleapis.com/language/translate/v2",
                    "api_key": "your_google_api_key_here",
                    "status": "active",
                    "priority": 3,
                    "endpoints": {
                        "translate": {
                            "url": "",
                            "method": "POST",
                            "rate_limit": 100,
                            "timeout": 30
                        }
                    }
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default API configuration at {self.config_file}")
    
    def _initialize_rate_limiters(self):
        """Initialize rate limiters for each integration"""
        for integration_name, integration in self.integrations.items():
            self.rate_limiters[integration_name] = {
                "requests": [],
                "last_reset": datetime.now(),
                "current_count": 0
            }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, integration_name: str, endpoint_name: str) -> bool:
        """Check if request is within rate limit"""
        if integration_name not in self.rate_limiters:
            return True
        
        integration = self.integrations.get(integration_name)
        if not integration:
            return True
        
        endpoint = integration.endpoints.get(endpoint_name)
        if not endpoint:
            return True
        
        rate_limiter = self.rate_limiters[integration_name]
        now = datetime.now()
        
        # Reset counter if minute has passed
        if (now - rate_limiter["last_reset"]).total_seconds() >= 60:
            rate_limiter["current_count"] = 0
            rate_limiter["last_reset"] = now
        
        # Check if within limit
        if rate_limiter["current_count"] >= endpoint.rate_limit:
            return False
        
        rate_limiter["current_count"] += 1
        return True
    
    def _get_cache_key(self, integration_name: str, endpoint_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        cache_data = {
            "integration": integration_name,
            "endpoint": endpoint_name,
            "params": params
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[APIResponse]:
        """Get cached response if available and not expired"""
        if cache_key not in self.cache:
            return None
        
        cached_data = self.cache[cache_key]
        if datetime.now() > cached_data["expires_at"]:
            del self.cache[cache_key]
            return None
        
        return cached_data["response"]
    
    def _cache_response(self, cache_key: str, response: APIResponse, ttl: int):
        """Cache API response"""
        self.cache[cache_key] = {
            "response": response,
            "expires_at": datetime.now() + timedelta(seconds=ttl)
        }
    
    async def make_request(
        self,
        integration_name: str,
        endpoint_name: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        timeout: Optional[int] = None
    ) -> APIResponse:
        """
        Make API request with rate limiting and caching
        
        Args:
            integration_name: Name of integration
            endpoint_name: Name of endpoint
            data: Request data
            params: Query parameters
            use_cache: Whether to use caching
            timeout: Request timeout
            
        Returns:
            API response
        """
        if not self.session:
            raise RuntimeError("API integration manager not initialized. Use async context manager.")
        
        # Check if integration exists
        if integration_name not in self.integrations:
            return APIResponse(
                success=False,
                error=f"Integration {integration_name} not found"
            )
        
        integration = self.integrations[integration_name]
        
        # Check if integration is active
        if integration.status != IntegrationStatus.ACTIVE:
            return APIResponse(
                success=False,
                error=f"Integration {integration_name} is not active"
            )
        
        # Check if endpoint exists
        if endpoint_name not in integration.endpoints:
            return APIResponse(
                success=False,
                error=f"Endpoint {endpoint_name} not found in integration {integration_name}"
            )
        
        endpoint = integration.endpoints[endpoint_name]
        
        # Check rate limit
        if not self._check_rate_limit(integration_name, endpoint_name):
            return APIResponse(
                success=False,
                error=f"Rate limit exceeded for {integration_name}/{endpoint_name}"
            )
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(integration_name, endpoint_name, data or {})
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        # Prepare request
        url = f"{integration.base_url}{endpoint.url}"
        headers = endpoint.headers.copy()
        headers["Authorization"] = f"Bearer {integration.api_key}"
        
        if data:
            headers["Content-Type"] = "application/json"
        
        # Make request with retries
        start_time = time.time()
        last_error = None
        
        for attempt in range(integration.max_retries):
            try:
                async with self.session.request(
                    method=endpoint.method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=timeout or endpoint.timeout
                ) as response:
                    
                    response_time = time.time() - start_time
                    response_data = await response.json() if response.content_type == "application/json" else await response.text()
                    
                    api_response = APIResponse(
                        success=response.status < 400,
                        data=response_data,
                        status_code=response.status,
                        response_time=response_time,
                        rate_limit_remaining=int(response.headers.get("X-RateLimit-Remaining", 0)),
                        metadata={
                            "integration": integration_name,
                            "endpoint": endpoint_name,
                            "attempt": attempt + 1
                        }
                    )
                    
                    if not api_response.success:
                        api_response.error = f"HTTP {response.status}: {response_data}"
                    
                    # Cache successful responses
                    if use_cache and api_response.success:
                        cache_key = self._get_cache_key(integration_name, endpoint_name, data or {})
                        self._cache_response(cache_key, api_response, integration.cache_ttl)
                    
                    return api_response
                    
            except asyncio.TimeoutError:
                last_error = "Request timeout"
            except Exception as e:
                last_error = str(e)
            
            # Wait before retry
            if attempt < integration.max_retries - 1:
                await asyncio.sleep(integration.retry_delay * (2 ** attempt))
        
        return APIResponse(
            success=False,
            error=f"Request failed after {integration.max_retries} attempts: {last_error}",
            response_time=time.time() - start_time
        )
    
    async def classify_document_with_ai(
        self,
        document_content: str,
        providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classify document using AI providers
        
        Args:
            document_content: Document content to classify
            providers: List of providers to use (in priority order)
            
        Returns:
            Classification results
        """
        if not providers:
            providers = ["openai", "huggingface"]
        
        results = {}
        
        for provider in providers:
            if provider not in self.integrations:
                continue
            
            try:
                if provider == "openai":
                    response = await self.make_request(
                        "openai",
                        "chat_completion",
                        data={
                            "model": "gpt-3.5-turbo",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a document classification expert. Classify the following document and return the result in JSON format with document_type, confidence, and reasoning fields."
                                },
                                {
                                    "role": "user",
                                    "content": f"Classify this document: {document_content[:1000]}"
                                }
                            ],
                            "max_tokens": 200,
                            "temperature": 0.1
                        }
                    )
                    
                    if response.success:
                        results[provider] = {
                            "document_type": "novel",  # Simplified for demo
                            "confidence": 0.95,
                            "reasoning": "AI analysis completed",
                            "response_time": response.response_time
                        }
                
                elif provider == "huggingface":
                    response = await self.make_request(
                        "huggingface",
                        "text_classification",
                        data={"inputs": document_content[:512]}
                    )
                    
                    if response.success:
                        results[provider] = {
                            "document_type": "positive",  # Simplified for demo
                            "confidence": 0.88,
                            "reasoning": "Hugging Face analysis completed",
                            "response_time": response.response_time
                        }
                
            except Exception as e:
                logger.error(f"Error with provider {provider}: {e}")
                results[provider] = {
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    async def translate_document(
        self,
        text: str,
        target_language: str = "es",
        source_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Translate document using translation services
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            Translation results
        """
        try:
            response = await self.make_request(
                "google_translate",
                "translate",
                params={
                    "key": self.integrations["google_translate"].api_key,
                    "q": text,
                    "target": target_language,
                    "source": source_language
                }
            )
            
            if response.success:
                return {
                    "translated_text": f"Translated: {text}",  # Simplified for demo
                    "source_language": source_language,
                    "target_language": target_language,
                    "confidence": 0.95,
                    "response_time": response.response_time
                }
            else:
                return {
                    "error": response.error,
                    "success": False
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    async def check_grammar(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Check grammar using grammar services
        
        Args:
            text: Text to check
            
        Returns:
            Grammar check results
        """
        # This would integrate with Grammarly API or similar service
        return {
            "grammar_score": 85,
            "suggestions": [
                "Consider using 'their' instead of 'there'",
                "Add a comma after 'however'"
            ],
            "error_count": 2,
            "success": True
        }
    
    async def check_plagiarism(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Check plagiarism using plagiarism detection services
        
        Args:
            text: Text to check
            
        Returns:
            Plagiarism check results
        """
        # This would integrate with Copyscape or similar service
        return {
            "plagiarism_score": 5.2,
            "is_original": True,
            "sources_found": 0,
            "similarity_percentage": 5.2,
            "success": True
        }
    
    def get_integration_status(self, integration_name: str) -> Dict[str, Any]:
        """Get status of specific integration"""
        if integration_name not in self.integrations:
            return {"error": "Integration not found"}
        
        integration = self.integrations[integration_name]
        rate_limiter = self.rate_limiters.get(integration_name, {})
        
        return {
            "name": integration.name,
            "provider": integration.provider.value,
            "status": integration.status.value,
            "priority": integration.priority,
            "endpoints": list(integration.endpoints.keys()),
            "rate_limit_info": {
                "current_count": rate_limiter.get("current_count", 0),
                "last_reset": rate_limiter.get("last_reset"),
                "endpoints": {
                    name: {"rate_limit": endpoint.rate_limit}
                    for name, endpoint in integration.endpoints.items()
                }
            },
            "cache_size": len([k for k in self.cache.keys() if k.startswith(integration_name)])
        }
    
    def get_all_integrations_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        return {
            integration_name: self.get_integration_status(integration_name)
            for integration_name in self.integrations.keys()
        }
    
    def clear_cache(self, integration_name: Optional[str] = None):
        """Clear cache for specific integration or all"""
        if integration_name:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(integration_name)]
            for key in keys_to_remove:
                del self.cache[key]
        else:
            self.cache.clear()
    
    def update_integration_status(self, integration_name: str, status: IntegrationStatus):
        """Update integration status"""
        if integration_name in self.integrations:
            self.integrations[integration_name].status = status
            logger.info(f"Updated {integration_name} status to {status.value}")

# Example usage
if __name__ == "__main__":
    async def main():
        async with APIIntegrationManager() as api_manager:
            # Get integration status
            status = api_manager.get_all_integrations_status()
            print("Integration Status:")
            for name, info in status.items():
                print(f"  {name}: {info['status']}")
            
            # Example: Classify document
            document = "This is a sample document about artificial intelligence and machine learning."
            classification = await api_manager.classify_document_with_ai(document)
            print(f"\nClassification Results: {classification}")
            
            # Example: Translate document
            translation = await api_manager.translate_document(document, "es")
            print(f"Translation Results: {translation}")
    
    # Run example
    asyncio.run(main())



























