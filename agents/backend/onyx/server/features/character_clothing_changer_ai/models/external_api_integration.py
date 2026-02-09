"""
External API Integration for Flux2 Clothing Changer
====================================================

Integration with external APIs and services.
"""

import requests
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """API provider enumeration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


@dataclass
class APIRequest:
    """API request information."""
    provider: APIProvider
    endpoint: str
    method: str
    headers: Dict[str, str]
    data: Dict[str, Any]
    timeout: float = 30.0
    retries: int = 3


@dataclass
class APIResponse:
    """API response information."""
    success: bool
    status_code: int
    data: Any
    response_time: float
    error: Optional[str] = None


class ExternalAPIIntegration:
    """External API integration system."""
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        default_retries: int = 3,
        enable_caching: bool = True,
    ):
        """
        Initialize external API integration.
        
        Args:
            default_timeout: Default timeout in seconds
            default_retries: Default number of retries
            enable_caching: Enable response caching
        """
        self.default_timeout = default_timeout
        self.default_retries = default_retries
        self.enable_caching = enable_caching
        
        self.api_configs: Dict[APIProvider, Dict[str, Any]] = {}
        self.request_history: List[APIRequest] = []
        self.response_cache: Dict[str, APIResponse] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
        }
    
    def configure_provider(
        self,
        provider: APIProvider,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Configure an API provider.
        
        Args:
            provider: API provider
            base_url: Base URL for API
            api_key: Optional API key
            headers: Optional default headers
        """
        self.api_configs[provider] = {
            "base_url": base_url,
            "api_key": api_key,
            "headers": headers or {},
        }
        logger.info(f"Configured provider: {provider.value}")
    
    def make_request(
        self,
        provider: APIProvider,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        use_cache: bool = True,
    ) -> APIResponse:
        """
        Make API request.
        
        Args:
            provider: API provider
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            headers: Optional headers
            timeout: Optional timeout
            retries: Optional retries
            use_cache: Use response cache
            
        Returns:
            API response
        """
        if provider not in self.api_configs:
            return APIResponse(
                success=False,
                status_code=0,
                data=None,
                response_time=0.0,
                error=f"Provider {provider.value} not configured",
            )
        
        config = self.api_configs[provider]
        url = f"{config['base_url']}/{endpoint.lstrip('/')}"
        
        # Check cache
        if use_cache and self.enable_caching:
            cache_key = self._generate_cache_key(provider, endpoint, method, data)
            if cache_key in self.response_cache:
                cached = self.response_cache[cache_key]
                logger.debug(f"Cache hit for {endpoint}")
                return cached
        
        # Prepare headers
        request_headers = config["headers"].copy()
        if config.get("api_key"):
            request_headers["Authorization"] = f"Bearer {config['api_key']}"
        if headers:
            request_headers.update(headers)
        
        # Prepare request
        timeout = timeout or self.default_timeout
        retries = retries or self.default_retries
        
        request = APIRequest(
            provider=provider,
            endpoint=endpoint,
            method=method,
            headers=request_headers,
            data=data or {},
            timeout=timeout,
            retries=retries,
        )
        
        # Execute with retries
        last_error = None
        for attempt in range(retries):
            try:
                start_time = time.time()
                
                if method.upper() == "GET":
                    response = requests.get(
                        url,
                        headers=request_headers,
                        params=data,
                        timeout=timeout,
                    )
                elif method.upper() == "POST":
                    response = requests.post(
                        url,
                        headers=request_headers,
                        json=data,
                        timeout=timeout,
                    )
                else:
                    response = requests.request(
                        method,
                        url,
                        headers=request_headers,
                        json=data,
                        timeout=timeout,
                    )
                
                response_time = time.time() - start_time
                
                # Parse response
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
                
                api_response = APIResponse(
                    success=response.status_code < 400,
                    status_code=response.status_code,
                    data=response_data,
                    response_time=response_time,
                    error=None if response.status_code < 400 else response_data,
                )
                
                # Update statistics
                self.stats["total_requests"] += 1
                if api_response.success:
                    self.stats["successful_requests"] += 1
                else:
                    self.stats["failed_requests"] += 1
                self.stats["total_response_time"] += response_time
                
                # Cache response
                if use_cache and self.enable_caching and api_response.success:
                    cache_key = self._generate_cache_key(provider, endpoint, method, data)
                    self.response_cache[cache_key] = api_response
                
                self.request_history.append(request)
                return api_response
                
            except Exception as e:
                last_error = str(e)
                if attempt < retries - 1:
                    wait_time = 0.5 * (2 ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {retries} attempts failed: {e}")
        
        # All retries failed
        self.stats["total_requests"] += 1
        self.stats["failed_requests"] += 1
        
        return APIResponse(
            success=False,
            status_code=0,
            data=None,
            response_time=0.0,
            error=last_error,
        )
    
    def _generate_cache_key(
        self,
        provider: APIProvider,
        endpoint: str,
        method: str,
        data: Optional[Dict[str, Any]],
    ) -> str:
        """Generate cache key."""
        import hashlib
        key_data = {
            "provider": provider.value,
            "endpoint": endpoint,
            "method": method,
            "data": json.dumps(data or {}, sort_keys=True),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API integration statistics."""
        avg_response_time = (
            self.stats["total_response_time"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0.0
        )
        success_rate = (
            self.stats["successful_requests"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0.0
        )
        
        return {
            **self.stats,
            "avg_response_time": avg_response_time,
            "success_rate": success_rate,
            "cache_size": len(self.response_cache),
        }


