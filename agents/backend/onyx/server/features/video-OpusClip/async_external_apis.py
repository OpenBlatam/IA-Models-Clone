"""
Async External API Operations for Video-OpusClip

Dedicated async functions for external API operations with connection pooling,
rate limiting, retry logic, and caching.
"""

import asyncio
import aiohttp
import aiofiles
import time
import json
import hashlib
from typing import (
    List, Dict, Any, Optional, Union, Tuple, AsyncIterator, 
    Callable, Awaitable, TypeVar, Protocol
)
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import structlog
from datetime import datetime, timedelta
import aioredis

logger = structlog.get_logger()

# Type variables
T = TypeVar('T')
R = TypeVar('R')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for external API operations."""
    base_url: str
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    enable_caching: bool = True
    cache_ttl: int = 300
    max_connections: int = 50
    max_connections_per_host: int = 10
    enable_ssl_verification: bool = True
    user_agent: str = "Video-OpusClip/1.0"
    headers: Dict[str, str] = field(default_factory=dict)

class APIType(Enum):
    """Types of external APIs."""
    YOUTUBE = "youtube"
    OPENAI = "openai"
    STABILITY_AI = "stability_ai"
    ELEVENLABS = "elevenlabs"
    CUSTOM = "custom"

class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

# =============================================================================
# ASYNC HTTP CLIENT WITH CONNECTION POOLING
# =============================================================================

class AsyncHTTPClient:
    """Async HTTP client with connection pooling and rate limiting."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_per_minute,
            requests_per_hour=config.rate_limit_per_hour
        )
        self.cache = None
        if config.enable_caching:
            self.cache = APICache(config.cache_ttl)
    
    async def initialize(self):
        """Initialize HTTP session."""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=self.config.enable_ssl_verification
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        headers.update(self.config.headers)
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        logger.info(f"HTTP client initialized for {self.config.base_url}")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
    
    async def request(
        self,
        method: HTTPMethod,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and caching."""
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Check cache for GET requests
        if method == HTTPMethod.GET and use_cache and cache_key and self.cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
        
        # Rate limiting
        await self.rate_limiter.wait_if_needed()
        
        # Prepare request
        request_headers = headers or {}
        
        try:
            async with self.session.request(
                method=method.value,
                url=url,
                json=data,
                params=params,
                headers=request_headers
            ) as response:
                
                response.raise_for_status()
                result = await response.json()
                
                # Cache GET responses
                if method == HTTPMethod.GET and use_cache and cache_key and self.cache:
                    await self.cache.set(cache_key, result)
                
                logger.debug(
                    f"API request successful",
                    method=method.value,
                    url=url,
                    status=response.status
                )
                
                return result
                
        except aiohttp.ClientError as e:
            logger.error(
                f"API request failed",
                method=method.value,
                url=url,
                error=str(e)
            )
            raise
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request(HTTPMethod.GET, endpoint, params=params, cache_key=cache_key, use_cache=use_cache)
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request(HTTPMethod.POST, endpoint, data=data, params=params)
    
    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.request(HTTPMethod.PUT, endpoint, data=data, params=params)
    
    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.request(HTTPMethod.DELETE, endpoint, params=params)

# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = []
        self.hour_requests = []
        self._lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        async with self._lock:
            now = time.time()
            
            # Clean old requests
            self.minute_requests = [req for req in self.minute_requests if now - req < 60]
            self.hour_requests = [req for req in self.hour_requests if now - req < 3600]
            
            # Check minute limit
            if len(self.minute_requests) >= self.requests_per_minute:
                wait_time = 60 - (now - self.minute_requests[0])
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Check hour limit
            if len(self.hour_requests) >= self.requests_per_hour:
                wait_time = 3600 - (now - self.hour_requests[0])
                if wait_time > 0:
                    logger.warning(f"Hourly rate limit exceeded, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Add current request
            self.minute_requests.append(now)
            self.hour_requests.append(now)

# =============================================================================
# API CACHING
# =============================================================================

class APICache:
    """Cache for API responses."""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        async with self._lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    async def set(self, key: str, value: Any):
        """Set cached value."""
        async with self._lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    async def clear(self):
        """Clear cache."""
        async with self._lock:
            self.cache.clear()
            self.timestamps.clear()

# =============================================================================
# EXTERNAL API OPERATIONS
# =============================================================================

class AsyncExternalAPIOperations:
    """Dedicated async external API operations."""
    
    def __init__(self, http_client: AsyncHTTPClient):
        self.client = http_client
        self.metrics = {
            "requests_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_response_time": 0.0
        }
    
    async def make_request(
        self,
        method: HTTPMethod,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        retry_on_error: bool = True
    ) -> Dict[str, Any]:
        """Make API request with retry logic and metrics."""
        start_time = time.perf_counter()
        
        # Check cache
        if method == HTTPMethod.GET and use_cache and cache_key and self.client.cache:
            cached_result = await self.client.cache.get(cache_key)
            if cached_result is not None:
                self.metrics["cache_hits"] += 1
                return cached_result
        
        self.metrics["cache_misses"] += 1
        
        # Make request with retry logic
        for attempt in range(self.client.config.max_retries):
            try:
                result = await self.client.request(
                    method, endpoint, data, params, cache_key=cache_key, use_cache=use_cache
                )
                
                # Update metrics
                response_time = time.perf_counter() - start_time
                self.metrics["requests_made"] += 1
                self.metrics["total_response_time"] += response_time
                
                logger.debug(
                    f"API request successful",
                    method=method.value,
                    endpoint=endpoint,
                    response_time=f"{response_time:.3f}s",
                    attempt=attempt + 1
                )
                
                return result
                
            except Exception as e:
                self.metrics["errors"] += 1
                
                if attempt < self.client.config.max_retries - 1 and retry_on_error:
                    wait_time = self.client.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"API request failed, retrying in {wait_time}s",
                        method=method.value,
                        endpoint=endpoint,
                        error=str(e),
                        attempt=attempt + 1
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"API request failed after {self.client.config.max_retries} attempts",
                        method=method.value,
                        endpoint=endpoint,
                        error=str(e)
                    )
                    raise

# =============================================================================
# YOUTUBE API OPERATIONS
# =============================================================================

class AsyncYouTubeAPI:
    """Dedicated async YouTube API operations."""
    
    def __init__(self, api_operations: AsyncExternalAPIOperations):
        self.api = api_operations
    
    async def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get video information from YouTube."""
        endpoint = f"videos?id={video_id}&part=snippet,contentDetails,statistics"
        cache_key = f"youtube:video:{video_id}"
        
        result = await self.api.make_request(
            HTTPMethod.GET,
            endpoint,
            cache_key=cache_key,
            use_cache=True
        )
        
        return result
    
    async def get_video_captions(self, video_id: str, language: str = "en") -> List[Dict[str, Any]]:
        """Get video captions from YouTube."""
        endpoint = f"captions?videoId={video_id}&part=snippet"
        cache_key = f"youtube:captions:{video_id}:{language}"
        
        result = await self.api.make_request(
            HTTPMethod.GET,
            endpoint,
            cache_key=cache_key,
            use_cache=True
        )
        
        # Filter captions by language
        captions = []
        for item in result.get("items", []):
            if item["snippet"]["language"] == language:
                captions.append(item)
        
        return captions
    
    async def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for videos on YouTube."""
        endpoint = f"search?q={query}&part=snippet&maxResults={max_results}&type=video"
        cache_key = f"youtube:search:{hashlib.md5(query.encode()).hexdigest()}"
        
        result = await self.api.make_request(
            HTTPMethod.GET,
            endpoint,
            cache_key=cache_key,
            use_cache=True
        )
        
        return result.get("items", [])
    
    async def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get videos from a specific channel."""
        endpoint = f"search?channelId={channel_id}&part=snippet&maxResults={max_results}&type=video&order=date"
        cache_key = f"youtube:channel:{channel_id}:videos"
        
        result = await self.api.make_request(
            HTTPMethod.GET,
            endpoint,
            cache_key=cache_key,
            use_cache=True
        )
        
        return result.get("items", [])

# =============================================================================
# OPENAI API OPERATIONS
# =============================================================================

class AsyncOpenAIAPI:
    """Dedicated async OpenAI API operations."""
    
    def __init__(self, api_operations: AsyncExternalAPIOperations):
        self.api = api_operations
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using OpenAI API."""
        endpoint = "chat/completions"
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        result = await self.api.make_request(
            HTTPMethod.POST,
            endpoint,
            data=data,
            use_cache=False  # Don't cache AI responses
        )
        
        return result
    
    async def generate_captions(
        self,
        audio_text: str,
        style: str = "casual",
        language: str = "en"
    ) -> List[str]:
        """Generate captions from audio text."""
        prompt = f"""
        Generate engaging captions for a video based on this audio transcript:
        "{audio_text}"
        
        Style: {style}
        Language: {language}
        
        Generate 3-5 captions that are:
        1. Engaging and viral-worthy
        2. Under 60 characters each
        3. Include relevant emojis
        4. Optimized for social media
        
        Return only the captions, one per line.
        """
        
        result = await self.generate_text(prompt, max_tokens=500, temperature=0.8)
        
        # Parse captions from response
        content = result["choices"][0]["message"]["content"]
        captions = [line.strip() for line in content.split('\n') if line.strip()]
        
        return captions
    
    async def analyze_video_content(
        self,
        video_description: str,
        video_title: str
    ) -> Dict[str, Any]:
        """Analyze video content for optimization."""
        prompt = f"""
        Analyze this video content for optimization:
        
        Title: {video_title}
        Description: {video_description}
        
        Provide analysis in JSON format with:
        1. Content type (educational, entertainment, tutorial, etc.)
        2. Target audience
        3. Viral potential score (1-10)
        4. Recommended hashtags
        5. Optimal posting time
        6. Suggested improvements
        """
        
        result = await self.generate_text(prompt, max_tokens=800, temperature=0.5)
        
        # Parse JSON response
        content = result["choices"][0]["message"]["content"]
        try:
            analysis = json.loads(content)
            return analysis
        except json.JSONDecodeError:
            logger.warning("Failed to parse OpenAI response as JSON")
            return {"error": "Failed to parse response"}

# =============================================================================
# STABILITY AI API OPERATIONS
# =============================================================================

class AsyncStabilityAIAPI:
    """Dedicated async Stability AI API operations."""
    
    def __init__(self, api_operations: AsyncExternalAPIOperations):
        self.api = api_operations
    
    async def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg_scale: float = 7.0
    ) -> Dict[str, Any]:
        """Generate image using Stability AI."""
        endpoint = "v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        data = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "samples": 1,
            "steps": steps
        }
        
        result = await self.api.make_request(
            HTTPMethod.POST,
            endpoint,
            data=data,
            use_cache=False
        )
        
        return result
    
    async def generate_video_thumbnail(
        self,
        video_title: str,
        video_description: str
    ) -> Dict[str, Any]:
        """Generate video thumbnail."""
        prompt = f"""
        Create an eye-catching thumbnail for a video titled "{video_title}".
        
        Description: {video_description}
        
        Style: Modern, vibrant, attention-grabbing, suitable for social media
        """
        
        return await self.generate_image(prompt, width=1280, height=720)

# =============================================================================
# ELEVENLABS API OPERATIONS
# =============================================================================

class AsyncElevenLabsAPI:
    """Dedicated async ElevenLabs API operations."""
    
    def __init__(self, api_operations: AsyncExternalAPIOperations):
        self.api = api_operations
    
    async def text_to_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_monolingual_v1"
    ) -> Dict[str, Any]:
        """Convert text to speech."""
        endpoint = f"text-to-speech/{voice_id}"
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        result = await self.api.make_request(
            HTTPMethod.POST,
            endpoint,
            data=data,
            use_cache=False
        )
        
        return result
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voices."""
        endpoint = "voices"
        cache_key = "elevenlabs:voices"
        
        result = await self.api.make_request(
            HTTPMethod.GET,
            endpoint,
            cache_key=cache_key,
            use_cache=True
        )
        
        return result.get("voices", [])
    
    async def generate_video_narration(
        self,
        script: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    ) -> Dict[str, Any]:
        """Generate narration for video."""
        # Split script into chunks for better processing
        chunks = [script[i:i+500] for i in range(0, len(script), 500)]
        
        audio_segments = []
        for chunk in chunks:
            result = await self.text_to_speech(chunk, voice_id)
            audio_segments.append(result)
        
        return {
            "audio_segments": audio_segments,
            "total_segments": len(audio_segments)
        }

# =============================================================================
# BATCH API OPERATIONS
# =============================================================================

class AsyncBatchAPIOperations:
    """Batch API operations for improved performance."""
    
    def __init__(self, api_operations: AsyncExternalAPIOperations):
        self.api = api_operations
    
    async def batch_get_video_info(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch get video information."""
        if not video_ids:
            return []
        
        # YouTube API allows up to 50 video IDs per request
        batch_size = 50
        results = []
        
        for i in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[i:i + batch_size]
            video_ids_str = ",".join(batch_ids)
            
            endpoint = f"videos?id={video_ids_str}&part=snippet,contentDetails,statistics"
            cache_key = f"youtube:batch:{hashlib.md5(video_ids_str.encode()).hexdigest()}"
            
            result = await self.api.make_request(
                HTTPMethod.GET,
                endpoint,
                cache_key=cache_key,
                use_cache=True
            )
            
            results.extend(result.get("items", []))
        
        return results
    
    async def batch_generate_captions(
        self,
        audio_texts: List[str],
        style: str = "casual"
    ) -> List[List[str]]:
        """Batch generate captions for multiple audio texts."""
        tasks = []
        for audio_text in audio_texts:
            task = self._generate_single_caption(audio_text, style)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        captions = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Caption generation failed: {result}")
                captions.append([])
            else:
                captions.append(result)
        
        return captions
    
    async def _generate_single_caption(self, audio_text: str, style: str) -> List[str]:
        """Generate captions for a single audio text."""
        # This would use the OpenAI API
        # For now, return mock captions
        return [
            f"Amazing content about {audio_text[:20]}... 🎬",
            f"Check out this incredible {style} video! 🔥",
            f"You won't believe what happens next! 😱"
        ]

# =============================================================================
# API METRICS AND MONITORING
# =============================================================================

class APIMetricsCollector:
    """Collect and analyze API metrics."""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_response_time": 0.0,
            "requests_by_endpoint": {},
            "errors_by_type": {}
        }
        self._lock = asyncio.Lock()
    
    async def record_request(
        self,
        endpoint: str,
        method: str,
        success: bool,
        response_time: float,
        error_type: Optional[str] = None
    ):
        """Record API request metrics."""
        async with self._lock:
            self.metrics["total_requests"] += 1
            self.metrics["total_response_time"] += response_time
            
            if success:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
                if error_type:
                    self.metrics["errors_by_type"][error_type] = \
                        self.metrics["errors_by_type"].get(error_type, 0) + 1
            
            # Track requests by endpoint
            endpoint_key = f"{method} {endpoint}"
            self.metrics["requests_by_endpoint"][endpoint_key] = \
                self.metrics["requests_by_endpoint"].get(endpoint_key, 0) + 1
    
    async def record_cache_access(self, hit: bool):
        """Record cache access."""
        async with self._lock:
            if hit:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        async with self._lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            if metrics["total_requests"] > 0:
                metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
                metrics["avg_response_time"] = metrics["total_response_time"] / metrics["total_requests"]
            
            total_cache_access = metrics["cache_hits"] + metrics["cache_misses"]
            if total_cache_access > 0:
                metrics["cache_hit_rate"] = metrics["cache_hits"] / total_cache_access
            
            return metrics

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_api_config(**kwargs) -> APIConfig:
    """Create API configuration."""
    return APIConfig(**kwargs)

def create_async_http_client(config: APIConfig) -> AsyncHTTPClient:
    """Create async HTTP client."""
    return AsyncHTTPClient(config)

def create_async_external_api_operations(http_client: AsyncHTTPClient) -> AsyncExternalAPIOperations:
    """Create async external API operations."""
    return AsyncExternalAPIOperations(http_client)

def create_async_youtube_api(api_operations: AsyncExternalAPIOperations) -> AsyncYouTubeAPI:
    """Create async YouTube API operations."""
    return AsyncYouTubeAPI(api_operations)

def create_async_openai_api(api_operations: AsyncExternalAPIOperations) -> AsyncOpenAIAPI:
    """Create async OpenAI API operations."""
    return AsyncOpenAIAPI(api_operations)

def create_async_stability_ai_api(api_operations: AsyncExternalAPIOperations) -> AsyncStabilityAIAPI:
    """Create async Stability AI API operations."""
    return AsyncStabilityAIAPI(api_operations)

def create_async_elevenlabs_api(api_operations: AsyncExternalAPIOperations) -> AsyncElevenLabsAPI:
    """Create async ElevenLabs API operations."""
    return AsyncElevenLabsAPI(api_operations)

def create_async_batch_api_operations(api_operations: AsyncExternalAPIOperations) -> AsyncBatchAPIOperations:
    """Create async batch API operations."""
    return AsyncBatchAPIOperations(api_operations)

def create_api_metrics_collector() -> APIMetricsCollector:
    """Create API metrics collector."""
    return APIMetricsCollector()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def setup_external_api(
    api_type: APIType,
    base_url: str,
    api_key: Optional[str] = None,
    **kwargs
) -> AsyncExternalAPIOperations:
    """Setup external API with default configuration."""
    config = create_api_config(
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )
    
    http_client = create_async_http_client(config)
    await http_client.initialize()
    
    return create_async_external_api_operations(http_client)

async def close_external_api(api_operations: AsyncExternalAPIOperations):
    """Close external API connection."""
    await api_operations.client.close()

def get_api_metrics(api_operations: AsyncExternalAPIOperations) -> Dict[str, Any]:
    """Get API operation metrics."""
    return api_operations.metrics.copy() 