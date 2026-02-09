# Key Conventions v15 - Ultra-Optimized SEO Service

## Overview
This document outlines the essential coding conventions, architectural patterns, and best practices for the Ultra-Optimized SEO Service v15. These conventions ensure code consistency, maintainability, and optimal performance across the entire codebase.

## 1. Code Style Conventions

### 1.1 Python Code Style
```python
# ✅ CORRECT: Follow PEP 8 with optimizations
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncGenerator
from pydantic import BaseModel, Field, ConfigDict, computed_field

class SEOAnalysisService:
    """Ultra-optimized SEO analysis service with async operations."""
    
    def __init__(self, config: ConfigDict):
        self.config = config
        self._cache_manager = None  # Private attribute
    
    async def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze SEO for a single URL with caching."""
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        return await self._perform_analysis(url)
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format (private method)."""
        return url.startswith(('http://', 'https://'))

# ❌ INCORRECT: Avoid these patterns
class seoService:  # Wrong naming
    def __init__(self,Config):  # Wrong spacing
        self.Config=Config  # Wrong assignment
```

### 1.2 Naming Conventions
```python
# ✅ CORRECT: Clear, descriptive naming
class LazyLoadingConfig(BaseModel):
    chunk_size: int = Field(default=100)
    max_items: int = Field(default=10000)
    enable_streaming: bool = Field(default=True)

async def analyze_seo_content(params: AnalysisParamsModel) -> AnalysisResultModel:
    """Analyze SEO content with optimized processing."""
    pass

# Constants and configuration
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
CACHE_TTL = 3600

# Private methods and attributes
def _validate_input(self, data: Dict[str, Any]) -> bool:
    pass

# ❌ INCORRECT: Avoid unclear naming
class Config:  # Too generic
    def analyze(self, p):  # Unclear parameter name
        pass
```

### 1.3 Type Hints and Annotations
```python
# ✅ CORRECT: Comprehensive type hints
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncGenerator

async def process_bulk_analysis(
    params: BulkSEOParams,
    config: LazyLoadingConfig
) -> AsyncGenerator[BulkSEOResult, None]:
    """Process bulk SEO analysis with lazy loading."""
    pass

def calculate_seo_score(metrics: Dict[str, float]) -> float:
    """Calculate overall SEO score from metrics."""
    return sum(metrics.values()) / len(metrics)

# ❌ INCORRECT: Missing or unclear type hints
def process_data(data):  # No type hints
    pass

def analyze(url: str) -> dict:  # Too generic return type
    pass
```

## 2. Architectural Conventions

### 2.1 RORO Pattern (Receive Object, Return Object)
```python
# ✅ CORRECT: RORO pattern implementation
class CrawlParamsModel(BaseModel):
    url: str = Field(..., description="URL to crawl")
    depth: int = Field(default=2, ge=1, le=5)
    timeout: int = Field(default=30, ge=1, le=300)

class CrawlResultModel(BaseModel):
    url: str = Field(..., description="Crawled URL")
    title: Optional[str] = Field(None, description="Page title")
    status_code: int = Field(..., ge=100, le=599)

async def crawl_url(params: CrawlParamsModel) -> CrawlResultModel:
    """Crawl URL with structured input/output."""
    # Implementation
    return CrawlResultModel(url=params.url, status_code=200)

# ❌ INCORRECT: Avoid multiple parameters
async def crawl_url(url: str, depth: int, timeout: int) -> Dict[str, Any]:
    pass
```

### 2.2 Async/Await Conventions
```python
# ✅ CORRECT: Proper async patterns
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    """Analyze SEO with async operations."""
    try:
        # Parallel execution for I/O-bound tasks
        crawl_task = crawl_url(params)
        performance_task = analyze_performance(params)
        
        crawl_result, performance_result = await asyncio.gather(
            crawl_task, performance_task, return_exceptions=True
        )
        
        return await build_result(crawl_result, performance_result)
    
    except Exception as e:
        logger.error("SEO analysis failed", error=str(e))
        raise

# Pure functions (no I/O) should not be async
def calculate_score(metrics: Dict[str, float]) -> float:
    """Calculate score (pure function, no async needed)."""
    return sum(metrics.values()) / len(metrics)

# ❌ INCORRECT: Unnecessary async
async def calculate_score(metrics: Dict[str, float]) -> float:  # No I/O
    return sum(metrics.values()) / len(metrics)
```

### 2.3 Error Handling Conventions
```python
# ✅ CORRECT: Structured error handling
class SEOException(Exception):
    """Base exception for SEO service."""
    pass

class URLValidationError(SEOException):
    """URL validation error."""
    pass

class CrawlError(SEOException):
    """Crawl error."""
    pass

async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    """Analyze SEO with proper error handling."""
    try:
        if not is_valid_url(params.url):
            raise URLValidationError(f"Invalid URL: {params.url}")
        
        result = await perform_analysis(params)
        return result
    
    except URLValidationError:
        # Re-raise specific exceptions
        raise
    except Exception as e:
        # Log and wrap generic exceptions
        logger.error("Unexpected error in SEO analysis", error=str(e))
        raise SEOException(f"Analysis failed: {str(e)}")

# ❌ INCORRECT: Generic error handling
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    try:
        return await perform_analysis(params)
    except:  # Too broad
        return None
```

## 3. Performance Conventions

### 3.1 Caching Patterns
```python
# ✅ CORRECT: Optimized caching with fallback
class CacheManager:
    """Advanced caching with Redis and memory fallback."""
    
    async def get(self, key: str, cache_type: str = 'auto') -> Optional[Dict[str, Any]]:
        """Get cached result with intelligent fallback."""
        if cache_type == 'memory' or not self.redis_client:
            return self._get_from_memory(key)
        
        # Try Redis first
        try:
            cached = await self.redis_client.get(key)
            if cached:
                return orjson.loads(cached)  # Ultra-fast deserialization
        except Exception as e:
            logger.warning("Redis cache retrieval failed", error=str(e))
        
        # Fallback to memory cache
        return self._get_from_memory(key)
    
    async def set(self, key: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        """Set cached result with intelligent storage."""
        # Always store in memory for fast access
        self.memory_cache[key] = data
        self.memory_ttl[key] = time.time() + ttl
        
        # Store in Redis if available
        if self.redis_client:
            try:
                serialized_data = orjson.dumps(data)  # Ultra-fast serialization
                await self.redis_client.setex(key, ttl, serialized_data)
            except Exception as e:
                logger.warning("Redis cache storage failed", error=str(e))

# ❌ INCORRECT: Simple caching without fallback
def get_cached_data(key: str) -> Optional[Dict[str, Any]]:
    return redis_client.get(key)  # No fallback
```

### 3.2 Connection Pooling
```python
# ✅ CORRECT: Connection pooling for external services
class GlobalState:
    """Global state with connection pooling."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.http_client: Optional[httpx.AsyncClient] = None

async def get_redis() -> redis.Redis:
    """Get Redis client with async connection pooling."""
    if not state.redis_client:
        state.redis_client = redis.from_url(
            config.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            retry_on_timeout=True,
            health_check_interval=30
        )
    return state.redis_client

async def get_http_client() -> httpx.AsyncClient:
    """Get HTTP client with connection pooling."""
    if not state.http_client:
        state.http_client = httpx.AsyncClient(
            timeout=config.timeout,
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            ),
            http2=True,
            follow_redirects=True
        )
    return state.http_client

# ❌ INCORRECT: Creating new connections for each request
async def make_request(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:  # New connection each time
        return await client.get(url)
```

### 3.3 Parallel Execution
```python
# ✅ CORRECT: Parallel execution for I/O-bound tasks
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    """Analyze SEO with parallel execution."""
    start_time = time.perf_counter()
    
    # Parallel execution of independent tasks
    crawl_task = crawl_url(CrawlParamsModel(url=params.url))
    performance_task = analyze_performance(PerformanceParamsModel(url=params.url))
    
    crawl_result, performance_result = await asyncio.gather(
        crawl_task, performance_task, return_exceptions=True
    )
    
    # Handle exceptions from parallel tasks
    if isinstance(crawl_result, Exception):
        logger.error("Crawl failed", error=str(crawl_result))
        crawl_result = CrawlResultModel(url=params.url, status_code=500, error=str(crawl_result))
    
    if isinstance(performance_result, Exception):
        logger.error("Performance analysis failed", error=str(performance_result))
        performance_result = PerformanceResultModel(error=str(performance_result))
    
    # Build final result
    result = await build_seo_result(crawl_result, performance_result, params)
    
    # Log performance metrics
    duration = time.perf_counter() - start_time
    logger.info("SEO analysis completed", duration=duration, url=params.url)
    
    return result

# ❌ INCORRECT: Sequential execution
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    crawl_result = await crawl_url(params.url)  # Sequential
    performance_result = await analyze_performance(params.url)  # Sequential
    return build_result(crawl_result, performance_result)
```

## 4. Data Validation Conventions

### 4.1 Pydantic Model Conventions
```python
# ✅ CORRECT: Comprehensive Pydantic models
class SEOParamsModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',  # Strict validation
        frozen=True,     # Immutable models
        use_enum_values=True,
        populate_by_name=True
    )
    
    url: str = Field(
        ..., 
        description="URL to analyze",
        min_length=1,
        max_length=2048,
        pattern=r'^https?://.+'
    )
    keywords: List[str] = Field(
        default_factory=list,
        max_items=100,
        description="Keywords to check"
    )
    depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Crawl depth"
    )
    
    @validator('url')
    def validate_url(cls, v: str) -> str:
        """Custom URL validation."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @computed_field
    @property
    def is_valid(self) -> bool:
        """Check if parameters are valid."""
        return bool(self.url and self.url.startswith(('http://', 'https://')))

# ❌ INCORRECT: Basic models without validation
class SEOParams:
    def __init__(self, url: str):
        self.url = url  # No validation
```

### 4.2 Input Sanitization
```python
# ✅ CORRECT: Input sanitization and validation
def sanitize_url(url: str) -> str:
    """Sanitize URL for safe processing."""
    return urllib.parse.quote(url, safe=':/?=&')

def validate_and_sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize input data."""
    sanitized = {}
    
    if 'url' in data:
        url = str(data['url']).strip()
        if not is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        sanitized['url'] = sanitize_url(url)
    
    if 'keywords' in data:
        keywords = data['keywords']
        if isinstance(keywords, list):
            sanitized['keywords'] = [
                str(k).strip() for k in keywords 
                if k and len(str(k).strip()) <= 100
            ][:100]  # Limit to 100 keywords
    
    return sanitized

# ❌ INCORRECT: No input validation
def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    return data  # No validation or sanitization
```

## 5. Logging Conventions

### 5.1 Structured Logging
```python
# ✅ CORRECT: Structured logging with context
import structlog

logger = structlog.get_logger()

async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    """Analyze SEO with structured logging."""
    start_time = time.perf_counter()
    
    logger.info(
        "Starting SEO analysis",
        url=params.url,
        keywords_count=len(params.keywords),
        depth=params.depth
    )
    
    try:
        result = await perform_analysis(params)
        
        duration = time.perf_counter() - start_time
        logger.info(
            "SEO analysis completed successfully",
            url=params.url,
            score=result.score,
            duration=duration,
            status="success"
        )
        
        return result
    
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(
            "SEO analysis failed",
            url=params.url,
            error=str(e),
            duration=duration,
            status="error"
        )
        raise

# ❌ INCORRECT: Basic logging
import logging
logger = logging.getLogger(__name__)

def analyze_seo(url: str):
    logger.info(f"Analyzing {url}")  # No structured data
    # ...
    logger.info("Done")  # No context
```

### 5.2 Performance Logging
```python
# ✅ CORRECT: Performance metrics logging
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    """Analyze SEO with performance tracking."""
    start_time = time.perf_counter()
    
    # Track individual operation times
    crawl_start = time.perf_counter()
    crawl_result = await crawl_url(params)
    crawl_duration = time.perf_counter() - crawl_start
    
    analysis_start = time.perf_counter()
    analysis_result = await analyze_content(crawl_result)
    analysis_duration = time.perf_counter() - analysis_start
    
    total_duration = time.perf_counter() - start_time
    
    # Log detailed performance metrics
    logger.info(
        "SEO analysis performance",
        url=params.url,
        total_duration=total_duration,
        crawl_duration=crawl_duration,
        analysis_duration=analysis_duration,
        crawl_status_code=crawl_result.status_code,
        analysis_score=analysis_result.overall_score
    )
    
    return build_result(crawl_result, analysis_result)

# ❌ INCORRECT: No performance tracking
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    crawl_result = await crawl_url(params)
    analysis_result = await analyze_content(crawl_result)
    return build_result(crawl_result, analysis_result)
```

## 6. API Design Conventions

### 6.1 Endpoint Design
```python
# ✅ CORRECT: Consistent API endpoint design
@app.post("/analyze", response_model=SEOResponse)
async def analyze_seo_endpoint(
    request: SEORequest,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
) -> SEOResponse:
    """Analyze SEO for given URL with caching and rate limiting."""
    try:
        # Validate and process request
        params = SEOParamsModel(**request.model_dump(mode='json'))
        result = await analyze_seo(params)
        
        # Add background task for metrics
        background_tasks.add_task(log_metrics, params.url, result.score)
        
        return SEOResponse(**result.model_dump(mode='json'))
    
    except ValidationError as e:
        logger.warning("Validation error", errors=e.errors())
        raise HTTPException(status_code=400, detail="Invalid request data")
    
    except Exception as e:
        logger.error("Unexpected error in SEO analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# ❌ INCORRECT: Inconsistent endpoint design
@app.post("/seo")
async def seo_analysis(url: str, keywords: str = None):
    # No validation, no error handling, no rate limiting
    return {"url": url, "score": 50}
```

### 6.2 Response Models
```python
# ✅ CORRECT: Consistent response models
class SEOResponse(BaseModel):
    """Standard SEO analysis response."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True
    )
    
    url: str = Field(..., description="Analyzed URL")
    score: float = Field(..., ge=0, le=100, description="SEO score")
    title: Optional[str] = Field(None, max_length=512)
    description: Optional[str] = Field(None, max_length=1024)
    keywords: List[str] = Field(default_factory=list, max_items=100)
    errors: List[str] = Field(default_factory=list, max_items=50)
    warnings: List[str] = Field(default_factory=list, max_items=50)
    suggestions: List[str] = Field(default_factory=list, max_items=50)
    timestamp: float = Field(default_factory=time.time)
    
    @computed_field
    @property
    def is_optimized(self) -> bool:
        """Check if SEO score indicates optimization."""
        return self.score >= 80.0
    
    @computed_field
    @property
    def needs_improvement(self) -> bool:
        """Check if improvements are needed."""
        return self.score < 60.0

# ❌ INCORRECT: Inconsistent response structure
def analyze_seo(url: str) -> Dict[str, Any]:
    return {
        "url": url,
        "score": 75,
        "data": {...}  # Inconsistent structure
    }
```

## 7. Testing Conventions

### 7.1 Unit Test Structure
```python
# ✅ CORRECT: Comprehensive unit tests
import pytest
from unittest.mock import AsyncMock, patch

class TestSEOAnalysis:
    """Test suite for SEO analysis functionality."""
    
    @pytest.fixture
    def sample_params(self) -> SEOParamsModel:
        """Sample parameters for testing."""
        return SEOParamsModel(
            url="https://example.com",
            keywords=["seo", "optimization"],
            depth=2
        )
    
    @pytest.mark.asyncio
    async def test_analyze_seo_success(self, sample_params: SEOParamsModel):
        """Test successful SEO analysis."""
        with patch('seo_service.crawl_url') as mock_crawl, \
             patch('seo_service.analyze_content') as mock_analyze:
            
            mock_crawl.return_value = CrawlResultModel(
                url=sample_params.url,
                status_code=200,
                title="Test Page"
            )
            mock_analyze.return_value = AnalysisResultModel(
                title_score=85.0,
                description_score=90.0,
                headings_score=80.0,
                keywords_score=75.0,
                links_score=85.0,
                images_score=70.0
            )
            
            result = await analyze_seo(sample_params)
            
            assert result.url == sample_params.url
            assert result.score > 0
            assert result.score <= 100
            assert result.title == "Test Page"
    
    @pytest.mark.asyncio
    async def test_analyze_seo_invalid_url(self):
        """Test SEO analysis with invalid URL."""
        params = SEOParamsModel(url="invalid-url")
        
        with pytest.raises(URLValidationError):
            await analyze_seo(params)
    
    @pytest.mark.asyncio
    async def test_analyze_seo_network_error(self, sample_params: SEOParamsModel):
        """Test SEO analysis with network error."""
        with patch('seo_service.crawl_url', side_effect=Exception("Network error")):
            result = await analyze_seo(sample_params)
            
            assert result.url == sample_params.url
            assert result.score == 0.0
            assert len(result.errors) > 0

# ❌ INCORRECT: Basic tests without proper structure
def test_seo():
    result = analyze_seo("https://example.com")
    assert result is not None
```

### 7.2 Performance Tests
```python
# ✅ CORRECT: Performance testing
import pytest
import psutil
import gc
import asyncio

class TestPerformance:
    """Performance test suite."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_lazy_loading(self):
        """Test memory usage with lazy loading."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process large dataset with lazy loading
        config = LazyLoadingConfig(chunk_size=100)
        loader = LazyDataLoader(config)
        
        large_dataset = [{"id": i, "url": f"https://example{i}.com"} 
                        for i in range(100000)]
        
        async for chunk in loader.stream_data(large_dataset):
            # Process chunk
            pass
        
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should be minimal memory increase
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request handling."""
        urls = [f"https://example{i}.com" for i in range(100)]
        
        start_time = time.perf_counter()
        
        # Process URLs concurrently
        tasks = []
        for url in urls:
            params = SEOParamsModel(url=url)
            task = analyze_seo(params)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.perf_counter() - start_time
        
        # Should complete within reasonable time
        assert duration < 30.0  # Less than 30 seconds
        assert len(results) == 100

# ❌ INCORRECT: No performance testing
def test_basic_functionality():
    # Only tests functionality, not performance
    pass
```

## 8. Configuration Conventions

### 8.1 Environment Configuration
```python
# ✅ CORRECT: Structured configuration management
@dataclass
class Config:
    """Application configuration with validation."""
    debug: bool = Field(default=False, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8000, description="Port to bind")
    workers: int = Field(default=multiprocessing.cpu_count(), description="Number of workers")
    max_connections: int = Field(default=1000, description="Max connections")
    timeout: int = Field(default=30, description="Request timeout")
    rate_limit: int = Field(default=100, description="Rate limit per minute")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    mongo_url: str = Field(default="mongodb://localhost:27017", description="MongoDB URL")
    
    @validator('port')
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('workers')
    def validate_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError('Workers must be at least 1')
        return min(v, multiprocessing.cpu_count() * 2)

# Load configuration from environment
def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        debug=os.getenv('DEBUG', 'false').lower() == 'true',
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', '8000')),
        workers=int(os.getenv('WORKERS', str(multiprocessing.cpu_count()))),
        max_connections=int(os.getenv('MAX_CONNECTIONS', '1000')),
        timeout=int(os.getenv('TIMEOUT', '30')),
        rate_limit=int(os.getenv('RATE_LIMIT', '100')),
        cache_ttl=int(os.getenv('CACHE_TTL', '3600')),
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
        mongo_url=os.getenv('MONGO_URL', 'mongodb://localhost:27017')
    )

# ❌ INCORRECT: Hard-coded configuration
class Config:
    HOST = "0.0.0.0"
    PORT = 8000
    # No validation, no environment variable support
```

### 8.2 Feature Flags
```python
# ✅ CORRECT: Feature flag management
class FeatureFlags:
    """Feature flags for gradual rollout."""
    
    def __init__(self, config: Config):
        self.enable_lazy_loading = config.get('ENABLE_LAZY_LOADING', True)
        self.enable_streaming = config.get('ENABLE_STREAMING', True)
        self.enable_compression = config.get('ENABLE_COMPRESSION', True)
        self.enable_caching = config.get('ENABLE_CACHING', True)
    
    def is_lazy_loading_enabled(self) -> bool:
        """Check if lazy loading is enabled."""
        return self.enable_lazy_loading
    
    def is_streaming_enabled(self) -> bool:
        """Check if streaming is enabled."""
        return self.enable_streaming

# Usage in code
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    if feature_flags.is_lazy_loading_enabled():
        return await analyze_seo_lazy(params)
    else:
        return await analyze_seo_eager(params)

# ❌ INCORRECT: Hard-coded feature decisions
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    # Always use same approach, no flexibility
    return await analyze_seo_eager(params)
```

## 9. Security Conventions

### 9.1 Input Validation
```python
# ✅ CORRECT: Comprehensive input validation
def validate_url(url: str) -> bool:
    """Validate URL format and security."""
    try:
        parsed = urllib.parse.urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ('http', 'https'):
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'javascript:', 'data:', 'file:', 'ftp:',
            'localhost', '127.0.0.1', '0.0.0.0'
        ]
        
        for pattern in suspicious_patterns:
            if pattern in url.lower():
                return False
        
        # Check length
        if len(url) > 2048:
            return False
        
        return True
    
    except Exception:
        return False

def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize input data to prevent injection attacks."""
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            sanitized[key] = value.replace('<script>', '').replace('</script>', '')
        elif isinstance(value, list):
            sanitized[key] = [str(v).strip() for v in value if v]
        else:
            sanitized[key] = value
    
    return sanitized

# ❌ INCORRECT: No input validation
def process_url(url: str):
    # No validation, potential security issues
    return requests.get(url)
```

### 9.2 Rate Limiting
```python
# ✅ CORRECT: Comprehensive rate limiting
class RateLimiter:
    """Rate limiter with Redis and memory fallback."""
    
    async def check_rate_limit(
        self, 
        client_id: str, 
        max_requests: int = 100, 
        window: int = 60
    ) -> Dict[str, Any]:
        """Check rate limit for client."""
        try:
            # Try Redis first
            if self.redis_client:
                return await self._check_rate_limit_redis(
                    client_id, max_requests, window
                )
        except Exception as e:
            logger.warning("Redis rate limiting failed", error=str(e))
        
        # Fallback to memory
        return await self._check_rate_limit_memory(
            client_id, max_requests, window
        )
    
    async def _check_rate_limit_redis(
        self, 
        client_id: str, 
        max_requests: int, 
        window: int
    ) -> Dict[str, Any]:
        """Check rate limit using Redis."""
        key = f"rate_limit:{client_id}"
        current_time = time.time()
        
        # Use Redis pipeline for atomic operations
        async with self.redis_client.pipeline() as pipe:
            await pipe.zremrangebyscore(key, 0, current_time - window)
            await pipe.zadd(key, {str(current_time): current_time})
            await pipe.zcard(key)
            await pipe.expire(key, window)
            results = await pipe.execute()
        
        request_count = results[2]
        allowed = request_count <= max_requests
        
        return {
            'allowed': allowed,
            'remaining': max(0, max_requests - request_count),
            'reset_time': current_time + window
        }

# ❌ INCORRECT: No rate limiting
@app.post("/analyze")
async def analyze_seo(request: SEORequest):
    # No rate limiting, potential abuse
    return await perform_analysis(request)
```

## 10. Monitoring Conventions

### 10.1 Metrics Collection
```python
# ✅ CORRECT: Comprehensive metrics collection
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_COUNT = Counter(
    'seo_requests_total',
    'Total number of SEO requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'seo_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'seo_active_requests',
    'Number of active requests'
)

CACHE_HIT_RATIO = Gauge(
    'seo_cache_hit_ratio',
    'Cache hit ratio'
)

async def analyze_seo_endpoint(request: SEORequest) -> SEOResponse:
    """Analyze SEO with comprehensive metrics."""
    ACTIVE_REQUESTS.inc()
    start_time = time.perf_counter()
    
    try:
        result = await analyze_seo(request)
        
        # Record success metrics
        REQUEST_COUNT.labels(
            method='POST',
            endpoint='/analyze',
            status='200'
        ).inc()
        
        REQUEST_DURATION.labels(
            method='POST',
            endpoint='/analyze'
        ).observe(time.perf_counter() - start_time)
        
        return result
    
    except Exception as e:
        # Record error metrics
        REQUEST_COUNT.labels(
            method='POST',
            endpoint='/analyze',
            status='500'
        ).inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()

# ❌ INCORRECT: No metrics collection
async def analyze_seo_endpoint(request: SEORequest) -> SEOResponse:
    # No monitoring, no observability
    return await analyze_seo(request)
```

### 10.2 Health Checks
```python
# ✅ CORRECT: Comprehensive health checks
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "15.0.0",
        "checks": {}
    }
    
    # Check Redis
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check MongoDB
    try:
        mongo_client = await get_mongo()
        await mongo_client.admin.command('ping')
        health_status["checks"]["mongodb"] = "healthy"
    except Exception as e:
        health_status["checks"]["mongodb"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check memory usage
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    health_status["checks"]["memory_usage_mb"] = memory_usage
    
    # Check CPU usage
    cpu_percent = process.cpu_percent()
    health_status["checks"]["cpu_percent"] = cpu_percent
    
    # Determine overall status
    if health_status["status"] == "healthy" and memory_usage > 1000:
        health_status["status"] = "degraded"
    
    return health_status

# ❌ INCORRECT: Basic health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

## Conclusion

These key conventions ensure:

1. **Code Consistency**: Uniform patterns across the codebase
2. **Performance Optimization**: Best practices for high-performance applications
3. **Maintainability**: Clear, readable, and well-structured code
4. **Reliability**: Comprehensive error handling and validation
5. **Security**: Input validation and rate limiting
6. **Observability**: Comprehensive monitoring and logging
7. **Scalability**: Efficient resource management and caching
8. **Testability**: Well-structured tests and mocking

Following these conventions ensures the Ultra-Optimized SEO Service v15 maintains high quality, performance, and reliability in production environments. 