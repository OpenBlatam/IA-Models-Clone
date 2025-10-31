# Architectural Principles

## 1. Clean Architecture Principles

### Dependency Rule
```python
# Domain layer (innermost) - no dependencies on external frameworks
class ContentRequest:
    def __init__(self, text: str, priority: Priority):
        self.text = text
        self.priority = priority

# Application layer - orchestrates use cases
class GenerateContentUseCase:
    def __init__(self, content_repo: ContentRepository):
        self.content_repo = content_repo
    
    async def execute(self, request: ContentRequest) -> ContentResult:
        return await self.content_repo.generate(request)

# Infrastructure layer (outermost) - implements interfaces
class FastAPIContentRepository(ContentRepository):
    async def generate(self, request: ContentRequest) -> ContentResult:
        # Implementation details
        pass
```

### SOLID Principles
```python
# Single Responsibility
class VideoProcessor:
    def process_video(self, video_data: bytes) -> VideoResult:
        # Only handles video processing
        pass

class AudioProcessor:
    def process_audio(self, audio_data: bytes) -> AudioResult:
        # Only handles audio processing
        pass

# Open/Closed Principle
class BaseProcessor:
    def process(self, data: Any) -> Any:
        raise NotImplementedError

class VideoProcessor(BaseProcessor):
    def process(self, data: VideoData) -> VideoResult:
        # Implementation
        pass

# Liskov Substitution
def process_content(processor: BaseProcessor, data: Any) -> Any:
    return processor.process(data)  # Works with any BaseProcessor subclass

# Interface Segregation
class ContentGenerator:
    def generate_text(self) -> str:
        pass

class MediaProcessor:
    def process_media(self) -> bytes:
        pass

# Dependency Inversion
class ContentService:
    def __init__(self, generator: ContentGenerator, processor: MediaProcessor):
        self.generator = generator
        self.processor = processor
```

## 2. Performance Optimization Principles

### Asynchronous Processing
```python
async def process_batch(requests: List[ContentRequest]) -> List[ContentResult]:
    tasks = [process_single(request) for request in requests]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def process_single(request: ContentRequest) -> ContentResult:
    async with aiohttp.ClientSession() as session:
        async with session.post('/api/generate', json=request.dict()) as resp:
            return await resp.json()
```

### Multi-Level Caching
```python
class CacheManager:
    def __init__(self):
        self.l1_cache = {}  # Memory cache
        self.l2_cache = redis.Redis()  # Redis cache
        self.l3_cache = DiskCache()  # Disk cache
    
    async def get(self, key: str) -> Optional[Any]:
        # L1 check
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # L2 check
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value
        
        # L3 check
        value = await self.l3_cache.get(key)
        if value:
            await self.l2_cache.set(key, value)
            self.l1_cache[key] = value
            return value
        
        return None
```

### Resource Management
```python
class ResourceManager:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
        self.memory_pool = MemoryPool(max_size=1024*1024*100)  # 100MB pool
    
    async def process_with_limits(self, data: bytes) -> bytes:
        async with self.semaphore:
            async with self.memory_pool.allocate(len(data)) as memory:
                return await self.process_data(data, memory)
```

## 3. Scalability Principles

### Horizontal Scaling
```python
class LoadBalancer:
    def __init__(self, instances: List[str]):
        self.instances = instances
        self.current_index = 0
    
    def get_next_instance(self) -> str:
        instance = self.instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instances)
        return instance

class AutoScaler:
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
    
    async def scale_based_on_load(self, current_load: float):
        if current_load > 0.8 and self.current_instances < self.max_instances:
            await self.scale_up()
        elif current_load < 0.3 and self.current_instances > self.min_instances:
            await self.scale_down()
```

### Event-Driven Architecture
```python
class EventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
    
    def subscribe(self, event_type: str, handler: Callable):
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, data: Any):
        for handler in self.subscribers[event_type]:
            await handler(data)

class ContentEvent:
    def __init__(self, content_id: str, event_type: str, data: Any):
        self.content_id = content_id
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.utcnow()
```

## 4. Security Principles

### Input Validation
```python
from pydantic import BaseModel, validator
import re

class ContentRequest(BaseModel):
    text: str
    max_length: int = 1000
    
    @validator('text')
    def validate_text(cls, v):
        if len(v) > cls.max_length:
            raise ValueError(f'Text too long: {len(v)} > {cls.max_length}')
        if re.search(r'<script|javascript:', v, re.IGNORECASE):
            raise ValueError('Potentially malicious content detected')
        return v.strip()

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 3600):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    async def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Remove old requests
        user_requests[:] = [req for req in user_requests if now - req < self.window]
        
        if len(user_requests) >= self.max_requests:
            return False
        
        user_requests.append(now)
        return True
```

### Authentication & Authorization
```python
class JWTManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def create_token(self, user_id: str, permissions: List[str]) -> str:
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return None

class PermissionChecker:
    def __init__(self):
        self.permission_map = {
            'admin': ['read', 'write', 'delete'],
            'user': ['read', 'write'],
            'guest': ['read']
        }
    
    def has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        return required_permission in user_permissions
```

## 5. Monitoring & Observability Principles

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

class RequestLogger:
    def __init__(self):
        self.logger = structlog.get_logger()
    
    async def log_request(self, request_id: str, method: str, path: str, 
                         duration: float, status_code: int):
        self.logger.info(
            "request_processed",
            request_id=request_id,
            method=method,
            path=path,
            duration_ms=duration * 1000,
            status_code=status_code
        )

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(int)
        self.timers = defaultdict(list)
    
    def increment(self, metric_name: str, value: int = 1):
        self.metrics[metric_name] += value
    
    def record_timing(self, metric_name: str, duration: float):
        self.timers[metric_name].append(duration)
    
    def get_metrics(self) -> Dict:
        return {
            'counters': dict(self.metrics),
            'timers': {k: {'avg': sum(v)/len(v), 'count': len(v)} 
                      for k, v in self.timers.items() if v}
        }
```

### Health Checks
```python
class HealthChecker:
    def __init__(self):
        self.checks = []
    
    def add_check(self, name: str, check_func: Callable) -> None:
        self.checks.append((name, check_func))
    
    async def run_checks(self) -> Dict[str, bool]:
        results = {}
        for name, check_func in self.checks:
            try:
                results[name] = await check_func()
            except Exception as e:
                logger.error(f"Health check failed: {name}", error=str(e))
                results[name] = False
        return results

class DatabaseHealthCheck:
    def __init__(self, db_session):
        self.db_session = db_session
    
    async def check(self) -> bool:
        try:
            await self.db_session.execute("SELECT 1")
            return True
        except Exception:
            return False
```

## 6. Error Handling Principles

### Graceful Degradation
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            
            raise e

class RetryHandler:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
        
        raise last_exception
```

## 7. Configuration Management Principles

### Environment-Based Configuration
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://localhost/os_content"
    redis_url: str = "redis://localhost:6379"
    
    # Security
    secret_key: str
    jwt_expiration_hours: int = 24
    
    # Performance
    max_workers: int = 10
    cache_ttl: int = 3600
    
    # Monitoring
    prometheus_port: int = 9090
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class ConfigurationManager:
    def __init__(self):
        self.settings = Settings()
        self._validate_config()
    
    def _validate_config(self):
        if not self.settings.secret_key:
            raise ValueError("SECRET_KEY must be set")
    
    def get_database_config(self) -> Dict:
        return {
            'url': self.settings.database_url,
            'pool_size': self.settings.max_workers,
            'max_overflow': self.settings.max_workers * 2
        }
```

## 8. Testing Principles

### Test-Driven Development
```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestContentService:
    @pytest.fixture
    def mock_repository(self):
        return Mock(spec=ContentRepository)
    
    @pytest.fixture
    def content_service(self, mock_repository):
        return ContentService(mock_repository)
    
    @pytest.mark.asyncio
    async def test_generate_content_success(self, content_service, mock_repository):
        # Arrange
        request = ContentRequest(text="Test content", priority=Priority.HIGH)
        expected_result = ContentResult(id="123", content="Generated content")
        mock_repository.generate.return_value = expected_result
        
        # Act
        result = await content_service.generate_content(request)
        
        # Assert
        assert result == expected_result
        mock_repository.generate.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_generate_content_failure(self, content_service, mock_repository):
        # Arrange
        request = ContentRequest(text="Test content", priority=Priority.HIGH)
        mock_repository.generate.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(ContentGenerationError):
            await content_service.generate_content(request)
```

### Integration Testing
```python
class TestAPIIntegration:
    @pytest.fixture
    async def client(self):
        from fastapi.testclient import TestClient
        from production_app import app
        async with TestClient(app) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_generate_video_endpoint(self, client):
        response = await client.post("/api/v1/video/generate", json={
            "text": "Test video content",
            "duration": 10,
            "quality": "high"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "video_url" in data
        assert "processing_time" in data
```

## 9. Deployment Principles

### Immutable Infrastructure
```dockerfile
# Multi-stage build for optimized production image
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim as production
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
RUN useradd --create-home app && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["python", "production_app.py"]
```

### Blue-Green Deployment
```python
class DeploymentManager:
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.blue_instances = []
        self.green_instances = []
        self.current_color = 'blue'
    
    async def deploy_new_version(self, new_instances: List[str]):
        # Deploy to inactive color
        inactive_color = 'green' if self.current_color == 'blue' else 'blue'
        inactive_instances = getattr(self, f'{inactive_color}_instances')
        
        # Deploy and health check
        for instance in new_instances:
            if await self.health_check(instance):
                inactive_instances.append(instance)
        
        # Switch traffic
        if inactive_instances:
            self.load_balancer.update_instances(inactive_instances)
            self.current_color = inactive_color
            
            # Cleanup old instances
            old_instances = getattr(self, f'{self.get_opposite_color()}_instances')
            await self.cleanup_instances(old_instances)
            setattr(self, f'{self.get_opposite_color()}_instances', [])
```

## 10. Data Management Principles

### Repository Pattern
```python
from abc import ABC, abstractmethod

class ContentRepository(ABC):
    @abstractmethod
    async def save(self, content: Content) -> str:
        pass
    
    @abstractmethod
    async def get_by_id(self, content_id: str) -> Optional[Content]:
        pass
    
    @abstractmethod
    async def update(self, content_id: str, content: Content) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, content_id: str) -> bool:
        pass

class PostgreSQLContentRepository(ContentRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory
    
    async def save(self, content: Content) -> str:
        async with self.session_factory() as session:
            session.add(content)
            await session.commit()
            return content.id
    
    async def get_by_id(self, content_id: str) -> Optional[Content]:
        async with self.session_factory() as session:
            return await session.get(Content, content_id)
```

### Data Validation
```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List
from datetime import datetime

class ContentModel(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1, max_length=10000)
    type: str = Field(..., regex='^(text|video|audio|image)$')
    priority: str = Field(..., regex='^(low|medium|high|urgent)$')
    metadata: Optional[Dict] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('text')
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    @validator('metadata')
    def validate_metadata(cls, v):
        if v and len(str(v)) > 1000:
            raise ValueError('Metadata too large')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

These principles ensure the system is maintainable, scalable, secure, and performant while following industry best practices for modern software development. 