# FastAPI Implementation Guide

## Overview

This guide provides comprehensive information about the enhanced FastAPI implementation for the HeyGen AI backend. The application includes advanced features, production-ready configurations, and enterprise-grade capabilities.

## 🏗️ Architecture Overview

### Core Components

```
HeyGen AI FastAPI Application
├── Main Application (main.py)
├── API Router System
├── Middleware Stack
├── Error Handling System
├── WebSocket Manager
├── Background Task Manager
├── Monitoring & Metrics
└── Security Layer
```

### Key Features

- **Advanced Middleware**: Security, rate limiting, caching, monitoring
- **Real-time Communication**: WebSocket support with connection management
- **Background Tasks**: Asynchronous task processing with monitoring
- **Comprehensive Error Handling**: Structured error responses and logging
- **Production Monitoring**: Prometheus metrics, structured logging, health checks
- **Security**: JWT authentication, rate limiting, input validation
- **Performance**: Gzip compression, connection pooling, caching

## 🚀 Getting Started

### Basic Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python -m heygen_ai.main

# Or using uvicorn directly
uvicorn heygen_ai.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Configuration

```bash
# .env file
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO
ENFORCE_HTTPS=true
ALLOWED_ORIGINS=["http://localhost:3000", "https://heygen.ai"]
ALLOWED_HOSTS=["localhost", "heygen.ai", "*.heygen.ai"]
SESSION_SECRET_KEY=your-secret-key
SESSION_MAX_AGE=3600
```

## 📊 API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/` | GET | API information | None |
| `/health` | GET | Health check | None |
| `/metrics` | GET | Prometheus metrics | None |
| `/ready` | GET | Readiness probe | None |
| `/live` | GET | Liveness probe | None |
| `/info` | GET | System information | None |

### API v2 Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/api/v2/videos/generate` | POST | Generate video | Required |
| `/api/v2/videos/status` | POST | Get video status | Required |
| `/api/v2/videos/list` | POST | List user videos | Required |
| `/api/v2/videos/{id}/download` | GET | Download video | Required |
| `/api/v2/videos/{id}` | DELETE | Delete video | Required |

## 🔧 Middleware Stack

### Security Middleware

```python
class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and protection"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https:; "
            "media-src 'self' https:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response
```

### Rate Limiting Middleware

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app, rate_limit_per_minute: int = 100):
        super().__init__(app)
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_counts = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        self.request_counts = {
            ip: count for ip, (count, timestamp) in self.request_counts.items()
            if current_time - timestamp < 60
        }
        
        # Check rate limit
        if client_ip in self.request_counts:
            count, _ = self.request_counts[client_ip]
            if count >= self.rate_limit_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            self.request_counts[client_ip] = (count + 1, current_time)
        else:
            self.request_counts[client_ip] = (1, current_time)
        
        return await call_next(request)
```

### Cache Middleware

```python
class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching"""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache = {}
    
    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        cache_key = f"{request.method}:{request.url.path}:{request.query_params}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_response, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                CACHE_HITS.inc(cache_type="response")
                return cached_response
            else:
                del self.cache[cache_key]
        
        CACHE_MISSES.inc(cache_type="response")
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            self.cache[cache_key] = (response, current_time)
        
        return response
```

## 📡 WebSocket Management

### WebSocket Manager

```python
class WebSocketManager:
    """Main WebSocket manager for handling connections and message broadcasting."""
    
    def __init__(
        self,
        heartbeat_interval: int = 30,
        cleanup_interval: int = 60,
        max_connections: int = 1000,
        max_subscriptions_per_connection: int = 50
    ):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.topic_subscriptions: Dict[str, Set[str]] = {}
        self.heartbeat_interval = heartbeat_interval
        self.cleanup_interval = cleanup_interval
        self.max_connections = max_connections
        self.max_subscriptions_per_connection = max_subscriptions_per_connection
```

### WebSocket Usage

```python
# Connect to WebSocket
websocket = await websocket_manager.connect(websocket, user_id)

# Subscribe to video updates
await websocket_manager.subscribe_to_video(connection_id, video_id)

# Broadcast video progress
await websocket_manager.broadcast_video_progress(video_id, 0.5, "processing")

# Broadcast video completion
await websocket_manager.broadcast_video_complete(
    video_id, 
    "https://example.com/video.mp4", 
    120.5, 
    1024000
)
```

## 🔄 Background Task Management

### Background Task Manager

```python
class BackgroundTaskManager:
    """Background task manager with monitoring and error handling."""
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def add_task(
        self, 
        task_id: str, 
        coro: Awaitable, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> asyncio.Task:
        """Add a background task."""
        task = asyncio.create_task(coro)
        self.tasks[task_id] = task
        
        self.task_metadata[task_id] = {
            "created_at": datetime.now(),
            "status": "running",
            "metadata": metadata or {}
        }
        
        # Add done callback
        task.add_done_callback(lambda t: self._task_done_callback(task_id, t))
        
        return task
```

### Background Task Usage

```python
# Add background task
task_manager = get_background_task_manager()
task = await task_manager.add_task(
    "video_processing_123",
    process_video(video_id, script, avatar_id),
    metadata={"video_id": video_id, "user_id": user_id}
)

# Check task status
task_info = task_manager.get_task_info("video_processing_123")
print(f"Task status: {task_info['status']}")

# Cancel task
await task_manager.cancel_task("video_processing_123")
```

## 📊 Monitoring & Metrics

### Prometheus Metrics

```python
# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_REQUESTS = Counter('http_requests_active', 'Active HTTP requests')

# WebSocket metrics
WEBSOCKET_CONNECTIONS = Counter('websocket_connections_total', 'Total WebSocket connections')

# Background task metrics
BACKGROUND_TASKS = Counter('background_tasks_total', 'Total background tasks', ['task_type', 'status'])

# Cache metrics
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
```

### Metrics Endpoint

```python
@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

## 🔐 Security Features

### Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Get current user from JWT token."""
    token = credentials.credentials
    
    try:
        # Decode JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        
        return user_id
        
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
```

### Input Validation

```python
class RequestValidator:
    """Request validation and sanitization utilities."""
    
    @staticmethod
    def validate_content_length(request: Request, max_size: int = 10 * 1024 * 1024) -> bool:
        """Validate request content length."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                return size <= max_size
            except ValueError:
                return False
        return True
    
    @staticmethod
    def validate_content_type(request: Request, allowed_types: List[str]) -> bool:
        """Validate request content type."""
        content_type = request.headers.get("content-type", "")
        return any(allowed_type in content_type for allowed_type in allowed_types)
```

## 🚀 Performance Optimization

### Redis Caching

```python
class RedisCache:
    """Redis-backed caching with TTL and compression."""
    
    def __init__(self, redis_client: redis.Redis, prefix: str = "cache"):
        self.redis = redis_client
        self.prefix = prefix
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            data = await self.redis.get(self._make_key(key))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL."""
        try:
            data = json.dumps(value)
            return await self.redis.setex(self._make_key(key), ttl, data)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
```

### Rate Limiting with Redis

```python
class RedisRateLimiter:
    """Redis-backed rate limiter with sliding window."""
    
    def __init__(self, redis_client: redis.Redis, prefix: str = "rate_limit"):
        self.redis = redis_client
        self.prefix = prefix
    
    async def is_allowed(
        self, 
        key: str, 
        max_requests: int, 
        window_seconds: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed within rate limit."""
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Create Redis key
        redis_key = f"{self.prefix}:{key}"
        
        # Remove old entries and add current request
        await self.redis.zremrangebyscore(redis_key, 0, window_start)
        await self.redis.zadd(redis_key, {str(current_time): current_time})
        await self.redis.expire(redis_key, window_seconds)
        
        # Count requests in window
        request_count = await self.redis.zcard(redis_key)
        
        # Get remaining requests
        remaining = max(0, max_requests - request_count)
        
        # Check if allowed
        allowed = request_count < max_requests
        
        return allowed, {
            "limit": max_requests,
            "remaining": remaining,
            "reset_time": current_time + window_seconds,
            "window_seconds": window_seconds
        }
```

## 🛠️ Error Handling

### Custom Exception Handlers

```python
@app.exception_handler(HeyGenBaseError)
async def heygen_exception_handler(request: Request, exc: HeyGenBaseError):
    """Handle HeyGen-specific exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )
```

## 📝 Logging

### Structured Logging

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info("Request processed", 
           request_id=request_id,
           user_id=user_id,
           duration=duration)
```

## 🔧 Configuration

### Settings Management

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Database
    database_url: str
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # CORS
    allowed_origins: List[str] = ["http://localhost:3000"]
    allowed_hosts: List[str] = ["localhost"]
    
    # Rate Limiting
    rate_limit_per_minute: int = 100
    
    # Caching
    cache_ttl: int = 300
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## 🚀 Deployment

### Production Configuration

```python
# main.py
def main():
    """Main application entry point"""
    settings = get_settings()
    
    # Create application
    app = create_app()
    
    # Configure uvicorn settings
    uvicorn_config = {
        "app": app,
        "host": settings.api_host,
        "port": settings.api_port,
        "workers": settings.api_workers,
        "log_level": settings.log_level.lower(),
        "access_log": True,
        "use_colors": not settings.is_production(),
        "reload": settings.debug,
        "reload_dirs": ["."] if settings.debug else None,
        "loop": "asyncio",
        "http": "httptools",
        "ws": "websockets",
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
        "limit_concurrency": 1000,
        "limit_max_requests": 10000,
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": 30
    }
    
    # Start server
    uvicorn.run(**uvicorn_config)
```

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "heygen_ai.main"]
```

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: heygen-ai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: heygen-ai-api
  template:
    metadata:
      labels:
        app: heygen-ai-api
    spec:
      containers:
      - name: api
        image: heygen-ai-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: heygen-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 📚 Best Practices

### 1. Error Handling

- Always use structured error responses
- Log errors with context
- Provide user-friendly error messages
- Use appropriate HTTP status codes

### 2. Performance

- Use async/await for I/O operations
- Implement caching for expensive operations
- Use connection pooling for databases
- Monitor performance metrics

### 3. Security

- Validate all inputs
- Use HTTPS in production
- Implement rate limiting
- Set appropriate security headers
- Use JWT for authentication

### 4. Monitoring

- Use structured logging
- Implement health checks
- Monitor application metrics
- Set up alerting

### 5. Testing

- Write unit tests for business logic
- Write integration tests for API endpoints
- Use test fixtures and mocks
- Test error scenarios

## 🔍 Troubleshooting

### Common Issues

1. **Connection Timeouts**
   - Check Redis connection
   - Verify database connectivity
   - Monitor system resources

2. **Rate Limiting Issues**
   - Check Redis availability
   - Verify rate limit configuration
   - Monitor request patterns

3. **Memory Issues**
   - Check for memory leaks
   - Monitor background tasks
   - Review caching strategy

4. **Performance Issues**
   - Check database queries
   - Monitor cache hit rates
   - Review async operations

### Debug Mode

```python
# Enable debug mode
DEBUG=true
LOG_LEVEL=DEBUG

# This will enable:
# - Detailed error messages
# - Request/response logging
# - Development endpoints
# - Hot reload
```

## 📖 API Documentation

The API documentation is automatically generated and available at:

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI JSON**: `/openapi.json`

### Example API Usage

```python
import requests

# Generate video
response = requests.post(
    "https://api.heygen.ai/api/v2/videos/generate",
    headers={"Authorization": "Bearer your-token"},
    json={
        "script": "Hello, this is a test video.",
        "avatar_id": "avatar_123",
        "voice_id": "voice_456",
        "language": "en",
        "style": "professional"
    }
)

video_id = response.json()["video_id"]

# Check video status
status_response = requests.post(
    "https://api.heygen.ai/api/v2/videos/status",
    headers={"Authorization": "Bearer your-token"},
    json={"video_id": video_id}
)

print(f"Video status: {status_response.json()['status']}")
```

## 🤝 Contributing

1. Follow the code style guidelines
2. Write tests for new features
3. Update documentation
4. Use conventional commit messages
5. Create pull requests for changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 