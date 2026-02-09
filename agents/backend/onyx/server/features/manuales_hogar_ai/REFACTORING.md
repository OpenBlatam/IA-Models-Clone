# Refactoring Guide - Manuales Hogar AI

## 🎯 Refactoring Overview

This document describes the refactoring done to align with microservices, serverless, and cloud-native best practices.

## ✨ Key Improvements

### 1. Advanced Middleware Stack

#### Logging Middleware
- ✅ Structured logging with request IDs
- ✅ Request/response tracking
- ✅ Duration measurement
- ✅ Error logging with stack traces

#### Security Middleware
- ✅ Security headers (X-Frame-Options, CSP, HSTS, etc.)
- ✅ CORS configuration
- ✅ Content validation
- ✅ Referrer policy

#### Rate Limiting Middleware
- ✅ Redis-based rate limiting
- ✅ In-memory fallback
- ✅ Per-minute and per-hour limits
- ✅ Rate limit headers in responses

#### Tracing Middleware
- ✅ OpenTelemetry integration
- ✅ Distributed tracing
- ✅ OTLP exporter support
- ✅ Request span tracking

#### Metrics Middleware
- ✅ Prometheus metrics
- ✅ Request counters
- ✅ Duration histograms
- ✅ In-progress gauges

### 2. Resilience Patterns

#### Circuit Breaker
- ✅ Circuit breaker pattern implementation
- ✅ Configurable failure threshold
- ✅ Recovery timeout
- ✅ State management (CLOSED, OPEN, HALF_OPEN)

#### Retry Pattern
- ✅ Exponential backoff
- ✅ Configurable retry attempts
- ✅ Exception handling
- ✅ Async and sync support

### 3. Caching

#### Redis Cache
- ✅ High-performance Redis cache
- ✅ Async operations
- ✅ JSON serialization
- ✅ TTL support
- ✅ Pattern-based clearing

### 4. Configuration

#### Enhanced Settings
- ✅ Environment-based configuration
- ✅ Redis configuration
- ✅ Security settings
- ✅ Rate limiting configuration
- ✅ Monitoring configuration
- ✅ Circuit breaker settings
- ✅ Retry configuration
- ✅ Serverless optimization flags

### 5. API Enhancements

#### Metrics Endpoint
- ✅ `/metrics` endpoint for Prometheus
- ✅ Standard Prometheus format
- ✅ Content type headers

#### Health Checks
- ✅ Enhanced health checks
- ✅ Dependency checking
- ✅ Status reporting

## 📁 New File Structure

```
manuales_hogar_ai/
├── middleware/
│   ├── __init__.py
│   ├── logging_middleware.py      # Structured logging
│   ├── security_middleware.py     # Security headers
│   ├── rate_limit_middleware.py   # Rate limiting
│   ├── tracing_middleware.py      # OpenTelemetry tracing
│   └── metrics_middleware.py      # Prometheus metrics
├── infrastructure/
│   ├── circuit_breaker.py         # Circuit breaker pattern
│   ├── retry.py                   # Retry with backoff
│   └── cache_redis.py             # Redis cache
├── api/
│   └── routes/
│       └── metrics.py             # Prometheus endpoint
└── config/
    └── settings.py                # Enhanced configuration
```

## 🔧 Configuration

### Environment Variables

```bash
# Redis
REDIS_URL=redis://localhost:6379/0

# Environment
ENVIRONMENT=prod
DEBUG=false

# Security
ALLOWED_ORIGINS=https://example.com,https://app.example.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Monitoring
ENABLE_PROMETHEUS=true
ENABLE_TRACING=true
OTLP_ENDPOINT=http://otel-collector:4317

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60.0

# Retry
RETRY_MAX_ATTEMPTS=3
RETRY_DELAY=1.0
RETRY_BACKOFF=2.0

# Serverless
COLD_START_OPTIMIZATION=true
```

## 📊 Monitoring

### Prometheus Metrics

Available metrics:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration histogram
- `http_requests_in_progress` - Current in-progress requests

Access metrics at: `http://localhost:8000/metrics`

### Distributed Tracing

Tracing is enabled via OpenTelemetry:
- Spans for each request
- Attributes for method, path, status code
- Error tracking
- Duration measurement

## 🚀 Usage Examples

### Using Circuit Breaker

```python
from manuales_hogar_ai.infrastructure.circuit_breaker import circuit_breaker

@circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
async def call_external_service():
    # Your code here
    pass
```

### Using Retry

```python
from manuales_hogar_ai.infrastructure.retry import retry

@retry(max_attempts=3, delay=1.0, backoff=2.0)
async def unreliable_operation():
    # Your code here
    pass
```

### Using Redis Cache

```python
from manuales_hogar_ai.infrastructure.cache_redis import get_cache

cache = await get_cache()
await cache.set("key", {"data": "value"}, ttl=3600)
value = await cache.get("key")
```

## 🔒 Security Improvements

1. **Security Headers**: All responses include security headers
2. **CORS**: Configurable CORS with origin validation
3. **Rate Limiting**: Protection against abuse
4. **Content Validation**: CSP and content type validation

## 📈 Performance Improvements

1. **Redis Caching**: Fast cache layer for frequently accessed data
2. **Async Operations**: All I/O operations are async
3. **Connection Pooling**: Efficient database and Redis connections
4. **Cold Start Optimization**: Flags for serverless optimization

## 🧪 Testing

Test the new features:

```bash
# Test metrics endpoint
curl http://localhost:8000/metrics

# Test rate limiting
for i in {1..100}; do curl http://localhost:8000/api/v1/health; done

# Test security headers
curl -I http://localhost:8000/api/v1/health
```

## 📚 Dependencies

New dependencies added:
- `prometheus-client` - Prometheus metrics
- `opentelemetry-*` - Distributed tracing
- `redis` - Redis client
- `tenacity` - Retry utilities

## 🎯 Next Steps

1. **Message Broker**: Add RabbitMQ/Kafka for event-driven architecture
2. **API Gateway**: Integrate with Kong or AWS API Gateway
3. **Service Mesh**: Consider Istio/Linkerd for advanced routing
4. **Advanced Caching**: Implement cache warming strategies
5. **Load Testing**: Validate performance under load

## 📝 Migration Notes

### Breaking Changes
- None - All changes are backward compatible

### Configuration Changes
- New environment variables required for full functionality
- Redis recommended for production (optional for development)

### Deployment
- No changes to deployment process
- New dependencies in requirements.txt
- Docker images will include new dependencies

---

**Last Updated**: 2024-01-XX
**Version**: 2.0.0




