# Middleware Guide - Addiction Recovery AI

## ✅ Recommended Middleware

### Core Middleware Files

#### `middleware/error_handler.py` - **USE THIS** ✅
- **Status**: Canonical error handling middleware
- **Purpose**: Centralized error handling
- **Features**: Exception handling, error responses, logging

#### `middleware/rate_limit.py` - **USE THIS** ✅
- **Status**: Canonical rate limiting middleware
- **Purpose**: API rate limiting
- **Features**: Request throttling, rate limit headers

#### `middleware/logging_middleware.py` - **USE THIS** ✅
- **Status**: Canonical logging middleware
- **Purpose**: Request/response logging
- **Features**: Structured logging, request tracking

## 📋 Performance Middleware Options

### `middleware/performance.py` - Basic Performance
- **Status**: ✅ Active (Basic)
- **Purpose**: Basic performance monitoring
- **Features**: Request timing, performance metrics logging

```python
from middleware.performance import PerformanceMonitoringMiddleware

app.add_middleware(PerformanceMonitoringMiddleware)
```

### `middleware/performance_middleware.py` - Advanced Performance
- **Status**: ✅ Active (Advanced)
- **Purpose**: Advanced performance optimization
- **Features**: 
  - Response compression
  - Connection pooling
  - Request batching
  - Response caching headers
  - Gzip compression

```python
from middleware.performance_middleware import PerformanceMiddleware

app.add_middleware(PerformanceMiddleware)
```

**When to use:**
- High-throughput APIs
- Low-latency requirements
- Production environments
- When you need advanced optimizations

### `middleware/performance_integrator.py` - Performance Integration
- **Status**: ✅ Active (Integration)
- **Purpose**: Integrates multiple performance features
- **Use Case**: When you need comprehensive performance integration

### `middleware/speed_middleware.py` - Speed Optimization
- **Status**: ✅ Active (Specialized)
- **Purpose**: Speed-specific optimizations
- **Use Case**: Speed-critical applications

### `middleware/ultra_speed_middleware.py` - Ultra Speed
- **Status**: ✅ Active (Specialized)
- **Purpose**: Ultra-speed optimizations
- **Use Case**: Maximum speed requirements

## 🔒 Security Middleware

### `middleware/security_advanced.py` - Advanced Security
- **Status**: ✅ Active
- **Purpose**: Advanced security features
- **Features**: Enhanced security headers, threat detection

### `middleware/oauth2_middleware.py` - OAuth2
- **Status**: ✅ Active
- **Purpose**: OAuth2 authentication middleware
- **Use Case**: OAuth2 authentication flows

## 📊 Observability Middleware

### `middleware/aws_observability.py` - AWS Observability
- **Status**: ✅ Active (AWS-specific)
- **Purpose**: AWS CloudWatch integration
- **Use Case**: AWS deployments

### `middleware/opentelemetry_middleware.py` - OpenTelemetry
- **Status**: ✅ Active
- **Purpose**: OpenTelemetry tracing
- **Use Case**: Distributed tracing

## 🚦 Rate Limiting & Throttling

### `middleware/rate_limit.py` - Rate Limiting
- **Status**: ✅ Canonical
- **Purpose**: API rate limiting

### `middleware/throttling_middleware.py` - Throttling
- **Status**: ✅ Active
- **Purpose**: Request throttling
- **Note**: Different from rate limiting, use based on needs

## 🏗️ Middleware Structure

```
middleware/
├── error_handler.py              # ✅ Canonical error handling
├── rate_limit.py                 # ✅ Canonical rate limiting
├── logging_middleware.py          # ✅ Canonical logging
├── performance.py                 # ✅ Basic performance
├── performance_middleware.py      # ✅ Advanced performance
├── performance_integrator.py      # ✅ Performance integration
├── speed_middleware.py            # ✅ Speed optimization
├── ultra_speed_middleware.py      # ✅ Ultra speed
├── security_advanced.py           # ✅ Advanced security
├── oauth2_middleware.py           # ✅ OAuth2
├── aws_observability.py           # ✅ AWS observability
└── opentelemetry_middleware.py    # ✅ OpenTelemetry
```

## 📝 Usage Examples

### Basic Setup
```python
from fastapi import FastAPI
from middleware.error_handler import ErrorHandlerMiddleware
from middleware.rate_limit import RateLimitMiddleware
from middleware.logging_middleware import LoggingMiddleware

app = FastAPI()

app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)
```

### Advanced Performance Setup
```python
from middleware.performance_middleware import PerformanceMiddleware

app.add_middleware(PerformanceMiddleware)
```

### AWS Deployment
```python
from middleware.aws_observability import AWSObservabilityMiddleware

app.add_middleware(AWSObservabilityMiddleware)
```

## 🎯 Quick Reference

| Middleware | Purpose | Status | When to Use |
|------------|---------|--------|-------------|
| `error_handler.py` | Error handling | ✅ Canonical | Always |
| `rate_limit.py` | Rate limiting | ✅ Canonical | Always |
| `logging_middleware.py` | Logging | ✅ Canonical | Always |
| `performance.py` | Basic performance | ✅ Active | Basic monitoring |
| `performance_middleware.py` | Advanced performance | ✅ Active | High-throughput APIs |
| `security_advanced.py` | Advanced security | ✅ Active | Production |
| `aws_observability.py` | AWS observability | ✅ Active | AWS deployments |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `core/middleware_config.py` for middleware setup
- See `API_GUIDE.md` for API structure






