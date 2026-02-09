# Advanced Features Guide

This document describes all advanced features implemented for production-ready AWS deployment.

## 🎯 Features Overview

### 1. **Distributed Tracing with OpenTelemetry**
- **Purpose**: Track requests across microservices
- **Implementation**: OpenTelemetry with OTLP exporter
- **Configuration**: Set `OTLP_ENDPOINT` environment variable
- **Benefits**: 
  - End-to-end request tracing
  - Performance bottleneck identification
  - Service dependency mapping

### 2. **Rate Limiting**
- **Purpose**: Protect API from abuse
- **Implementation**: SlowAPI with Redis backend
- **Configuration**: 
  - `ENABLE_RATE_LIMITING=true`
  - `REDIS_URL=redis://...`
- **Benefits**:
  - Prevents DDoS attacks
  - Fair resource allocation
  - Cost control

### 3. **Circuit Breaker**
- **Purpose**: Prevent cascading failures
- **Implementation**: Custom middleware with configurable thresholds
- **Configuration**:
  - `ENABLE_CIRCUIT_BREAKER=true`
  - `CIRCUIT_BREAKER_FAILURE_THRESHOLD=5`
  - `CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60`
- **Benefits**:
  - Fast failure detection
  - Automatic recovery
  - System stability

### 4. **Redis Caching**
- **Purpose**: Reduce database load and improve response times
- **Implementation**: Middleware with Redis backend
- **Configuration**:
  - `ENABLE_REDIS_CACHE=true`
  - `REDIS_URL=redis://...`
  - `CACHE_TTL=300` (5 minutes)
- **Benefits**:
  - Faster API responses
  - Reduced database load
  - Lower costs

### 5. **Structured Logging**
- **Purpose**: Better log analysis and observability
- **Implementation**: JSON logging with request IDs
- **Configuration**: Automatic, no configuration needed
- **Benefits**:
  - Easy log parsing
  - Request tracing
  - Better debugging

### 6. **Security Headers**
- **Purpose**: Protect against common web vulnerabilities
- **Implementation**: Middleware adding security headers
- **Headers Added**:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Strict-Transport-Security`
  - `Content-Security-Policy`
  - `Referrer-Policy`
  - `Permissions-Policy`

### 7. **OAuth2 Authentication**
- **Purpose**: Secure API access with JWT tokens
- **Implementation**: FastAPI OAuth2 with JWT
- **Features**:
  - Access tokens (30 min expiry)
  - Refresh tokens (7 days expiry)
  - Token blacklisting
  - Scope-based authorization
  - Rate limiting on login

### 8. **Celery Workers**
- **Purpose**: Background task processing
- **Implementation**: Celery with Redis broker
- **Task Queues**:
  - `trajectory`: Trajectory optimization
  - `model`: Model inference
  - `data`: Data processing
  - `reports`: Report generation
- **Benefits**:
  - Async processing
  - Scalable workers
  - Task retry logic

### 9. **Kafka Event Streaming**
- **Purpose**: Event-driven architecture
- **Implementation**: Kafka producer/consumer
- **Event Types**:
  - `trajectory.optimized`
  - `movement.started`
  - `movement.completed`
  - `movement.failed`
  - `collision.detected`
  - `robot.connected`
  - `robot.disconnected`
- **Benefits**:
  - Decoupled services
  - Event replay
  - Scalable messaging

### 10. **Prometheus Metrics**
- **Purpose**: Application monitoring
- **Implementation**: Prometheus client library
- **Metrics Exposed**:
  - HTTP request metrics
  - Application metrics (movements, optimizations)
  - System metrics (connections, tasks)
  - Performance metrics (durations)
- **Endpoint**: `/metrics`

### 11. **API Gateway Integration**
- **Purpose**: Centralized API management
- **Features**:
  - Rate limiting (50 req/sec, burst 100)
  - Request/response transformation
  - Security filtering
  - Access logging

### 12. **WAF (Web Application Firewall)**
- **Purpose**: DDoS and attack protection
- **Rules**:
  - AWS Managed Common Rule Set
  - Known Bad Inputs Rule Set
  - Rate limiting (2000 req/5min per IP)
- **Benefits**:
  - Protection against OWASP Top 10
  - DDoS mitigation
  - Bot protection

## 🔧 Configuration

### Environment Variables

```bash
# OpenTelemetry
OTLP_ENDPOINT=http://localhost:4317
OTLP_INSECURE=true
ENABLE_TRACING=true

# Rate Limiting
ENABLE_RATE_LIMITING=true
REDIS_URL=redis://localhost:6379/0

# Circuit Breaker
ENABLE_CIRCUIT_BREAKER=true
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60

# Caching
ENABLE_REDIS_CACHE=true
CACHE_TTL=300

# OAuth2
JWT_SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=robot-movement-ai

# Celery
REDIS_URL=redis://localhost:6379/0
REDIS_RESULT_BACKEND=redis://localhost:6379/1

# Prometheus
ENABLE_PROMETHEUS=true
```

## 📊 Monitoring

### Prometheus Metrics

Access metrics at: `http://your-api/metrics`

Key metrics to monitor:
- `http_requests_total`: Total requests by method, endpoint, status
- `http_request_duration_seconds`: Request latency
- `robot_movements_total`: Robot movement counts
- `trajectory_optimizations_total`: Optimization counts
- `active_connections`: Current connections
- `active_tasks`: Background task counts

### Grafana Dashboards

Import the provided Grafana dashboard JSON for:
- Request rate and latency
- Error rates
- System resource usage
- Business metrics (movements, optimizations)

## 🚀 Usage Examples

### Using Celery Workers

```python
from aws.workers.tasks.trajectory_tasks import optimize_trajectory

# Queue trajectory optimization
result = optimize_trajectory.delay(
    waypoints=[...],
    obstacles=[...],
    optimization_type="astar"
)

# Get result
optimized_trajectory = result.get(timeout=30)
```

### Publishing Events

```python
from aws.messaging.kafka_config import publish_event, EventTypes

# Publish movement event
publish_event(
    EventTypes.MOVEMENT_STARTED,
    {
        "robot_id": "robot-1",
        "target_position": [0.5, 0.3, 0.2]
    }
)
```

### OAuth2 Authentication

```python
from fastapi import Depends
from aws.security.oauth2_config import get_current_active_user

@app.get("/api/v1/protected/endpoint")
async def protected_endpoint(
    current_user = Depends(get_current_active_user)
):
    return {"user": current_user.username}
```

## 🔒 Security Best Practices

1. **Always use HTTPS** in production
2. **Rotate JWT secrets** regularly
3. **Use strong passwords** for Redis/Kafka
4. **Enable WAF** for public APIs
5. **Monitor rate limits** and adjust as needed
6. **Review security headers** regularly
7. **Use least privilege** IAM roles
8. **Encrypt secrets** in Secrets Manager

## 📈 Performance Optimization

1. **Enable Redis caching** for frequently accessed data
2. **Use Celery workers** for long-running tasks
3. **Monitor Prometheus metrics** for bottlenecks
4. **Adjust rate limits** based on traffic patterns
5. **Use circuit breakers** to prevent cascading failures
6. **Optimize database queries** (use indexes)
7. **Enable connection pooling** for databases

## 🐛 Troubleshooting

### Rate Limiting Issues
- Check Redis connection
- Verify rate limit configuration
- Review CloudWatch logs

### Circuit Breaker Open
- Check external service health
- Review failure logs
- Wait for recovery timeout

### Kafka Connection Issues
- Verify bootstrap servers
- Check security groups
- Review MSK cluster status

### Celery Tasks Not Processing
- Check worker logs
- Verify Redis connection
- Check queue configuration

## 📚 Additional Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [OAuth2 Specification](https://oauth.net/2/)















