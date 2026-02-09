# BUL Enhanced API Documentation

## Overview

The BUL (Business Universal Language) Enhanced API is a modern, production-ready FastAPI application that provides AI-powered document generation for SMEs. This API features comprehensive security, performance monitoring, caching, and enterprise-grade features.

## Features

### Core Features
- **AI-Powered Document Generation**: Generate business documents using advanced AI models
- **Intelligent Agent Management**: Automatic agent selection based on document requirements
- **Multi-format Export**: Support for Markdown, HTML, PDF, DOCX, and TXT formats
- **Real-time Processing**: WebSocket support for real-time updates
- **Batch Processing**: Generate multiple documents simultaneously
- **Advanced Caching**: Redis-based caching for improved performance

### Security Features
- **Authentication & Authorization**: JWT-based authentication with role-based access control
- **Rate Limiting**: Advanced rate limiting with burst protection
- **Threat Detection**: Real-time threat detection and blocking
- **Input Validation**: Comprehensive input validation and sanitization
- **Security Headers**: Automatic security headers for all responses

### Performance Features
- **Connection Pooling**: Optimized database connection pooling
- **Response Compression**: Automatic response compression
- **Performance Monitoring**: Real-time performance metrics
- **Caching Strategies**: Multi-level caching for optimal performance
- **Async Processing**: Full async/await support throughout

## API Endpoints

### Core Endpoints

#### `GET /`
Root endpoint with system information and capabilities.

**Response:**
```json
{
  "message": "BUL Enhanced API - Business Universal Language",
  "version": "3.0.0",
  "status": "running",
  "docs": "/docs",
  "features": [...],
  "capabilities": {
    "languages": ["es", "en", "pt", "fr", "de", "it", "ru", "zh", "ja"],
    "formats": ["markdown", "html", "pdf", "docx", "txt"],
    "business_areas": [...],
    "document_types": [...]
  }
}
```

#### `GET /health`
Comprehensive health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "3.0.0",
  "uptime": 3600.0,
  "components": {
    "engine": {"status": "healthy", "initialized": true},
    "agent_manager": {"status": "healthy", "initialized": true},
    "cache": {"status": "healthy", "enabled": true},
    "rate_limiter": {"status": "healthy"}
  },
  "metrics": {
    "total_requests": 1000,
    "uptime_seconds": 3600,
    "requests_per_minute": 16.67
  },
  "performance": {
    "avg_response_time": 0.5,
    "cache_hit_rate": 0.85,
    "error_rate": 0.01
  },
  "dependencies": {
    "openrouter": "healthy",
    "database": "healthy",
    "cache_backend": "healthy"
  }
}
```

### Document Generation

#### `POST /generate`
Generate a single business document.

**Request Body:**
```json
{
  "query": "Create a marketing strategy for a small restaurant",
  "business_area": "marketing",
  "document_type": "marketing_strategy",
  "company_name": "Restaurant ABC",
  "industry": "food_service",
  "company_size": "small",
  "target_audience": "local customers",
  "language": "es",
  "format": "markdown",
  "style": "professional",
  "priority": "normal",
  "cache_ttl": 3600,
  "include_metadata": true
}
```

**Response:**
```json
{
  "id": "doc_123456",
  "request_id": "req_789012",
  "content": "# Marketing Strategy for Restaurant ABC\n\n...",
  "title": "Marketing Strategy for Restaurant ABC",
  "summary": "Comprehensive marketing strategy...",
  "business_area": "marketing",
  "document_type": "marketing_strategy",
  "word_count": 1500,
  "processing_time": 2.5,
  "confidence_score": 0.85,
  "created_at": "2024-01-01T00:00:00Z",
  "agent_used": "Marketing Expert Agent",
  "format": "markdown",
  "style": "professional",
  "metadata": {...},
  "quality_score": 0.88,
  "readability_score": 0.82
}
```

#### `POST /generate/batch`
Generate multiple documents in batch.

**Request Body:**
```json
{
  "requests": [
    {
      "query": "Create a business plan",
      "business_area": "strategy",
      "document_type": "business_plan"
    },
    {
      "query": "Create a marketing strategy",
      "business_area": "marketing",
      "document_type": "marketing_strategy"
    }
  ],
  "parallel": true,
  "priority": "normal",
  "max_concurrent": 5
}
```

**Response:**
```json
[
  {
    "id": "doc_123456",
    "content": "# Business Plan\n\n...",
    "title": "Business Plan",
    "processing_time": 3.2,
    "agent_used": "Strategy Expert Agent"
  },
  {
    "id": "doc_123457",
    "content": "# Marketing Strategy\n\n...",
    "title": "Marketing Strategy",
    "processing_time": 2.8,
    "agent_used": "Marketing Expert Agent"
  }
]
```

### System Information

#### `GET /business-areas`
Get available business areas.

**Response:**
```json
[
  "marketing",
  "sales",
  "operations",
  "hr",
  "finance",
  "legal",
  "technical",
  "content",
  "strategy",
  "customer_service"
]
```

#### `GET /document-types`
Get available document types.

**Response:**
```json
[
  "business_plan",
  "marketing_strategy",
  "sales_proposal",
  "operational_manual",
  "hr_policy",
  "financial_report",
  "legal_contract",
  "technical_specification",
  "content_strategy",
  "strategic_plan",
  "customer_service_guide"
]
```

#### `GET /agents`
Get all available agents.

**Response:**
```json
[
  {
    "id": "agent_001",
    "name": "Marketing Expert Agent",
    "agent_type": "marketing",
    "experience_years": 5,
    "success_rate": 0.92,
    "total_documents_generated": 150,
    "average_rating": 4.8,
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z",
    "last_used": "2024-01-01T12:00:00Z"
  }
]
```

#### `GET /agents/stats`
Get agent statistics.

**Response:**
```json
{
  "total_agents": 10,
  "active_agents": 8,
  "total_documents_generated": 1500,
  "average_success_rate": 0.89,
  "agent_types": ["marketing", "sales", "strategy"],
  "is_initialized": true,
  "performance_metrics": {
    "avg_processing_time": 2.5,
    "success_rate": 0.89,
    "error_rate": 0.11
  }
}
```

### WebSocket Endpoints

#### `WS /ws/{user_id}`
WebSocket endpoint for real-time updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/user123');
```

**Message Types:**
- `ping`: Keep connection alive
- `subscribe`: Subscribe to specific updates
- `document_generation`: Real-time document generation updates
- `system_status`: System status updates

## Authentication

### JWT Token Authentication

The API uses JWT tokens for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

### API Key Authentication

For programmatic access, use API keys:

```
Authorization: Bearer bul_<your_api_key>
```

## Rate Limiting

The API implements comprehensive rate limiting:

- **Default**: 100 requests per minute
- **Document Generation**: 10 requests per minute
- **Batch Generation**: 5 requests per 5 minutes
- **Burst Protection**: Additional burst limits for high-traffic scenarios

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Error Handling

### Standard Error Response

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456",
  "status_code": 400,
  "suggestions": ["Suggestion 1", "Suggestion 2"]
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Access denied
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Performance Optimization

### Caching

The API implements multi-level caching:

1. **Response Caching**: Cached responses for identical requests
2. **Database Caching**: Cached database queries
3. **External API Caching**: Cached external API responses

Cache headers:
- `Cache-Control`: Cache control directives
- `ETag`: Entity tag for cache validation
- `Last-Modified`: Last modification time

### Compression

Automatic response compression for:
- JSON responses > 1KB
- Text-based content types
- API responses

## Security Features

### Security Headers

Automatic security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`

### Input Validation

Comprehensive input validation:
- Request size limits
- Parameter type validation
- SQL injection prevention
- XSS protection
- Path traversal prevention

### Threat Detection

Real-time threat detection:
- Suspicious request patterns
- Brute force attack detection
- Rate limit violations
- Malicious payload detection

## Monitoring and Metrics

### Health Checks

The `/health` endpoint provides comprehensive system health information:

- **System Status**: Overall system health
- **Component Status**: Individual component health
- **Performance Metrics**: Response times, throughput
- **Dependency Status**: External service status
- **Resource Usage**: Memory, CPU, disk usage

### Performance Metrics

Real-time performance metrics:
- Request count and rate
- Average response time
- Error rates
- Cache hit rates
- Database performance
- External API performance

### Logging

Comprehensive logging with structured JSON logs:
- Request/response logging
- Error logging with stack traces
- Security event logging
- Performance metrics logging
- Audit trail logging

## Configuration

### Environment Variables

Required environment variables:
```bash
# API Configuration
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_KEY=your_openai_key  # Optional

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/bul_db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=your_secret_key_here
JWT_ALGORITHM=HS256

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
```

### Configuration File

The API supports configuration via YAML/JSON files:

```yaml
api:
  openrouter_api_key: ${OPENROUTER_API_KEY}
  default_model: "openai/gpt-4"
  max_tokens: 4000
  temperature: 0.7

database:
  url: ${DATABASE_URL}
  pool_size: 10
  max_overflow: 20

cache:
  enabled: true
  backend: "redis"
  redis_url: ${REDIS_URL}
  default_ttl: 3600

security:
  secret_key: ${SECRET_KEY}
  rate_limit_requests: 100
  rate_limit_window: 60

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins: ["*"]
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.enhanced_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/bul_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: bul_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bul-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bul-api
  template:
    metadata:
      labels:
        app: bul-api
    spec:
      containers:
      - name: bul-api
        image: bul-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: bul-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: bul-secrets
              key: redis-url
---
apiVersion: v1
kind: Service
metadata:
  name: bul-api-service
spec:
  selector:
    app: bul-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Testing

### Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from api.enhanced_api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "BUL Enhanced API" in response.json()["message"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded", "unhealthy"]

def test_generate_document():
    response = client.post("/generate", json={
        "query": "Create a business plan",
        "business_area": "strategy",
        "document_type": "business_plan"
    })
    assert response.status_code == 200
    assert "content" in response.json()
```

### Integration Tests

```python
import pytest
import asyncio
from api.enhanced_api import app
from database.enhanced_database import get_database_manager

@pytest.mark.asyncio
async def test_database_connection():
    db_manager = await get_database_manager()
    health = await db_manager.health_check()
    assert health["database_manager"]["status"] == "healthy"

@pytest.mark.asyncio
async def test_cache_functionality():
    db_manager = await get_database_manager()
    await db_manager.set_cache("test_key", "test_value", 60)
    value = await db_manager.get_cache("test_key")
    assert value == "test_value"
```

## Best Practices

### API Usage

1. **Use HTTPS**: Always use HTTPS in production
2. **Handle Errors**: Implement proper error handling
3. **Rate Limiting**: Respect rate limits and implement backoff
4. **Caching**: Use appropriate cache headers
5. **Authentication**: Secure your API keys and tokens

### Performance

1. **Batch Requests**: Use batch endpoints for multiple documents
2. **Caching**: Leverage caching for repeated requests
3. **Compression**: Enable compression for large responses
4. **Connection Pooling**: Use connection pooling for database access

### Security

1. **Input Validation**: Validate all input data
2. **Authentication**: Implement proper authentication
3. **Authorization**: Use role-based access control
4. **Monitoring**: Monitor for security threats
5. **Logging**: Log security events

## Support

For support and questions:
- **Documentation**: `/docs` (Swagger UI)
- **ReDoc**: `/redoc` (Alternative documentation)
- **OpenAPI Schema**: `/openapi.json`
- **Health Check**: `/health`

## Changelog

### Version 3.0.0
- Enhanced API with modern FastAPI patterns
- Comprehensive security features
- Advanced performance monitoring
- Multi-level caching
- WebSocket support
- Batch processing
- Enterprise-grade features

### Version 2.0.0
- Basic document generation
- Simple agent management
- Basic caching
- Standard error handling

### Version 1.0.0
- Initial release
- Basic API functionality