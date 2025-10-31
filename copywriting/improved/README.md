# Improved Copywriting Service

A clean, high-performance FastAPI-based copywriting service following modern Python and FastAPI best practices.

## Features

- **Clean Architecture**: Modular, maintainable code structure
- **Async Operations**: Full async/await support for high performance
- **Type Safety**: Comprehensive type hints with Pydantic v2
- **Error Handling**: Robust error handling with custom exceptions
- **Caching**: Redis-based caching for improved performance
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Monitoring**: Prometheus metrics and health checks
- **Testing**: Comprehensive test suite with pytest
- **Documentation**: Auto-generated API documentation

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="sqlite+aiosqlite:///./copywriting.db"
export REDIS_URL="redis://localhost:6379/0"
export SECURITY_SECRET_KEY="your-secret-key-here"
```

### Running the Service

```bash
# Development mode
python -m improved.main

# Production mode
uvicorn improved.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v2/copywriting/health

## API Usage

### Generate Copywriting Content

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v2/copywriting/generate",
        json={
            "topic": "AI-Powered Marketing",
            "target_audience": "Marketing professionals",
            "key_points": ["Automation", "Personalization", "ROI"],
            "tone": "professional",
            "style": "direct_response",
            "purpose": "sales",
            "word_count": 500,
            "include_cta": True,
            "variants_count": 3
        }
    )
    
    result = response.json()
    print(f"Generated {result['total_variants']} variants")
```

### Batch Generation

```python
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v2/copywriting/generate/batch",
        json={
            "requests": [
                {
                    "topic": "Topic 1",
                    "target_audience": "Audience 1"
                },
                {
                    "topic": "Topic 2", 
                    "target_audience": "Audience 2"
                }
            ]
        }
    )
    
    result = response.json()
    print(f"Processed {result['success_count']} requests")
```

### Submit Feedback

```python
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v2/copywriting/feedback",
        json={
            "variant_id": "variant-uuid-here",
            "rating": 4,
            "feedback_text": "Great content!",
            "improvements": ["More specific examples"],
            "is_helpful": True
        }
    )
```

## Configuration

The service uses environment variables for configuration. Key settings:

### Database
- `DB_URL`: Database connection string
- `DB_POOL_SIZE`: Connection pool size (default: 10)
- `DB_MAX_OVERFLOW`: Max overflow connections (default: 20)

### Redis
- `REDIS_URL`: Redis connection string
- `REDIS_MAX_CONNECTIONS`: Max Redis connections (default: 10)

### API
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `API_WORKERS`: Number of workers (default: 1)
- `API_RATE_LIMIT_REQUESTS`: Rate limit requests per minute (default: 100)

### Security
- `SECURITY_SECRET_KEY`: Secret key for JWT (required)
- `SECURITY_API_KEYS`: Comma-separated list of valid API keys

### Caching
- `CACHE_ENABLED`: Enable caching (default: true)
- `CACHE_DEFAULT_TTL`: Default cache TTL in seconds (default: 300)
- `CACHE_COPYWRITING_TTL`: Copywriting cache TTL (default: 600)

## Architecture

```
improved/
├── __init__.py          # Package initialization
├── app.py              # FastAPI application factory
├── main.py             # Application entry point
├── config.py           # Configuration management
├── schemas.py          # Pydantic models
├── services.py         # Business logic
├── routes.py           # API routes
├── exceptions.py       # Custom exceptions
├── requirements.txt    # Dependencies
├── README.md          # This file
└── tests/             # Test suite
    ├── __init__.py
    └── test_schemas.py
```

## Key Improvements

### 1. Clean Architecture
- Separation of concerns with distinct layers
- Dependency injection for better testability
- Modular design for maintainability

### 2. Type Safety
- Comprehensive type hints throughout
- Pydantic v2 models for validation
- Enum types for constrained values

### 3. Error Handling
- Custom exception hierarchy
- Structured error responses
- Proper HTTP status codes

### 4. Performance
- Async/await throughout
- Redis caching
- Connection pooling
- Rate limiting

### 5. Monitoring
- Health checks
- Prometheus metrics
- Request logging
- Performance tracking

### 6. Testing
- Comprehensive test suite
- Async test support
- Schema validation tests
- Error handling tests

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=improved

# Run specific test file
pytest tests/test_schemas.py
```

### Code Quality

```bash
# Format code
black improved/

# Sort imports
isort improved/

# Lint code
flake8 improved/

# Type checking
mypy improved/
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY improved/ ./improved/
EXPOSE 8000

CMD ["uvicorn", "improved.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# Production settings
export ENVIRONMENT=production
export DEBUG=false
export API_WORKERS=4
export DB_URL=postgresql+asyncpg://user:pass@localhost/copywriting
export REDIS_URL=redis://localhost:6379/0
export SECURITY_SECRET_KEY=your-production-secret-key
```

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass
5. Run code quality checks

## License

This project is part of the Blatam Academy curriculum.






























