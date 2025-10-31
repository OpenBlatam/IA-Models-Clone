# AI Video System - Quick Start Guide

## 🚀 Getting Started

This guide will help you get up and running with the refactored AI Video system quickly.

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- Docker (optional)

## 🛠️ Installation

### 1. Clone and Setup

```bash
# Navigate to the refactored directory
cd agents/backend/onyx/server/features/ai_video/refactored

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the refactored directory:

```env
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://user:pass@localhost/ai_video

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-here

# External Services
OPENAI_API_KEY=your-openai-api-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

### 3. Database Setup

```bash
# Create database
createdb ai_video

# Run migrations (when implemented)
alembic upgrade head
```

### 4. Start Services

```bash
# Start Redis
redis-server

# Start PostgreSQL (if not running)
sudo service postgresql start

# Start the application
python main.py
```

## 🎯 Basic Usage

### 1. Create a Video

```python
from application.use_cases.video_use_cases import CreateVideoUseCase, CreateVideoRequest
from core.entities import Video, Template, Avatar

# Create video request
request = CreateVideoRequest(
    title="My First AI Video",
    description="A test video with AI avatar",
    template_id=template_id,
    avatar_id=avatar_id,
    script_content="Hello, this is my first AI-generated video!",
    user_id=user_id,
    quality="high",
    format="mp4",
    aspect_ratio="16:9"
)

# Execute use case
use_case = CreateVideoUseCase(
    video_repository=video_repo,
    template_repository=template_repo,
    avatar_repository=avatar_repo,
    script_repository=script_repo
)

response = await use_case.execute(request)
print(f"Video created: {response.video_id}")
```

### 2. Get Video Status

```python
from application.use_cases.video_use_cases import GetVideoUseCase, GetVideoRequest

request = GetVideoRequest(
    video_id=video_id,
    user_id=user_id
)

use_case = GetVideoUseCase(video_repository=video_repo)
response = await use_case.execute(request)

print(f"Status: {response.video['status']}")
print(f"Progress: {response.progress}%")
```

### 3. List User Videos

```python
from application.use_cases.video_use_cases import ListVideosUseCase, ListVideosRequest

request = ListVideosRequest(
    user_id=user_id,
    skip=0,
    limit=10
)

use_case = ListVideosUseCase(video_repository=video_repo)
response = await use_case.execute(request)

for video in response.videos:
    print(f"- {video['title']}: {video['status']}")
```

## 🏗️ Architecture Overview

### Domain Layer

```python
# Entities represent business objects
from core.entities import Video, Template, Avatar, Script

# Value objects are immutable
from core.value_objects import VideoConfig

# Repositories define data access contracts
from core.repositories import VideoRepository, TemplateRepository
```

### Application Layer

```python
# Use cases implement business logic
from application.use_cases.video_use_cases import CreateVideoUseCase

# DTOs for data transfer
from application.dto import CreateVideoRequest, CreateVideoResponse
```

### Infrastructure Layer

```python
# Repository implementations
from infrastructure.persistence import PostgresVideoRepository

# External service integrations
from infrastructure.external_services import OpenAIService
```

### Presentation Layer

```python
# FastAPI routes
from presentation.api.routes import video_router

# Middleware
from presentation.middleware import auth_middleware, logging_middleware
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=core --cov=application --cov=infrastructure
```

### Test Examples

```python
# Unit test example
import pytest
from core.entities import Video
from application.use_cases.video_use_cases import CreateVideoUseCase

class TestCreateVideoUseCase:
    @pytest.mark.asyncio
    async def test_create_video_success(self):
        # Arrange
        use_case = CreateVideoUseCase(mock_repos)
        request = CreateVideoRequest(...)
        
        # Act
        response = await use_case.execute(request)
        
        # Assert
        assert response.video_id is not None
        assert response.status == "pending"
```

## 🔧 Configuration

### Settings Structure

```python
from shared.config.settings import get_settings

settings = get_settings()

# Access configuration
print(settings.database.url)
print(settings.cache.redis.url)
print(settings.security.secret_key)
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/ai_video
DATABASE_POOL_SIZE=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20

# Security
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# External Services
OPENAI_API_KEY=your-openai-key
AWS_REGION=us-east-1
```

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/metrics
```

### Logs

```bash
# View application logs
tail -f logs/app.log
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

```bash
# Build and run
docker build -t ai-video-system .
docker run -p 8000:8000 ai-video-system
```

### Production Configuration

```env
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql://prod_user:prod_pass@prod_db/ai_video
REDIS_URL=redis://prod_redis:6379/0
SECRET_KEY=production-secret-key
```

## 🔍 Debugging

### Development Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug settings
settings = get_settings()
settings.debug = True
```

### Common Issues

1. **Database Connection**: Check DATABASE_URL and PostgreSQL status
2. **Redis Connection**: Verify Redis is running and accessible
3. **Missing Dependencies**: Ensure all requirements are installed
4. **Permission Issues**: Check file permissions and database access

## 📚 Next Steps

1. **Explore the Codebase**: Review the architecture and implementation
2. **Run Examples**: Execute the provided code examples
3. **Add Features**: Implement new use cases and entities
4. **Write Tests**: Add comprehensive test coverage
5. **Deploy**: Set up production deployment

## 🆘 Support

- **Documentation**: Check the main documentation files
- **Issues**: Report bugs and feature requests
- **Community**: Join the development community
- **Examples**: Review the example implementations

## 🎉 Congratulations!

You're now ready to work with the refactored AI Video system! The new architecture provides a solid foundation for building scalable, maintainable video generation applications. 