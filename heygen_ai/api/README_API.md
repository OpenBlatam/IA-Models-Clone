# HeyGen AI API - FastAPI Implementation

Production-ready FastAPI application for the HeyGen AI Equivalent System with comprehensive features, scalability, and enterprise-grade architecture.

## 🚀 Features

### Core API Features
- **RESTful API**: Complete REST API with OpenAPI/Swagger documentation
- **Async Processing**: Background task processing for video generation
- **Database Integration**: SQLAlchemy with async support
- **Authentication**: JWT-based authentication with API keys
- **Rate Limiting**: Configurable rate limiting per user
- **CORS Support**: Cross-origin resource sharing configuration
- **File Upload/Download**: Secure file handling with streaming responses

### Scalability Features
- **Background Tasks**: Async video processing with Celery integration
- **Database Pooling**: Connection pooling for high concurrency
- **Caching**: Redis-based caching for improved performance
- **Load Balancing**: Ready for horizontal scaling
- **Monitoring**: Prometheus metrics and structured logging

### Security Features
- **Input Validation**: Pydantic models with comprehensive validation
- **Error Handling**: Structured error responses with proper HTTP codes
- **API Key Management**: Secure API key validation
- **CORS Protection**: Configurable cross-origin policies
- **Rate Limiting**: Protection against abuse

## 📁 Project Structure

```
api/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration management
├── database.py            # Database models and operations
├── start_server.py        # Server startup and lifecycle management
├── requirements-api.txt   # API-specific dependencies
├── README_API.md         # This documentation
├── .env.example          # Environment variables template
└── tests/                # API tests
    ├── test_main.py
    ├── test_database.py
    └── conftest.py
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- SQLite (development) or PostgreSQL (production)
- Redis (optional, for caching and background tasks)

### Setup

1. **Clone and navigate to the API directory**
```bash
cd agents/backend/onyx/server/features/heygen_ai/api
```

2. **Install dependencies**
```bash
pip install -r requirements-api.txt
```

3. **Environment configuration**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Initialize database**
```bash
# Database will be auto-created on first run
python -c "from database import init_database; import asyncio; asyncio.run(init_database())"
```

## 🚀 Quick Start

### Development Mode
```bash
# Start with auto-reload
python start_server.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
# Set environment
export ENVIRONMENT=production

# Start production server
python start_server.py

# Or use gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "timestamp": 1640995200.0,
  "version": "1.0.0",
  "components": {
    "transformer_model": true,
    "diffusion_manager": true,
    "experiment_tracker": true
  }
}
```

#### Video Generation
```bash
POST /api/v1/videos/generate
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "script": "A beautiful sunset over the ocean with gentle waves",
  "voice_id": "Voice 1",
  "language": "en",
  "quality": "medium",
  "duration": 30
}
```

Response:
```json
{
  "video_id": "video_1640995200_user-123",
  "status": "processing",
  "processing_time": 0.0,
  "metadata": {
    "script_length": 47,
    "quality": "medium",
    "voice_id": "Voice 1",
    "language": "en"
  }
}
```

#### Video Status
```bash
GET /api/v1/videos/{video_id}
Authorization: Bearer your-api-key
```

#### Download Video
```bash
GET /api/v1/videos/{video_id}/download
Authorization: Bearer your-api-key
```

#### List Voices
```bash
GET /api/v1/voices
```

#### List Models
```bash
GET /api/v1/models
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Database
DATABASE_URL=sqlite+aiosqlite:///./heygen_ai.db

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET=your-jwt-secret-change-in-production

# CORS
CORS_ORIGINS=*

# Processing
MAX_CONCURRENT_VIDEOS=10
DEFAULT_QUALITY=medium

# External Services
OPENROUTER_API_KEY=your-openrouter-key
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_TOKEN=your-huggingface-token

# Storage
STORAGE_TYPE=local
S3_BUCKET=your-s3-bucket
S3_ACCESS_KEY=your-s3-access-key
S3_SECRET_KEY=your-s3-secret-key

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Cache
CACHE_ENABLED=true
REDIS_URL=redis://localhost:6379
```

### Configuration Classes

The application supports different configuration classes:

- **DevelopmentSettings**: For development with debug enabled
- **ProductionSettings**: For production with security features
- **TestingSettings**: For testing with minimal resources

## 🔐 Authentication

### API Key Authentication

The API uses Bearer token authentication:

```bash
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8000/api/v1/videos/generate
```

### Creating API Keys

In production, implement proper API key management:

```python
from database import user_repo
import secrets

# Generate API key
api_key = secrets.token_urlsafe(32)

# Create user with API key
user_data = {
    "username": "test_user",
    "email": "user@example.com",
    "api_key": api_key
}
user = await user_repo.create_user(user_data)
```

## 📊 Database Models

### User Model
```python
class User(Base):
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    api_key = Column(String(255), unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

### Video Model
```python
class Video(Base):
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(100), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    script = Column(Text, nullable=False)
    voice_id = Column(String(50), nullable=False)
    status = Column(String(20), default="processing")
    file_path = Column(String(500), nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=api --cov-report=html
```

### Test Examples

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_video_generation():
    response = client.post(
        "/api/v1/videos/generate",
        headers={"Authorization": "Bearer test-api-key"},
        json={
            "script": "Test video generation",
            "voice_id": "Voice 1",
            "quality": "low"
        }
    )
    assert response.status_code == 200
    assert "video_id" in response.json()
```

## 🚀 Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["python", "start_server.py"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/heygen_ai
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=heygen_ai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine

volumes:
  postgres_data:
```

### Production Deployment

1. **Set environment variables**
```bash
export ENVIRONMENT=production
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost/heygen_ai
export SECRET_KEY=your-production-secret-key
```

2. **Run with Gunicorn**
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

3. **Use reverse proxy (Nginx)**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📈 Monitoring

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health | jq
```

### Logging
```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics
```python
from prometheus_client import Counter, Histogram

# Define metrics
video_generation_requests = Counter(
    'video_generation_requests_total',
    'Total video generation requests'
)

video_generation_duration = Histogram(
    'video_generation_duration_seconds',
    'Video generation duration'
)
```

## 🔧 Development

### Code Quality
```bash
# Format code
black api/

# Lint code
flake8 api/

# Type checking
mypy api/

# Sort imports
isort api/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install
```

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check DATABASE_URL in .env
   - Ensure database server is running
   - Verify network connectivity

2. **Model Loading Error**
   - Check if models are properly initialized
   - Verify model files exist
   - Check GPU memory availability

3. **Rate Limiting**
   - Check rate limit configuration
   - Verify API key is valid
   - Monitor request frequency

4. **File Upload Issues**
   - Check file size limits
   - Verify file permissions
   - Ensure storage directory exists

### Debug Mode
```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start server with debug
python start_server.py
```

## 📞 Support

- **Documentation**: Check this README and API docs
- **Issues**: Report bugs and feature requests
- **Discussions**: Community discussions and questions

## 🔄 Changelog

### v1.0.0
- Initial FastAPI implementation
- Complete REST API with OpenAPI documentation
- Database integration with SQLAlchemy
- Authentication and authorization
- Background task processing
- Comprehensive error handling
- Production-ready configuration
- Docker deployment support

---

*Built with FastAPI, SQLAlchemy, and modern Python best practices* 