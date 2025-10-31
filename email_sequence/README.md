# Email Sequence AI System

A comprehensive, AI-powered email sequence management system built with FastAPI, featuring LangChain integration, advanced analytics, and scalable architecture.

## 🚀 Features

### Core Functionality
- **AI-Powered Sequence Generation**: Create intelligent email sequences using LangChain and OpenAI
- **Advanced Personalization**: Dynamic content personalization based on subscriber data
- **Multi-Step Sequences**: Support for email, delay, condition, action, and webhook steps
- **A/B Testing**: Built-in A/B testing capabilities for optimization
- **Real-time Analytics**: Comprehensive tracking and analytics dashboard

### Technical Features
- **Async/Await Architecture**: Built with FastAPI for high performance
- **Database Integration**: SQLAlchemy 2.0 with async PostgreSQL support
- **Redis Caching**: High-performance caching for improved response times
- **Comprehensive Monitoring**: Prometheus metrics and structured logging
- **Error Handling**: Robust error handling with custom exceptions
- **Security**: JWT authentication, rate limiting, and security headers
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## 📋 Requirements

### System Requirements
- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- 4GB+ RAM
- 2+ CPU cores

### Python Dependencies
See `requirements-fastapi.txt` for complete dependency list.

Key dependencies:
- FastAPI 0.104+
- SQLAlchemy 2.0+
- Redis 5.0+
- LangChain 0.1+
- OpenAI 1.3+
- Pydantic 2.5+

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd email-sequence-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements-fastapi.txt
```

### 4. Environment Configuration
Create a `.env` file:
```env
# Application
APP_NAME=Email Sequence AI
DEBUG=false
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/email_sequences

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=noreply@yourdomain.com
FROM_NAME=Your Company

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256

# Performance
MAX_CONCURRENT_SEQUENCES=50
MAX_CONCURRENT_EMAILS=100
CACHE_TTL_SECONDS=300
```

### 5. Database Setup
```bash
# Create database
createdb email_sequences

# Run migrations (if using Alembic)
alembic upgrade head
```

### 6. Start the Application
```bash
# Development
python main.py

# Production
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
Include JWT token in Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

### Key Endpoints

#### Sequences
- `GET /email-sequences` - List sequences
- `POST /email-sequences` - Create sequence
- `GET /email-sequences/{id}` - Get sequence
- `PUT /email-sequences/{id}` - Update sequence
- `DELETE /email-sequences/{id}` - Delete sequence
- `POST /email-sequences/{id}/activate` - Activate sequence

#### Subscribers
- `POST /email-sequences/{id}/subscribers` - Add subscribers
- `GET /subscribers` - List subscribers
- `POST /subscribers` - Create subscriber

#### Analytics
- `GET /email-sequences/{id}/analytics` - Get sequence analytics
- `GET /analytics/overview` - Get overview analytics

#### Health
- `GET /health` - Health check

### Example Usage

#### Create a Sequence
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/email-sequences",
        json={
            "name": "Welcome Series",
            "description": "Welcome new subscribers",
            "target_audience": "New subscribers",
            "goals": ["Welcome", "Onboard", "Engage"],
            "tone": "friendly",
            "steps": [
                {
                    "step_type": "email",
                    "order": 1,
                    "name": "Welcome Email",
                    "subject": "Welcome to our community!",
                    "content": "Thank you for joining us..."
                }
            ]
        },
        headers={"Authorization": "Bearer <token>"}
    )
```

## 🏗️ Architecture

### Project Structure
```
email_sequence/
├── api/                    # FastAPI routes and schemas
│   ├── routes.py          # API endpoints
│   ├── schemas.py         # Pydantic models
│   └── __init__.py
├── core/                  # Core functionality
│   ├── config.py          # Configuration management
│   ├── database.py        # Database models and connection
│   ├── cache.py           # Redis caching
│   ├── dependencies.py    # Dependency injection
│   ├── exceptions.py      # Custom exceptions
│   ├── middleware.py      # Custom middleware
│   ├── monitoring.py      # Performance monitoring
│   └── email_sequence_engine.py
├── models/                # Data models
│   ├── sequence.py
│   ├── template.py
│   └── subscriber.py
├── services/              # Business logic
│   ├── langchain_service.py
│   ├── delivery_service.py
│   └── analytics_service.py
├── main.py               # FastAPI application
└── requirements-fastapi.txt
```

### Key Components

#### 1. FastAPI Application (`main.py`)
- Application factory pattern
- Middleware configuration
- Exception handling
- CORS setup

#### 2. API Layer (`api/`)
- RESTful endpoints
- Request/response validation
- Error handling
- Authentication

#### 3. Core Services (`core/`)
- Configuration management
- Database integration
- Caching layer
- Monitoring and logging

#### 4. Business Logic (`services/`)
- LangChain integration
- Email delivery
- Analytics processing

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | Email Sequence AI |
| `DEBUG` | Debug mode | false |
| `DATABASE_URL` | Database connection | postgresql+asyncpg://... |
| `REDIS_URL` | Redis connection | redis://localhost:6379/0 |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `SMTP_HOST` | SMTP server | localhost |
| `SECRET_KEY` | JWT secret key | - |

### Performance Tuning

#### Database
- Connection pooling: `DB_POOL_SIZE=10`
- Max overflow: `DB_MAX_OVERFLOW=20`
- Connection recycling: 3600 seconds

#### Redis
- Max connections: `REDIS_MAX_CONNECTIONS=10`
- Connection pooling enabled
- Keep-alive enabled

#### Application
- Max concurrent sequences: `MAX_CONCURRENT_SEQUENCES=50`
- Max concurrent emails: `MAX_CONCURRENT_EMAILS=100`
- Cache TTL: `CACHE_TTL_SECONDS=300`

## 📊 Monitoring

### Metrics
- Request count and duration
- Email delivery metrics
- Cache hit/miss rates
- Database query performance
- Error rates and types

### Health Checks
- Database connectivity
- Redis connectivity
- Service status
- External API health

### Logging
- Structured JSON logging
- Request/response logging
- Error tracking
- Performance metrics

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements-fastapi.txt .
RUN pip install -r requirements-fastapi.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker"]
```

### Production Considerations
1. **Load Balancing**: Use nginx or similar
2. **Database**: Use connection pooling
3. **Caching**: Redis cluster for high availability
4. **Monitoring**: Prometheus + Grafana
5. **Logging**: Centralized logging system
6. **Security**: HTTPS, rate limiting, input validation

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test Coverage
```bash
pytest --cov=email_sequence tests/
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
black .
isort .
flake8 .
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API documentation at `/docs`

## 🔄 Changelog

### Version 2.0.0
- Complete FastAPI rewrite
- Async/await architecture
- Redis caching implementation
- Comprehensive monitoring
- Enhanced error handling
- Performance optimizations

### Version 1.0.0
- Initial release
- Basic email sequence functionality
- LangChain integration
- Simple analytics






























