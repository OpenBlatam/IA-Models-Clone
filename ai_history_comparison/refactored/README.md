# AI History Comparison System - Refactored Architecture

## 🚀 Overview

This is the refactored version of the AI History Comparison System, built with clean architecture principles and modern Python practices. The system provides comprehensive analysis, comparison, and tracking of AI model outputs over time.

## 🏗️ Architecture

The system follows **Clean Architecture** principles with clear separation of concerns:

```
refactored/
├── core/                    # Domain layer (business logic)
│   ├── domain.py           # Domain entities and value objects
│   ├── services.py         # Domain services
│   ├── events.py           # Domain events
│   └── specifications.py   # Business rules
├── application/            # Application layer (use cases)
│   ├── use_cases.py        # Business workflows
│   ├── dto.py             # Data transfer objects
│   └── services.py        # Application services
├── infrastructure/         # Infrastructure layer (external concerns)
│   ├── database.py        # Database management
│   ├── repositories.py    # Data access
│   └── services.py        # External services
├── presentation/          # Presentation layer (API)
│   ├── api.py            # FastAPI application
│   ├── endpoints.py      # API endpoints
│   └── middleware.py     # Middleware components
└── config/               # Configuration management
    ├── settings.py       # Application settings
    └── environment.py    # Environment configuration
```

## ✨ Key Features

### Core Functionality
- **Content Analysis**: Comprehensive analysis of AI-generated content
- **Model Comparison**: Compare different AI models and versions
- **Trend Analysis**: Track performance trends over time
- **Quality Reporting**: Generate detailed quality reports
- **Real-time Monitoring**: Monitor system health and performance

### Technical Features
- **Clean Architecture**: Separation of concerns with dependency inversion
- **Type Safety**: Full type hints with Pydantic validation
- **Async Support**: FastAPI with async/await throughout
- **Database Agnostic**: Repository pattern with SQLAlchemy
- **Configuration Management**: Environment-based configuration
- **Comprehensive Logging**: Structured logging with request tracking
- **Error Handling**: Global error handling with detailed responses
- **Security**: Authentication, rate limiting, and security headers
- **Documentation**: Auto-generated OpenAPI/Swagger documentation

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or poetry
- SQLite (default) or PostgreSQL (production)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-history-comparison/refactored
```

2. **Install dependencies**
```bash
# Using pip
pip install -r requirements.txt

# Using poetry
poetry install
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run the application**
```bash
# Development
python -m refactored.presentation.api

# Production
uvicorn refactored.presentation.api:app --host 0.0.0.0 --port 8000
```

## 📖 API Documentation

Once the application is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔧 Configuration

The system uses environment-based configuration with sensible defaults:

### Environment Variables

```bash
# Application
ENVIRONMENT=development
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=sqlite:///./ai_history.db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Security
SECURITY_API_KEY=your-api-key
SECURITY_ENABLE_AUTH=false
SECURITY_REQUESTS_PER_MINUTE=100

# Logging
LOGGING_LEVEL=INFO
LOGGING_ENABLE_FILE=false

# Cache
CACHE_ENABLE=true
CACHE_TTL=300
```

### Configuration Files

- **Development**: Uses `.env` file
- **Production**: Uses environment variables
- **Testing**: Uses in-memory database

## 🚀 Usage Examples

### Content Analysis

```python
import requests

# Analyze content
response = requests.post("http://localhost:8000/api/v1/analysis/analyze", json={
    "content": "This is a sample AI-generated content.",
    "model_version": "gpt-4",
    "metadata": {"source": "test"}
})

print(response.json())
```

### Model Comparison

```python
# Compare two entries
response = requests.post("http://localhost:8000/api/v1/comparison/compare", json={
    "entry1_id": "entry_1_id",
    "entry2_id": "entry_2_id",
    "comparison_type": "content_similarity"
})

print(response.json())
```

### Generate Report

```python
# Generate quality report
response = requests.post("http://localhost:8000/api/v1/reports/generate", json={
    "report_type": "comprehensive",
    "model_version": "gpt-4",
    "days": 30
})

print(response.json())
```

### Trend Analysis

```python
# Analyze trends
response = requests.post("http://localhost:8000/api/v1/trends/analyze", json={
    "model_version": "gpt-4",
    "metric": "quality_score",
    "days": 30
})

print(response.json())
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=refactored tests/

# Run specific test
pytest tests/test_analysis.py::test_analyze_content
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### System Summary
```bash
curl http://localhost:8000/api/v1/system/summary
```

### Job Statistics
```bash
curl http://localhost:8000/api/v1/system/jobs/statistics
```

## 🔒 Security

### Authentication
- API key authentication (optional)
- JWT token support (configurable)
- Role-based access control

### Rate Limiting
- Configurable requests per minute
- IP-based rate limiting
- Request tracking

### Security Headers
- X-Content-Type-Options
- X-Frame-Options
- X-XSS-Protection
- Strict-Transport-Security

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t ai-history-comparison .

# Run container
docker run -p 8000:8000 ai-history-comparison
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production Deployment

1. **Set environment variables**
2. **Use PostgreSQL database**
3. **Enable authentication**
4. **Configure logging**
5. **Set up monitoring**
6. **Use reverse proxy (nginx)**

## 📈 Performance

### Optimization Features
- Connection pooling
- Response caching
- Async processing
- Database indexing
- Query optimization

### Monitoring
- Request/response logging
- Performance metrics
- Error tracking
- Health checks

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

# Run linting
flake8 refactored/
black refactored/
mypy refactored/

# Run tests
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: See `/docs` endpoint
- **Issues**: Create GitHub issues
- **Discussions**: Use GitHub discussions
- **Email**: support@ai-history-comparison.com

## 🔄 Migration from Legacy System

The refactored system is designed to be a drop-in replacement for the legacy system:

1. **Database Migration**: Use the migration scripts
2. **API Compatibility**: Maintains backward compatibility
3. **Configuration**: Update environment variables
4. **Deployment**: Update deployment scripts

## 🎯 Roadmap

### Version 2.1
- [ ] WebSocket support for real-time updates
- [ ] Advanced caching with Redis
- [ ] GraphQL API
- [ ] Machine learning model integration

### Version 2.2
- [ ] Multi-tenant support
- [ ] Advanced analytics dashboard
- [ ] Export/import functionality
- [ ] Plugin system

### Version 3.0
- [ ] Microservices architecture
- [ ] Event-driven architecture
- [ ] Advanced AI integration
- [ ] Cloud-native deployment

---

**Built with ❤️ using Clean Architecture principles and modern Python practices.**




