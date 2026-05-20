# AI Document Processor - Refactored Architecture

## 🚀 Modern, Clean Architecture

A completely refactored AI document processing system with modern architecture, clean separation of concerns, and comprehensive error handling.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

## ✨ Features

### Core Features
- **Ultra-fast document processing** with parallel processing
- **Multiple document formats** support (PDF, Word, Markdown, HTML, XML)
- **AI-powered classification** and transformation
- **Streaming processing** for large files
- **Batch processing** capabilities
- **Real-time monitoring** and metrics

### Architecture Features
- **Clean Architecture** with separation of concerns
- **Dependency Injection** for better testability
- **Comprehensive Error Handling** with custom exceptions
- **Type Safety** with full type hints
- **Configuration Management** with validation
- **Async/Await** throughout the application
- **Modern FastAPI** with automatic documentation

### Performance Features
- **Memory optimization** with memory pools
- **Zero-copy operations** where possible
- **Intelligent caching** with multiple backends
- **Compression** for better performance
- **Connection pooling** for databases
- **Rate limiting** and security

## 🏗️ Architecture

### Directory Structure
```
src/
├── core/                 # Core business logic
│   ├── config.py        # Configuration management
│   ├── exceptions.py    # Custom exceptions
│   ├── processor.py     # Main processor
│   └── monitor.py       # Performance monitoring
├── models/              # Data models
│   ├── document.py      # Document models
│   ├── processing.py    # Processing models
│   └── errors.py        # Error models
├── services/            # Business services
│   ├── document_service.py
│   ├── ai_service.py
│   ├── transform_service.py
│   └── validation_service.py
├── api/                 # API layer
│   ├── app.py          # FastAPI application
│   ├── routes/         # API routes
│   ├── middleware/     # Custom middleware
│   └── dependencies.py # Dependency injection
└── utils/              # Utility functions
    ├── file_handler.py
    ├── text_extractor.py
    └── format_converter.py
```

### Design Patterns
- **Repository Pattern** for data access
- **Service Layer Pattern** for business logic
- **Dependency Injection** for loose coupling
- **Factory Pattern** for object creation
- **Observer Pattern** for event handling
- **Strategy Pattern** for processing algorithms

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or poetry
- Redis (optional, for caching)
- PostgreSQL (optional, for persistence)

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd ai-document-processor

# Install dependencies
pip install -r requirements.txt

# Run the application
python main_refactored.py
```

### Development Installation
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
flake8 src/
black src/
mypy src/
```

## ⚙️ Configuration

### Environment Variables
```bash
# Server Configuration
HOST=0.0.0.0
PORT=8001
WORKERS=1

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/dbname

# Cache Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_MAX_MEMORY_MB=1024

# AI Configuration
AI_API_KEY=your-api-key
AI_PROVIDER=openai
AI_MODEL=gpt-3.5-turbo

# Security Configuration
SECRET_KEY=your-secret-key
ENABLE_RATE_LIMITING=true
```

### Configuration File
```yaml
# config_refactored.yaml
server:
  host: "0.0.0.0"
  port: 8001
  workers: 1

database:
  url: "postgresql://user:password@localhost/dbname"
  pool_size: 10

cache:
  enabled: true
  backend: "redis"
  redis_url: "redis://localhost:6379/0"

ai:
  provider: "openai"
  api_key: "your-api-key"
  model: "gpt-3.5-turbo"
```

## 🚀 Usage

### Basic Usage
```python
from src.services.document_service import DocumentService
from src.models.document import Document
from src.models.processing import ProcessingConfig

# Create document service
service = DocumentService()

# Create document
document = await service.create_document("path/to/document.pdf")

# Process document
result = await service.process_document(
    document.id,
    ProcessingConfig(enable_ai_classification=True)
)

# Get results
print(f"Extracted text: {result.extracted_text}")
print(f"Classified type: {result.classified_type}")
```

### Batch Processing
```python
# Process multiple documents
document_ids = ["doc1", "doc2", "doc3"]
batch_result = await service.process_batch(document_ids)

print(f"Success rate: {batch_result.get_success_rate():.1f}%")
```

### API Usage
```bash
# Upload and process document
curl -X POST "http://localhost:8001/api/v1/documents" \
  -F "file=@document.pdf" \
  -F "filename=document.pdf"

# Get processing status
curl "http://localhost:8001/api/v1/processing/{result_id}"

# Get document
curl "http://localhost:8001/api/v1/documents/{document_id}"
```

## 📚 API Documentation

### Endpoints

#### Documents
- `POST /api/v1/documents` - Upload document
- `GET /api/v1/documents` - List documents
- `GET /api/v1/documents/{id}` - Get document
- `DELETE /api/v1/documents/{id}` - Delete document

#### Processing
- `POST /api/v1/processing` - Process document
- `POST /api/v1/processing/batch` - Batch process
- `GET /api/v1/processing/{id}` - Get processing result
- `GET /api/v1/processing/{id}/status` - Get processing status

#### Health & Metrics
- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - Performance metrics
- `GET /api/v1/health/detailed` - Detailed health check

### Interactive Documentation
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI Schema**: http://localhost:8001/openapi.json

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_document_service.py

# Run with verbose output
pytest -v
```

### Test Structure
```
tests/
├── unit/              # Unit tests
│   ├── test_models/
│   ├── test_services/
│   └── test_utils/
├── integration/       # Integration tests
│   ├── test_api/
│   └── test_processing/
└── fixtures/          # Test fixtures
    ├── documents/
    └── configs/
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config_refactored.yaml .

CMD ["python", "main_refactored.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/dbname
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: dbname
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password

  redis:
    image: redis:6-alpine
```

### Production Deployment
```bash
# Install production dependencies
pip install -r requirements-prod.txt

# Run with gunicorn
gunicorn src.api.app:app -w 4 -k uvicorn.workers.UvicornWorker

# Run with uvicorn
uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --workers 4
```

## 📊 Monitoring

### Health Checks
```bash
# Basic health check
curl http://localhost:8001/api/v1/health

# Detailed health check
curl http://localhost:8001/api/v1/health/detailed
```

### Metrics
```bash
# Get performance metrics
curl http://localhost:8001/api/v1/metrics

# Prometheus metrics
curl http://localhost:8001/metrics
```

### Logging
- **Structured logging** with JSON format
- **Request/response logging** with timing
- **Error tracking** with stack traces
- **Performance metrics** collection

## 🔧 Development

### Code Quality
```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/

# Security check
bandit -r src/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Adding New Features
1. Create feature branch
2. Add tests for new functionality
3. Implement feature with type hints
4. Update documentation
5. Run quality checks
6. Submit pull request

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add type hints to all functions
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Pydantic for data validation
- OpenAI for AI capabilities
- All contributors and users

## 📞 Support

- **Documentation**: [docs.example.com](https://docs.example.com)
- **Issues**: [GitHub Issues](https://github.com/example/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/discussions)
- **Email**: support@example.com

---

**Made with ❤️ by the AI Document Processor Team**



















