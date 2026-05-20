# Export IA — AI-Powered Document Export System

> Part of the [Blatam Academy Integrated Platform](../README.md)

AI-powered document export system — modular and functional architecture.

## 🚀 Features

- ✅ **Multi-Format Export** — PDF, DOCX, HTML, Markdown, and more
- ✅ **Full REST API** — Built with FastAPI
- ✅ **Async Task Management** — Asynchronous job processing
- ✅ **Quality Validation** — Content validation and quality improvements
- ✅ **Modular Architecture** — Scalable and maintainable
- ✅ **Flexible Configuration** — Environment-based configuration
- ✅ **Complete Documentation** — Interactive API docs included

## 📁 Project Structure

```
export_ia/
├── 📁 app/                          # Main application
│   ├── 📁 core/                     # Business logic
│   ├── 📁 api/                      # REST API
│   ├── 📁 exporters/                # Exporters
│   ├── 📁 services/                 # Services
│   └── 📁 utils/                    # Utilities
├── 📁 config/                       # Configuration
├── 📁 database/                     # Database
├── 📁 tests/                        # Tests
├── 📁 docs/                         # Documentation
├── 📁 scripts/                      # Scripts
├── 📁 docker/                       # Docker
├── 📁 examples/                     # Examples
├── 📄 requirements.txt              # Dependencies
└── 📄 README.md                     # This file
```

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/export-ia.git
cd export-ia
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
# Main dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your settings
```

## 🚀 Quick Start

### Run the API
```bash
# Development
python -m app.api.main

# Production
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### With Docker
```bash
docker-compose -f docker/docker-compose.yml up --build
curl http://localhost:8000/health
```

## 📚 API Endpoints

### Main Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/` | System information |
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/export` | Export document |
| `GET` | `/api/v1/export/{id}/status` | Task status |
| `GET` | `/api/v1/export/{id}/download` | Download file |
| `POST` | `/api/v1/validate` | Validate content |
| `GET` | `/api/v1/formats` | Supported formats |
| `GET` | `/api/v1/templates/{type}` | Templates |

### Usage Example

```bash
# Export document
curl -X POST "http://localhost:8000/api/v1/export" \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "title": "My Document",
      "sections": [
        {"heading": "Introduction", "content": "Content here..."}
      ]
    },
    "format": "pdf",
    "document_type": "report",
    "quality_level": "professional"
  }'

# Check status
curl "http://localhost:8000/api/v1/export/{task_id}/status"

# Download file
curl "http://localhost:8000/api/v1/export/{task_id}/download" -o document.pdf
```

## 🐍 Python SDK

### Basic Usage
```python
from app.core.engine import get_export_engine
from app.core.models import ExportConfig, ExportFormat, DocumentType

# Get engine
engine = get_export_engine()
await engine.initialize()

# Configure export
config = ExportConfig(
    format=ExportConfig.PDF,
    document_type=DocumentType.REPORT
)

# Export document
content = {
    "title": "My Document",
    "sections": [
        {"heading": "Introduction", "content": "Content here..."}
    ]
}

task_id = await engine.export_document(content, config)

# Wait for completion
result = await engine.wait_for_completion(task_id)
print(f"Exported file: {result['file_path']}")
```

## 🧪 Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=app

# Specific tests
pytest tests/unit/
pytest tests/integration/
```

## 📖 Documentation

### Available Docs
- **[API Documentation](docs/API.md)** — Complete API documentation
- **[Deployment Guide](docs/DEPLOYMENT.md)** — Deployment guide
- **[Development Guide](docs/DEVELOPMENT.md)** — Development guide
- **[Examples](examples/)** — Usage examples

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Configuration

### Environment Variables
```bash
# API
API_TITLE="Export IA API"
API_VERSION="2.0.0"
DEBUG=false

# Database
DATABASE_URL="sqlite:///./export_ia.db"

# Files
EXPORTS_DIR="./exports"
MAX_FILE_SIZE=52428800  # 50MB

# Logging
LOG_LEVEL="INFO"
LOG_FILE="./logs/export_ia.log"
```

## 🐳 Docker

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./export_ia.db
    volumes:
      - ./exports:/app/exports
    restart: unless-stopped
```

### Docker Commands
```bash
docker build -t export-ia .
docker run -p 8000:8000 export-ia
docker-compose up --build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes and commit (`git commit -m "feat: add new feature"`)
4. Push and create a Pull Request

## 📄 License

This project is under the MIT License. See [LICENSE](LICENSE) for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/export-ia/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/export-ia/discussions)

## 🎯 Roadmap

- [ ] **v2.1.0** — Performance improvements
- [ ] **v2.2.0** — New export formats
- [ ] **v2.3.0** — Advanced AI integration
- [ ] **v3.0.0** — Microservices architecture

---

**Export IA — Simple, fast, and professional document export!** 🚀

[← Back to Main README](../README.md)