# Document Workflow Chain v2.0 - Optimized & Refactored

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, modular document workflow chain system with advanced AI integration, real-time monitoring, and enterprise-grade features.

## 🚀 Key Features

### ⚡ Performance Optimized
- **Async/await throughout** for maximum concurrency
- **Advanced caching** with TTL and LRU eviction
- **Memory optimization** with weak references
- **Connection pooling** and request optimization
- **Compression** and serialization optimization

### 🏗️ Clean Architecture
- **Modular design** with clear separation of concerns
- **Dependency injection** for testability
- **Plugin system** for extensibility
- **Event-driven architecture** with WebSocket support
- **Type safety** with comprehensive type hints

### 🤖 Advanced AI Integration
- **Multiple AI providers** (OpenAI, Anthropic, Google, Azure)
- **Intelligent caching** of AI responses
- **Rate limiting** and error handling
- **Content analysis** and quality scoring
- **Multi-language support**

### 📊 Real-time Monitoring
- **Performance metrics** and analytics
- **WebSocket connections** for real-time updates
- **Health checks** and system monitoring
- **Error tracking** and logging
- **Usage analytics** and insights

### 🔒 Enterprise Security
- **JWT authentication** and authorization
- **Rate limiting** and DDoS protection
- **CORS configuration** and security headers
- **Input validation** and sanitization
- **Audit logging** and compliance

## 📁 Architecture

```
Document Workflow Chain v2.0/
├── workflow_chain_v2.py      # Core workflow engine
├── api_v2.py                 # FastAPI application
├── config_v2.py              # Configuration management
├── start_v2.py               # Optimized startup script
├── requirements_v2.txt       # Optimized dependencies
├── README_v2.md              # This file
├── modules/                  # Modular components
│   ├── core/                 # Core functionality
│   ├── workflow/             # Workflow management
│   ├── ai/                   # AI integration
│   ├── analytics/            # Analytics and monitoring
│   ├── api/                  # API components
│   └── content/              # Content processing
├── examples/                 # Usage examples
├── tests/                    # Test suite
├── docs/                     # Documentation
└── docker/                   # Containerization
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Redis (optional, for caching)
- PostgreSQL/MySQL (optional, for production)

### Quick Start

1. **Clone and setup**:
```bash
git clone <repository>
cd document-workflow-chain
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements_v2.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start the application**:
```bash
python start_v2.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t workflow-chain-v2 .
docker run -p 8000:8000 workflow-chain-v2
```

## 🔧 Configuration

### Environment Variables

```bash
# Application
APP_NAME="Document Workflow Chain v2.0"
APP_VERSION="2.0.0"
ENVIRONMENT="production"
DEBUG=false

# Server
HOST="0.0.0.0"
PORT=8000
WORKERS=4

# AI Configuration
AI_CLIENT_TYPE="openai"
AI_API_KEY="your-api-key"
AI_MODEL="gpt-4"
AI_MAX_TOKENS=4000
AI_TEMPERATURE=0.7

# Database
DATABASE_TYPE="postgresql"
DATABASE_URL="postgresql://user:pass@localhost/db"
DATABASE_POOL_SIZE=10

# Cache
CACHE_TYPE="redis"
CACHE_REDIS_URL="redis://localhost:6379/0"
CACHE_MAX_SIZE=10000

# Security
SECRET_KEY="your-secret-key"
CORS_ORIGINS="https://yourdomain.com"
RATE_LIMIT_PER_MINUTE=100

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
ENABLE_COMPRESSION=true
ENABLE_CACHING=true
```

### Configuration Files

Support for JSON, YAML, and TOML configuration files:

```yaml
# config.yaml
app:
  name: "Document Workflow Chain v2.0"
  version: "2.0.0"
  environment: "production"

ai:
  client_type: "openai"
  api_key: "your-api-key"
  model: "gpt-4"
  max_tokens: 4000

database:
  type: "postgresql"
  url: "postgresql://user:pass@localhost/db"
  pool_size: 10

cache:
  type: "redis"
  redis_url: "redis://localhost:6379/0"
  max_size: 10000
```

## 📚 API Documentation

### Core Endpoints

- **GET /** - API information and status
- **GET /health** - Health check and system status
- **GET /docs** - Interactive API documentation
- **GET /redoc** - Alternative API documentation

### Workflow Management

- **POST /api/v2/workflows** - Create workflow
- **GET /api/v2/workflows** - List workflows
- **GET /api/v2/workflows/{id}** - Get workflow
- **DELETE /api/v2/workflows/{id}** - Delete workflow
- **POST /api/v2/workflows/{id}/optimize** - Optimize workflow

### Node Management

- **POST /api/v2/nodes** - Add node
- **GET /api/v2/workflows/{id}/nodes** - Get workflow nodes
- **GET /api/v2/nodes/{id}** - Get node
- **PUT /api/v2/nodes/{id}** - Update node
- **DELETE /api/v2/nodes/{id}** - Delete node

### Real-time Features

- **WebSocket /ws/{client_id}** - Real-time updates
- **GET /api/v2/statistics** - System statistics
- **GET /metrics** - Prometheus metrics

### Example Usage

```python
import httpx
import asyncio

async def create_workflow():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v2/workflows",
            json={
                "name": "AI Blog Series",
                "description": "A series of AI-related blog posts"
            }
        )
        return response.json()

async def add_node(chain_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v2/nodes",
            json={
                "chain_id": chain_id,
                "title": "Introduction to AI",
                "content": "Artificial Intelligence is revolutionizing...",
                "prompt": "Write an introduction to AI"
            }
        )
        return response.json()

# Run the example
asyncio.run(create_workflow())
```

## 🔌 Plugin System

### Creating Plugins

```python
from workflow_chain_v2 import WorkflowChain

class CustomPlugin:
    async def after_node_add(self, node, chain):
        # Custom logic after node is added
        print(f"Node {node.id} added to chain {chain.id}")
    
    async def optimize_chain(self, chain):
        # Custom optimization logic
        return {"optimized": True}

# Register plugin
chain = WorkflowChain("My Chain")
chain.add_plugin("custom", CustomPlugin())
```

### Available Hooks

- `after_node_add` - Called after a node is added
- `before_node_add` - Called before a node is added
- `optimize_chain` - Called during chain optimization
- `chain_created` - Called when a chain is created
- `chain_deleted` - Called when a chain is deleted

## 📊 Monitoring & Analytics

### Performance Metrics

- Request/response times
- Memory and CPU usage
- Cache hit rates
- Error rates and types
- AI API usage and costs

### Real-time Monitoring

```python
# WebSocket connection for real-time updates
import websockets
import json

async def monitor_updates():
    async with websockets.connect("ws://localhost:8000/ws/monitor") as websocket:
        async for message in websocket:
            data = json.loads(message)
            print(f"Update: {data['type']} - {data['data']}")
```

### Health Checks

```bash
# Check system health
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/api/v2/statistics

# Get metrics
curl http://localhost:8000/metrics
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=workflow_chain_v2 --cov=api_v2

# Run specific test
pytest tests/test_workflow.py

# Run with verbose output
pytest -v
```

### Test Configuration

```python
# tests/conftest.py
import pytest
from workflow_chain_v2 import WorkflowChainManager

@pytest.fixture
async def workflow_manager():
    manager = WorkflowChainManager()
    yield manager
    # Cleanup after tests
```

## 🚀 Performance Optimization

### Caching Strategy

- **L1 Cache**: In-memory with LRU eviction
- **L2 Cache**: Redis for distributed caching
- **AI Response Cache**: Intelligent caching of AI responses
- **Database Query Cache**: Optimized database queries

### Memory Management

- **Weak references** for large objects
- **Connection pooling** for databases
- **Streaming responses** for large data
- **Garbage collection** optimization

### Concurrency

- **Async/await** throughout the application
- **Connection pooling** for external services
- **Rate limiting** to prevent overload
- **Circuit breakers** for fault tolerance

## 🔒 Security

### Authentication & Authorization

- JWT token-based authentication
- Role-based access control (RBAC)
- API key management
- Session management

### Input Validation

- Pydantic models for request validation
- SQL injection prevention
- XSS protection
- CSRF protection

### Rate Limiting

- Per-user rate limiting
- Per-IP rate limiting
- API endpoint rate limiting
- DDoS protection

## 📈 Scaling

### Horizontal Scaling

- Stateless application design
- Load balancer support
- Database connection pooling
- Redis clustering

### Vertical Scaling

- Multi-process support
- Thread pool optimization
- Memory optimization
- CPU optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_v2.txt
pip install -r requirements-dev.txt

# Run code formatting
black .
isort .

# Run linting
flake8 .
mypy .

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@yourdomain.com

## 🙏 Acknowledgments

- FastAPI team for the excellent framework
- OpenAI for AI capabilities
- The open-source community for various libraries

---

**Document Workflow Chain v2.0** - High-performance, enterprise-ready document workflow system with AI integration.




