# Document Workflow Chain v3.0 - Clean Architecture & DDD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20Architecture%20%2B%20DDD-purple.svg)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A highly advanced, enterprise-grade document workflow chain system built with **Clean Architecture** and **Domain-Driven Design** principles. This version represents a complete architectural refactoring with professional-grade patterns and practices.

## 🏗️ Architecture Overview

### Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   FastAPI API   │  │   WebSocket     │  │  Middleware │ │
│  │   Controllers   │  │   Handlers      │  │   Stack     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Use Cases     │  │   DTOs          │  │  Event      │ │
│  │   (Business     │  │   (Data         │  │  Handlers   │ │
│  │    Logic)       │  │    Transfer     │  │             │ │
│  │                 │  │    Objects)     │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     Domain Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Entities      │  │   Value Objects │  │  Domain     │ │
│  │   (Business     │  │   (Immutable    │  │  Services   │ │
│  │    Objects)     │  │    Data)        │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Repositories  │  │   Events        │  │  Exceptions │ │
│  │   (Interfaces)  │  │   (Domain       │  │  (Business  │ │
│  │                 │  │    Events)      │  │    Rules)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Database      │  │   External      │  │   Shared    │ │
│  │   (SQLAlchemy)  │  │   Services      │  │   Services  │ │
│  │                 │  │   (AI, Cache)   │  │   (DI, etc) │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 🏛️ Clean Architecture
- **Separation of Concerns**: Clear boundaries between layers
- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Testability**: Each layer can be tested independently
- **Maintainability**: Changes in one layer don't affect others

### 🎯 Domain-Driven Design
- **Rich Domain Models**: Business logic encapsulated in entities
- **Value Objects**: Immutable objects for type safety
- **Domain Events**: Event-driven architecture for loose coupling
- **Aggregates**: Consistency boundaries for business operations
- **Domain Services**: Business logic that doesn't belong to entities

### 🔧 Advanced Patterns
- **Dependency Injection**: IoC container with lifetime management
- **Repository Pattern**: Data access abstraction
- **CQRS**: Command Query Responsibility Segregation
- **Event Sourcing**: Event-driven state management
- **Factory Pattern**: Object creation abstraction

### ⚡ Performance & Scalability
- **Async/Await**: Non-blocking I/O throughout
- **Connection Pooling**: Efficient database connections
- **Caching**: Multi-level caching strategy
- **Rate Limiting**: Token bucket and sliding window algorithms
- **WebSocket**: Real-time bidirectional communication

### 🔒 Enterprise Security
- **JWT Authentication**: Stateless authentication
- **Authorization**: Role-based access control
- **Input Validation**: Pydantic models for request validation
- **Rate Limiting**: DDoS protection
- **CORS**: Cross-origin resource sharing

### 📊 Monitoring & Observability
- **Structured Logging**: JSON-formatted logs
- **Metrics**: Prometheus-compatible metrics
- **Health Checks**: Comprehensive health monitoring
- **Tracing**: Request tracing across layers
- **Error Handling**: Centralized error management

## 📁 Project Structure

```
src/
├── domain/                          # Domain Layer
│   ├── entities/                    # Domain Entities
│   │   ├── workflow_chain.py        # WorkflowChain Aggregate Root
│   │   └── workflow_node.py         # WorkflowNode Entity
│   ├── value_objects/               # Value Objects
│   │   ├── workflow_id.py           # WorkflowId Value Object
│   │   ├── node_id.py               # NodeId Value Object
│   │   ├── workflow_status.py       # WorkflowStatus Enum
│   │   └── priority.py              # Priority Enum
│   ├── repositories/                # Repository Interfaces
│   │   └── workflow_repository.py   # WorkflowRepository Interface
│   ├── services/                    # Domain Services
│   │   └── workflow_domain_service.py
│   ├── events/                      # Domain Events
│   │   ├── workflow_events.py       # Workflow Domain Events
│   │   └── node_events.py           # Node Domain Events
│   └── exceptions/                  # Domain Exceptions
│       ├── workflow_exceptions.py   # Workflow Domain Exceptions
│       └── node_exceptions.py       # Node Domain Exceptions
├── application/                     # Application Layer
│   ├── use_cases/                   # Use Cases (Business Logic)
│   │   ├── create_workflow_use_case.py
│   │   ├── add_node_use_case.py
│   │   ├── get_workflow_use_case.py
│   │   └── list_workflows_use_case.py
│   ├── dto/                         # Data Transfer Objects
│   │   ├── workflow_dto.py
│   │   └── node_dto.py
│   └── event_handlers/              # Event Handlers
│       ├── workflow_event_handlers.py
│       └── node_event_handlers.py
├── infrastructure/                  # Infrastructure Layer
│   ├── persistence/                 # Data Persistence
│   │   ├── sqlalchemy_workflow_repository.py
│   │   ├── models.py                # SQLAlchemy Models
│   │   └── database.py              # Database Configuration
│   └── external/                    # External Services
│       ├── ai_service.py            # AI Service Integration
│       └── cache_service.py         # Cache Service
├── presentation/                    # Presentation Layer
│   ├── api/                         # REST API
│   │   └── workflow_controller.py   # FastAPI Controllers
│   └── websocket/                   # WebSocket Handlers
│       └── workflow_websocket.py    # WebSocket Manager
├── shared/                          # Shared Services
│   ├── container.py                 # Dependency Injection Container
│   ├── events/                      # Event Bus
│   │   └── event_bus.py             # Event Bus Implementation
│   ├── middleware/                  # Middleware Components
│   │   ├── rate_limiter.py          # Rate Limiting
│   │   ├── cache.py                 # Caching
│   │   ├── auth.py                  # Authentication
│   │   ├── logging.py               # Logging
│   │   └── monitoring.py            # Monitoring
│   ├── exceptions/                  # Application Exceptions
│   └── utils/                       # Utility Functions
└── main.py                          # Application Entry Point
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL 12+ (or SQLite for development)
- Redis 6+ (optional, for caching)
- Docker & Docker Compose (optional)

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
pip install -r requirements_v3.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run database migrations**:
```bash
alembic upgrade head
```

5. **Start the application**:
```bash
python -m src.main
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.v3.yml up -d

# Or build manually
docker build -f Dockerfile.v3 -t workflow-chain-v3 .
docker run -p 8000:8000 workflow-chain-v3
```

## 🔧 Configuration

### Environment Variables

```bash
# Application
APP_NAME="Document Workflow Chain v3.0"
APP_VERSION="3.0.0"
ENVIRONMENT="production"
DEBUG=false

# Database
DATABASE_URL="postgresql://user:pass@localhost/db"
DATABASE_POOL_SIZE=10
DATABASE_ECHO=false

# Redis (Optional)
REDIS_URL="redis://localhost:6379/0"
CACHE_TTL=300

# Security
SECRET_KEY="your-secret-key"
JWT_ALGORITHM="HS256"
JWT_EXPIRE_MINUTES=30

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000

# AI Services
AI_PROVIDER="openai"
AI_API_KEY="your-api-key"
AI_MODEL="gpt-4"
AI_MAX_TOKENS=4000
```

## 📚 API Documentation

### REST API Endpoints

- **GET /** - Application information
- **GET /health** - Health check
- **GET /metrics** - Application metrics
- **GET /docs** - Interactive API documentation

### Workflow Management

- **POST /api/v3/workflows** - Create workflow
- **GET /api/v3/workflows** - List workflows (with pagination)
- **GET /api/v3/workflows/{id}** - Get workflow details
- **PUT /api/v3/workflows/{id}** - Update workflow
- **DELETE /api/v3/workflows/{id}** - Delete workflow

### Node Management

- **POST /api/v3/workflows/{id}/nodes** - Add node to workflow
- **GET /api/v3/workflows/{id}/nodes** - Get workflow nodes
- **PUT /api/v3/nodes/{id}** - Update node
- **DELETE /api/v3/nodes/{id}** - Delete node

### WebSocket Endpoints

- **WebSocket /ws/{connection_id}** - Real-time communication

### Example Usage

```python
import httpx
import asyncio

async def create_workflow():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v3/workflows",
            json={
                "name": "AI Blog Series",
                "description": "A series of AI-related blog posts",
                "settings": {
                    "max_nodes": 100,
                    "timeout": 300
                }
            },
            headers={"Authorization": "Bearer your-jwt-token"}
        )
        return response.json()

async def add_node(workflow_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8000/api/v3/workflows/{workflow_id}/nodes",
            json={
                "title": "Introduction to AI",
                "content": "Artificial Intelligence is revolutionizing...",
                "prompt": "Write an introduction to AI",
                "priority": 2,
                "tags": ["ai", "introduction"]
            },
            headers={"Authorization": "Bearer your-jwt-token"}
        )
        return response.json()

# Run the example
asyncio.run(create_workflow())
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                           # Unit Tests
│   ├── domain/                     # Domain Layer Tests
│   ├── application/                # Application Layer Tests
│   └── infrastructure/             # Infrastructure Layer Tests
├── integration/                    # Integration Tests
│   ├── api/                        # API Integration Tests
│   └── database/                   # Database Integration Tests
├── e2e/                           # End-to-End Tests
└── fixtures/                      # Test Fixtures
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with verbose output
pytest -v

# Run with parallel execution
pytest -n auto
```

## 📊 Monitoring & Observability

### Health Checks

```bash
# Check application health
curl http://localhost:8000/health

# Get detailed metrics
curl http://localhost:8000/metrics
```

### Logging

The application uses structured logging with JSON format:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "workflow_controller",
  "message": "Created workflow 123e4567-e89b-12d3-a456-426614174000",
  "workflow_id": "123e4567-e89b-12d3-a456-426614174000",
  "user_id": "user123",
  "duration_ms": 150
}
```

### Metrics

Prometheus-compatible metrics are available at `/metrics`:

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `workflow_operations_total` - Workflow operations
- `websocket_connections_active` - Active WebSocket connections
- `event_bus_events_published_total` - Published events

## 🔒 Security

### Authentication & Authorization

- **JWT Tokens**: Stateless authentication
- **Role-Based Access**: Granular permissions
- **API Keys**: Service-to-service authentication
- **Rate Limiting**: DDoS protection

### Input Validation

- **Pydantic Models**: Request/response validation
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Input sanitization
- **CSRF Protection**: Token-based protection

## 🚀 Performance Optimization

### Caching Strategy

- **L1 Cache**: In-memory with LRU eviction
- **L2 Cache**: Redis for distributed caching
- **Query Cache**: Database query optimization
- **Response Cache**: API response caching

### Database Optimization

- **Connection Pooling**: Efficient connection management
- **Query Optimization**: Indexed queries
- **Batch Operations**: Bulk operations
- **Read Replicas**: Read scaling

### Async Operations

- **Non-blocking I/O**: Async/await throughout
- **Background Tasks**: Fire-and-forget operations
- **Event Processing**: Asynchronous event handling
- **WebSocket**: Real-time communication

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Code Standards

- **Type Hints**: Comprehensive type annotations
- **Documentation**: Docstrings for all public methods
- **Testing**: Minimum 80% code coverage
- **Linting**: Black, isort, flake8, mypy
- **Architecture**: Follow Clean Architecture principles

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/
mypy src/

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Uncle Bob** for Clean Architecture principles
- **Eric Evans** for Domain-Driven Design
- **FastAPI** team for the excellent framework
- **SQLAlchemy** team for the ORM
- The open-source community for various libraries

---

**Document Workflow Chain v3.0** - Enterprise-grade document workflow system with Clean Architecture and Domain-Driven Design.




