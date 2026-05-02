# 🤖 GitHub Autonomous Agent

> Part of the [Blatam Academy Integrated Platform](../README.md)

**Intelligent autonomous agent that connects to GitHub repositories and executes instructions continuously from the frontend, working even with the computer off until the user stops it.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#️-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API](#-api)
- [Docker](#-docker)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)
- [Contribute](#-contribute)

## 🚀 Features

### Core Features
- ✅ **Universal Connection** — Connect to any GitHub repository (public or private)
- ✅ **Continuous Execution** — Process tasks autonomously and persistently
- ✅ **Remote Control** — Start/stop from the frontend in real-time
- ✅ **State Persistence** — Works even after system restarts
- ✅ **Real-Time Dashboard** — Complete monitoring of task status

### Advanced Features
- 🤖 **LLM Service with OpenRouter** — Access to multiple AI models (GPT-4, Claude, Gemini, etc.)
- ⚡ **Parallel Model Execution** — Run multiple models simultaneously to compare responses
- 📊 **A/B Testing Framework** — Compare models and prompts for continuous optimization
- 🔔 **Webhooks and Notifications** — Complete notification system for LLM events
- 📝 **Prompt Versioning** — Prompt versioning and management with rollback
- 🧪 **LLM Testing Framework** — Complete framework for automated model testing
- 🧠 **Semantic Caching** — Intelligent caching based on embeddings for similar responses
- 🚦 **Advanced Rate Limiting** — Sophisticated rate limiting with multiple strategies
- 📊 **Dashboard & Analytics** — Complete dashboard with detailed metrics and analytics
- 🔄 **Request Queue System** — Queue system with prioritization and timeout management
- ⚖️ **Load Balancer** — Intelligent load balancing with health checks and automatic failover
- 🔁 **Adaptive Retry System** — Intelligent retry system with adaptive backoff and error analysis
- 🎯 **Prompt Optimizer** — Automatic prompt optimization to improve clarity and efficiency
- 🛡️ **Model Fallback System** — Automatic fallback system when a model fails
- ⚡ **Performance Optimizer** — Automatic performance optimization with auto-tuning
- 🔄 **Robust Queue System** — Celery + Redis for asynchronous tasks
- 📊 **Logs and Monitoring** — Structured logging with Prometheus metrics
- 🔒 **Security** — JWT authentication, rate limiting, data validation
- 🐳 **Docker Ready** — Optimized containers for development and production
- 🔌 **RESTful API** — Complete API with automatic documentation (Swagger/OpenAPI)
- 🧪 **Full Testing** — Test suite with high coverage
- 📝 **Automatic Validation** — Configuration and dependency validation

### Technical Highlights
- **Modular Architecture** — Clean Architecture with separation of concerns
- **Dependency Injection** — DI system for better testability
- **Async/Await** — Asynchronous operations for better performance
- **Type Safety** — Full type hints with Pydantic
- **Error Handling** — Robust error handling with retry logic
- **Dependency Management** — Improved requirements.txt with updated versions and full documentation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React/Vue)                     │
│                  Control and Monitoring in Real-Time          │
└──────────────────────┬────────────────────────────────────────┘
                       │ HTTP/REST API
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Agent Routes │  │ GitHub Routes│  │ Task Routes │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└──────────────────────┬────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ GitHub Client│ │Task Processor│ │   Worker     │
│  (PyGithub)  │ │   (Core)     │ │  (Celery)    │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
        ┌──────────────────────────┐
        │   Storage & Queue        │
        │  ┌──────┐  ┌──────────┐ │
        │  │ SQLite│  │  Redis   │ │
        │  │ / PG  │  │ (Celery) │ │
        │  └──────┘  └──────────┘ │
        └──────────────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- Git
- Redis (for task queue)
- GitHub Token with appropriate permissions

### Option 1: Automatic Installation (Recommended)

#### Linux/macOS
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh --dev
```

#### Windows (PowerShell)
```powershell
.\scripts\setup.ps1 -Dev
```

### Option 2: Manual Installation

#### Local Development

```bash
# 1. Clone repository (if applicable)
git clone <repository-url>
cd github_autonomous_agent

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\Activate.ps1  # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your GitHub credentials

# 5. Validate configuration
python scripts/validate-env.py

# 6. Check dependencies
python scripts/check-dependencies.py

# 7. Start services (Redis, etc.)
./scripts/start-services.sh  # Linux/macOS

# 8. Run DB migrations (if applicable)
python scripts/migrate-db.py upgrade

# 9. Start application
python main.py
# Or with Make:
make run-dev
```

#### Production

```bash
# Install production dependencies
pip install -r requirements.txt
pip install -r requirements-prod.txt

# Configure environment variables
cp .env.example .env
# Edit .env with production configuration

# Start with uvicorn (recommended)
uvicorn main:app --host 0.0.0.0 --port 8030 --workers 4

# Or with gunicorn (alternative)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8030
```

### Option 3: Docker (Recommended for Production)

```bash
# Development
docker-compose up -d

# View logs
docker-compose logs -f app

# Production
docker build -t github-autonomous-agent:latest .
docker run -d -p 8030:8030 --env-file .env github-autonomous-agent:latest
```

📚 **See [QUICK_START.md](QUICK_START.md) for a 5-minute quick guide**  
📚 **See [REQUIREMENTS_GUIDE.md](REQUIREMENTS_GUIDE.md) for details on dependencies**  
📚 **See [DEVELOPMENT.md](DEVELOPMENT.md) for full development guide**

> 💡 **Note**: The `requirements.txt` file has been improved with updated versions, better documentation, and security guides. See the file for more details.

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

#### Mandatory

```bash
# GitHub API Token (requires permissions: repo, workflow, admin:repo_hook)
GITHUB_TOKEN=ghp_your_token_here

# Secret Key for JWT (generate a secure 32+ character key)
SECRET_KEY=your_super_secure_secret_key_here
```

#### Optional but Recommended

```bash
# Database (SQLite default, PostgreSQL for production)
DATABASE_URL=sqlite+aiosqlite:///./github_agent.db
# Or for PostgreSQL:
# DATABASE_URL=postgresql+asyncpg://user:password@localhost/github_agent

# Redis for task queue
REDIS_URL=redis://localhost:6379/0

# CORS (allowed origins, comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # json or text

# API
API_HOST=0.0.0.0
API_PORT=8030
```

#### Advanced

```bash
# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60

# Task Queue
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Monitoring
PROMETHEUS_ENABLED=false

# OpenRouter / LLM (Optional)
OPENROUTER_API_KEY=sk-or-v1-...
LLM_DEFAULT_MODELS=openai/gpt-4o-mini,anthropic/claude-3.5-sonnet,google/gemini-pro-1.5
LLM_ENABLED=true
LLM_TIMEOUT=60
LLM_MAX_PARALLEL_REQUESTS=10
```

### Validate Configuration

```bash
# Validate all environment variables
python scripts/validate-env.py

# Check installed dependencies
python scripts/check-dependencies.py

# Health check
python scripts/health-check.py
```

See `config/settings.py` for all available configuration options.

## 📖 Usage

### Basic Flow

1. **Connect Repository** — Connect your GitHub repository from the frontend
2. **Send Instructions** — Define tasks for the agent to execute
3. **Monitor Progress** — Watch the real-time dashboard
4. **Control Agent** — Start, pause, or stop the agent when needed

### API Usage Example

```bash
# 1. Connect repository
curl -X POST http://localhost:8030/api/v1/github/connect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "repo_owner": "user",
    "repo_name": "repository",
    "branch": "main"
  }'

# 2. Create task
curl -X POST http://localhost:8030/api/v1/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "instruction": "Analyze code and generate documentation",
    "priority": "high"
  }'

# 3. View task status
curl http://localhost:8030/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN"

# 4. Stop agent
curl -X POST http://localhost:8030/api/v1/agent/stop \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Interactive API Documentation

Once the server is started, visit:
- **Swagger UI**: http://localhost:8030/docs
- **ReDoc**: http://localhost:8030/redoc
- **OpenAPI JSON**: http://localhost:8030/openapi.json

## 🐳 Docker

### Development

```bash
# Start all services (app, redis, etc.)
docker-compose up -d

# View logs
docker-compose logs -f app

# Rebuild after changes
docker-compose up -d --build

# Stop services
docker-compose down
```

### Production

```bash
# Build image
docker build -t github-autonomous-agent:latest .

# Run container
docker run -d \
  --name github-agent \
  -p 8030:8030 \
  --env-file .env \
  --restart unless-stopped \
  github-autonomous-agent:latest

# View logs
docker logs -f github-agent
```

## 🧪 Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html --cov-report=term
```

## 🔍 Troubleshooting

> 📖 **For a complete troubleshooting guide, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

### Common Issues

1. **GitHub Connection Error**: Check token and permissions.
2. **Redis Unavailable**: Ensure Redis is running.
3. **Missing Dependencies**: Check `requirements.txt`.
4. **Database Error**: Check connection string.
5. **Port in Use**: Change port in `.env`.

## 📚 Documentation

### Main Documentation

- **[QUICK_START.md](QUICK_START.md)** — Quick Start in 5 Minutes ⚡
- **[LLM_SERVICE.md](LLM_SERVICE.md)** — LLM Service with OpenRouter 🤖
- **[LLM_ARCHITECTURE.md](LLM_ARCHITECTURE.md)** — Modular LLM Architecture 🏗️
- **[API_GUIDE.md](API_GUIDE.md)** — REST API Guide 📡
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — System Architecture 🏗️
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — Troubleshooting Guide 🔧
- **[DEVELOPMENT.md](DEVELOPMENT.md)** — Development Guide 💻
- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Deployment Guide 🚀
- **[REQUIREMENTS_GUIDE.md](REQUIREMENTS_GUIDE.md)** — Dependencies Guide 📦
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution Guide 🤝
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** — Complete Index 📚

## 🤝 Contribute

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

[← Back to Main README](../README.md)
