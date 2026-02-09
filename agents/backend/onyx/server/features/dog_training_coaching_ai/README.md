# Dog Training Coaching AI

AI-powered dog training coaching assistant using OpenRouter to provide personalized training advice and plans.

## Features

- 🤖 **AI Coaching**: Get expert advice on dog training questions
- 📋 **Training Plans**: Create personalized training plans based on breed, age, and goals
- 🎯 **Positive Reinforcement**: Focus on positive, science-based training methods
- 🔄 **Context-Aware**: Maintains conversation context for better advice
- 🛡️ **Rate Limiting**: Built-in rate limiting to protect API resources
- ✅ **Input Validation**: Comprehensive validation of all inputs
- 📊 **Health Monitoring**: Advanced health checks with OpenRouter status
- 🔒 **Error Handling**: Centralized error handling with structured logging

## Quick Start

### Installation

**Production (Recommended):**
```bash
pip install -r requirements.txt
```

**Minimal (Basic functionality only):**
```bash
pip install -r requirements-minimal.txt
```

**Development (With testing and code quality tools):**
```bash
pip install -r requirements-dev.txt
```

### Configuration

Set environment variables:

```bash
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional
OPENROUTER_MODEL=openai/gpt-4o
PORT=8030
```

### Run

**Direct:**
```bash
python main.py
```

**Using Make:**
```bash
make install
make run
```

**Using Docker:**
```bash
docker-compose up
```

The API will be available at `http://localhost:8030`

### Testing

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Lint and format
make lint
make format
```

## API Endpoints

### Health Check
```
GET /health
```

### Get Coaching Advice
```
POST /coach
```

Request body:
```json
{
  "question": "How do I teach my dog to sit?",
  "dog_breed": "Golden Retriever",
  "dog_age": "6 months",
  "dog_size": "large",
  "training_goal": "obedience",
  "experience_level": "beginner",
  "specific_issues": ["pulling on leash"]
}
```

### Create Training Plan
```
POST /training-plan
```

Request body:
```json
{
  "dog_breed": "German Shepherd",
  "dog_age": "1 year",
  "training_goals": ["obedience", "agility"],
  "time_available": "30 minutes per day",
  "experience_level": "intermediate"
}
```

### Analyze Behavior
```
POST /analyze-behavior
```

Request body:
```json
{
  "behavior_description": "My dog barks excessively when visitors arrive",
  "dog_breed": "Border Collie",
  "dog_age": "2 years",
  "frequency": "every time",
  "triggers": ["doorbell", "strangers"]
}
```

### Chat
```
POST /chat
```

Request body:
```json
{
  "message": "What's the best way to train a puppy?",
  "conversation_history": [],
  "dog_info": {"breed": "Labrador", "age": "3 months"}
}
```

### Track Training Progress
```
POST /training-progress
```

Request body:
```json
{
  "training_sessions": [...],
  "current_skills": ["sit", "stay"],
  "training_goals": ["heel", "come"],
  "time_period_days": 30
}
```

### Training Assessment
```
POST /training-assessment
```

Request body:
```json
{
  "dog_breed": "Golden Retriever",
  "dog_age": "1 year",
  "current_skills": ["sit", "stay", "down"],
  "training_goals": ["agility", "obedience"],
  "owner_experience": "intermediate"
}
```

### Get Training Resources
```
POST /training-resources
```

Request body:
```json
{
  "topic": "obedience",
  "level": "beginner",
  "format_preference": "video",
  "dog_breed": "German Shepherd"
}
```

### Analyze Training Trends
```
POST /training-trends
```

Request body:
```json
{
  "training_sessions": [...],
  "time_period_days": 30,
  "metrics_to_analyze": ["success_rate", "consistency"]
}
```

## OpenRouter Configuration

This service uses OpenRouter to access multiple AI models. Configure your API key in the environment:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

Or create a `.env` file (see `.env.example`):

```bash
cp .env.example .env
# Edit .env with your API key
```

Available models: https://openrouter.ai/models

## Examples

See `examples/example_usage.py` for usage examples.

## Quick Commands

```bash
# Start server
python main.py

# Or using scripts
./scripts/start.sh      # Linux/Mac
scripts/start.bat       # Windows

# Or using Docker
docker-compose up

# Or using Make
make run
```

## Development

### Project Structure
```
dog_training_coaching_ai/
├── api/routes/          # API endpoints
├── config/              # Configuration
├── core/                 # Core exceptions
├── infrastructure/       # External services (OpenRouter)
├── middleware/           # Request/response middleware
├── services/            # Business logic
├── utils/               # Utilities (logging, cache, validators)
├── tests/               # Test suite
└── examples/            # Usage examples
```

### Commands
- `make install` - Install dependencies
- `make run` - Run server
- `make test` - Run tests
- `make lint` - Check code quality
- `make format` - Format code
- `make clean` - Clean cache files

## Changelog

See `CHANGELOG.md` for version history.

## Architecture

### Core Components
- `main.py`: FastAPI application with lifespan events and middleware
- `config/app_config.py`: Configuration settings with environment variables
- `api/routes/training_routes.py`: API routes with dependency injection and rate limiting
- `services/coaching_service.py`: Core coaching logic using OpenRouter
- `schemas.py`: Pydantic models for request/response validation

### Infrastructure
- `infrastructure/openrouter/`: OpenRouter API client with retry logic
  - `api_client.py`: Low-level HTTP client
  - `openrouter_client.py`: High-level client abstraction

### Middleware
- `middleware/logging_middleware.py`: Structured request/response logging
- `middleware/error_middleware.py`: Centralized error handling

### Utilities
- `utils/logger.py`: Structured logging setup
- `utils/validators.py`: Input validation functions
- `utils/rate_limiter.py`: Rate limiting configuration

### Core
- `core/exceptions.py`: Custom exception classes

## Rate Limiting

Endpoints have different rate limits:
- `/coach`: 10 requests per minute
- `/training-plan`: 5 requests per minute
- `/analyze-behavior`: 10 requests per minute
- `/chat`: 20 requests per minute
- Other endpoints: 100 requests per hour (default)

## Validation

All inputs are validated:
- Dog breed: Minimum 2 characters
- Dog age: Valid format (puppy, young, adult, senior, or numeric)
- Training goals: Must be from valid list
- Experience level: Must be beginner, intermediate, or advanced

## Dependencies

### Core Libraries
- **FastAPI**: Modern async web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation and settings
- **httpx**: Async HTTP client for OpenRouter API

### Additional Features
- **structlog**: Structured logging with JSON output
- **slowapi**: Rate limiting (10/min for coaching, 5/min for plans)
- **redis**: Caching support (optional)
- **tenacity**: Retry logic with exponential backoff
- **prometheus-client**: Metrics and monitoring
- **pydantic**: Data validation and serialization

### Development Tools

#### Benchmarking
- `POST /api/v1/dev/benchmark` - Ejecutar benchmarks de funciones
- Utilidades: `Benchmark`, `benchmark_function()`, `benchmark_async_function()`

#### Profiling
- `GET /api/v1/dev/profiler/stats` - Obtener estadísticas de profiling
- Utilidades: `Profiler`, `profile_function()`

#### Configuration Management
- `GET /api/v1/dev/config` - Obtener configuración
- `POST /api/v1/dev/config/reload` - Recargar configuración
- Utilidades: `ConfigManager`, `get_config_manager()`

#### Advanced Logging
- `POST /api/v1/dev/logging/test` - Probar logging estructurado
- Utilidades: `StructuredLogger`, `PerformanceLogger`, `LogContext`, `log_performance()`

#### Testing Helpers
- `create_mock_service()` - Crear mocks de servicios
- `AsyncMock` - Mock para funciones async
- `create_test_client_config()` - Configuración de test client
- `mock_openrouter_response()` - Mock de respuestas OpenRouter

### Data Analysis

#### Statistical Analysis
- `POST /api/v1/analysis/describe` - Descripción estadística de datos
- `POST /api/v1/analysis/frequency` - Distribución de frecuencias
- `POST /api/v1/analysis/trends` - Análisis de tendencias temporales
- `POST /api/v1/analysis/anomalies` - Detección de anomalías
- `POST /api/v1/analysis/correlation` - Cálculo de correlación
- Utilidades: `DataAnalyzer`, `analyze_trends()`, `detect_anomalies()`, `calculate_correlation()`

### Optimization

#### Performance Optimization
- `Memoizer` - Cacheo de resultados de funciones
- `BatchProcessor` - Procesamiento optimizado de lotes
- `optimize_query_params()` - Optimización de parámetros
- `deduplicate_list()` - Eliminación de duplicados
- `LazyLoader` - Carga lazy de recursos pesados
- `debounce()` / `throttle()` - Control de frecuencia de llamadas

### Schema Validation

#### Advanced Validation
- `SchemaValidator` - Validador de esquemas flexible
- `validate_schema()` - Validación de datos contra esquema
- `create_schema()` - Crear esquemas desde definiciones
- `COMMON_SCHEMAS` - Esquemas predefinidos comunes

### Backup & Restore

#### Data Backup
- `POST /api/v1/backup/create` - Crear backup de datos
- `POST /api/v1/backup/restore` - Restaurar backup
- `GET /api/v1/backup/list` - Listar backups disponibles
- `POST /api/v1/backup/cleanup` - Limpiar backups antiguos
- Utilidades: `BackupManager` con soporte para JSON/Pickle y compresión

### Advanced Concurrency

#### Concurrency Control
- `SemaphorePool` - Pool de semáforos para control de concurrencia
- `TaskQueue` - Cola de tareas con prioridad
- `RateLimiter` - Rate limiter con ventana deslizante
- `run_with_timeout()` - Ejecutar con timeout
- `gather_with_limit()` - Ejecutar coroutines con límite de concurrencia
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Type checking
- **flake8**: Linting

## License

Part of Blatam Academy platform.

