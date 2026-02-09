# 👔 Character Clothing Changer AI

AI-powered tool for changing clothing in character images using Flux2 and DeepSeek models.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- HuggingFace token (for Flux2 model)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (optional, for Flux2)
export HUGGINGFACE_TOKEN=your_token_here

# Set DeepSeek API key (optional, for fallback)
export DEEPSEEK_API_KEY=your_api_key_here
```

### Running the Server

```bash
# Using Python
python run_server.py

# Or using the start script
./start.sh  # Linux/Mac
start.bat   # Windows
```

The server will start at `http://localhost:8002`

### Using the Web Interface

Open `http://localhost:8002` in your browser to access the web interface.

## 📖 Features

- **AI-Powered Clothing Change**: Change clothing in character images using AI
- **Dual Model Support**: Flux2 (primary) and DeepSeek (fallback)
- **ComfyUI Integration**: Generate safe tensors for ComfyUI workflows
- **Quality Metrics**: Assess result quality automatically
- **Batch Processing**: Process multiple images
- **Web Interface**: User-friendly web interface
- **API Access**: RESTful API for programmatic access

## 🏗️ Architecture

The project uses a modular architecture with advanced core systems:

```
character_clothing_changer_ai/
├── api/              # API layer (FastAPI)
├── core/             # Core business logic
│   ├── workflow.py              # Workflow system
│   ├── pipeline.py              # Pipeline system
│   ├── orchestrator.py          # Orchestrator system
│   ├── state_manager.py         # State management
│   ├── advanced_cache.py        # Advanced caching
│   ├── service_base.py          # Service base classes
│   ├── coordinator.py           # Component coordinator
│   ├── integration.py           # Integration system
│   ├── data_pipeline.py         # Data transformation pipeline
│   ├── serializer.py            # Serialization system
│   ├── structured_logging.py    # Structured logging
│   ├── config_builder.py        # Configuration builder
│   ├── scheduler.py             # Task scheduler
│   ├── advanced_queue.py        # Advanced queue
│   ├── batch_operations.py      # Batch operations
│   ├── handler_base.py         # Handler base classes
│   ├── processor_base.py        # Processor base classes
│   ├── result_aggregator.py     # Result aggregator
│   ├── performance_tuner.py     # Performance tuner
│   ├── resource_manager.py      # Resource manager
│   ├── rate_limiter.py          # Rate limiter
│   ├── circuit_breaker.py       # Circuit breaker
│   ├── event_bus.py             # Event bus
│   ├── telemetry.py             # Telemetry system
│   ├── health_check.py          # Health checks
│   ├── retry_manager.py         # Retry manager
│   ├── dependency_injection.py  # Dependency injection
│   ├── lifecycle.py             # Lifecycle management
│   ├── validation_manager.py    # Validation manager
│   ├── metrics_collector.py     # Metrics collector
│   ├── error_handler.py          # Error handler
│   ├── security.py               # Security manager
│   ├── middleware_base.py       # Middleware base
│   ├── observability.py          # Observability manager
│   ├── factory_base.py          # Factory base
│   ├── storage_base.py          # Storage base
│   ├── execution_context.py     # Execution context
│   ├── base_models.py           # Base models
│   ├── types.py                 # Type definitions
│   ├── interfaces.py            # Abstract interfaces
│   ├── constants.py             # Application constants
│   ├── helpers.py               # Common helpers
│   ├── async_utils.py           # Async utilities
│   ├── repository_base.py       # Repository base
│   ├── manager_base.py          # Manager base
│   ├── component_registry.py    # Component registry
│   ├── decorators.py            # Common decorators
│   ├── context_managers.py      # Context managers
│   ├── tracing.py               # Distributed tracing
│   ├── feature_flags.py         # Feature flags
│   ├── audit.py                 # Audit system
│   ├── backup.py                # Backup system
│   ├── migrations.py            # Migrations system
│   ├── api_versioning.py        # API versioning
│   ├── testing.py               # Testing utilities
│   ├── notifications.py         # Notification system
│   ├── webhooks.py              # Webhook manager
│   ├── alerting.py              # Alerting system
│   ├── reporting.py             # Reporting system
│   ├── analytics.py             # Analytics system
│   ├── monitoring_dashboard.py  # Monitoring dashboard
│   ├── plugin_system.py         # Plugin system
│   ├── optimizer.py             # Performance optimizer
│   ├── benchmark.py             # Benchmark system
│   ├── task_manager.py          # Task manager
│   ├── parallel_executor.py     # Parallel executor
│   └── executor_base.py         # Executor base
├── config/           # Configuration
├── models/           # AI models
├── static/           # Frontend assets
└── docs/             # Documentation
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.
See [ADVANCED_CORE_SYSTEMS.md](docs/refactoring/ADVANCED_CORE_SYSTEMS.md) for advanced core systems documentation.

## 📡 API Documentation

### Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/model/info` - Model information
- `POST /api/v1/initialize` - Initialize model
- `POST /api/v1/change-clothing` - Change clothing in image
- `GET /api/v1/tensors` - List saved tensors
- `GET /api/v1/tensor/{id}` - Download tensor

See [API.md](docs/API.md) for complete API documentation.

## 🔧 Configuration

Configuration is managed via environment variables:

```bash
# Model Configuration
CLOTHING_CHANGER_MODEL_ID=black-forest-labs/flux2-dev
CLOTHING_CHANGER_DEVICE=cuda  # or cpu, auto
CLOTHING_CHANGER_DTYPE=float16  # or float32

# API Configuration
CLOTHING_CHANGER_API_HOST=0.0.0.0
CLOTHING_CHANGER_API_PORT=8002

# Output Configuration
CLOTHING_CHANGER_OUTPUT_DIR=./comfyui_tensors

# DeepSeek Configuration
DEEPSEEK_API_KEY=your_api_key_here

# HuggingFace Configuration
HUGGINGFACE_TOKEN=your_token_here
```

## 🎨 Usage Examples

### Python API

```python
from character_clothing_changer_ai import ClothingChangerService
from character_clothing_changer_ai.config import ClothingChangerConfig
from PIL import Image

# Initialize service
config = ClothingChangerConfig.from_env()
service = ClothingChangerService(config=config)
service.initialize_model()

# Load image
image = Image.open("character.png")

# Change clothing
result = service.change_clothing(
    image=image,
    clothing_description="a red elegant dress",
    character_name="MyCharacter"
)

# Save result
result["changed_image"].save("result.png")
```

### REST API

```bash
# Change clothing
curl -X POST http://localhost:8002/api/v1/change-clothing \
  -F "image=@character.png" \
  -F "clothing_description=a red elegant dress" \
  -F "character_name=MyCharacter"
```

## 🧪 Testing

```bash
# Run tests (when implemented)
pytest tests/
```

## 📚 Documentation

- [Architecture](docs/ARCHITECTURE.md) - System architecture
- [API Documentation](docs/API.md) - API reference
- [Refactoring History](docs/REFACTORING_HISTORY.md) - Refactoring phases

## 🐛 Troubleshooting

### Model Not Loading

1. Check HuggingFace token: `echo $HUGGINGFACE_TOKEN`
2. Accept model terms: https://huggingface.co/black-forest-labs/flux2-dev
3. Check internet connection
4. System will fallback to DeepSeek if Flux2 fails

### Out of Memory

1. Reduce image size
2. Use CPU instead of CUDA
3. Reduce batch size
4. Use float16 instead of float32

### API Errors

1. Check server logs
2. Verify configuration
3. Check model initialization status: `GET /api/v1/health`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- Flux2 model by Black Forest Labs
- DeepSeek for fallback support
- FastAPI for the API framework
