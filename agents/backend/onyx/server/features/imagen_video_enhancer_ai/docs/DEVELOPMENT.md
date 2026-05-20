# Development Guide - Imagen Video Enhancer AI

## Development Setup

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

```bash
# Clone or navigate to project
cd imagen_video_enhancer_ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy
```

## Development Tools

### Testing

Run tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=imagen_video_enhancer_ai --cov-report=html

# Run specific test file
pytest tests/test_validators.py

# Run with verbose output
pytest -v
```

### Code Quality

Format code:

```bash
black imagen_video_enhancer_ai/
```

Lint code:

```bash
flake8 imagen_video_enhancer_ai/
```

Type checking:

```bash
mypy imagen_video_enhancer_ai/
```

### Development Utilities

The project includes development utilities in `utils/dev_helpers.py`:

```python
from imagen_video_enhancer_ai.utils.dev_helpers import (
    timing_decorator,
    log_function_call,
    save_debug_info,
    PerformanceProfiler
)

# Timing decorator
@timing_decorator
async def my_function():
    ...

# Logging decorator
@log_function_call
def my_function():
    ...

# Performance profiler
profiler = PerformanceProfiler()
profiler.start("operation")
# ... do work ...
profiler.end("operation")
stats = profiler.get_stats("operation")
```

### Error Reporting

Use the error reporter for advanced error tracking:

```python
from imagen_video_enhancer_ai.utils.error_reporter import (
    ErrorReporter,
    create_error_context
)

reporter = ErrorReporter()

try:
    # ... code ...
except Exception as e:
    context = create_error_context(
        user_id="123",
        task_id="task_456"
    )
    error_id = reporter.report_error(e, context, severity="error")
    # Use error_id for tracking
```

## Project Structure

```
imagen_video_enhancer_ai/
├── api/                    # API routes
│   ├── routes/             # Route modules
│   ├── dependencies.py     # Shared dependencies
│   ├── middleware.py       # Middleware
│   ├── models.py           # Request/response models
│   └── enhancer_api.py     # Main app
├── core/                   # Core logic
│   ├── enhancer_agent.py   # Main agent
│   ├── constants.py        # Constants
│   └── ...
├── infrastructure/         # External integrations
├── utils/                  # Utilities
│   ├── dev_helpers.py      # Development helpers
│   ├── error_reporter.py   # Error reporting
│   └── ...
├── tests/                  # Tests
└── docs/                   # Documentation
```

## Adding New Features

### Adding a New Route

1. Create route file in `api/routes/`:

```python
# api/routes/my_feature_routes.py
from fastapi import APIRouter
from ..dependencies import get_agent

router = APIRouter(tags=["my_feature"])

@router.get("/my-endpoint")
async def my_endpoint():
    agent = get_agent()
    # ... implementation ...
```

2. Register in `api/enhancer_api.py`:

```python
from .routes import my_feature_routes

app.include_router(my_feature_routes.router)
```

### Adding a New Service

1. Add service type to `core/service_handler.py`
2. Implement service handler
3. Add API endpoint if needed

### Adding Tests

1. Create test file in `tests/`
2. Use pytest fixtures
3. Mock external dependencies
4. Test both success and error cases

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Save Debug Info

```python
from imagen_video_enhancer_ai.utils.dev_helpers import save_debug_info

save_debug_info({
    "task_id": "123",
    "status": "processing"
}, "task_debug")
```

### Performance Profiling

```python
from imagen_video_enhancer_ai.utils.dev_helpers import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.start("enhancement")
# ... code ...
profiler.end("enhancement")
stats = profiler.get_stats("enhancement")
print(stats)
```

## Best Practices

1. **Use Type Hints**: Always use type hints for better IDE support
2. **Write Tests**: Write tests for new features
3. **Document Code**: Add docstrings to functions and classes
4. **Follow PEP 8**: Use black for formatting
5. **Handle Errors**: Use proper error handling and reporting
6. **Log Appropriately**: Use appropriate log levels
7. **Validate Input**: Always validate user input
8. **Use Constants**: Use constants from `core/constants.py`

## Common Issues

### Import Errors

If you get import errors, make sure:
- Virtual environment is activated
- Dependencies are installed
- Python path includes project root

### Test Failures

- Check that mocks are set up correctly
- Verify test data is valid
- Check async/await usage

### API Errors

- Check that agent is initialized
- Verify dependencies are set up
- Check error logs

## Contributing

1. Create feature branch
2. Make changes
3. Write/update tests
4. Run tests and linting
5. Update documentation
6. Submit pull request




