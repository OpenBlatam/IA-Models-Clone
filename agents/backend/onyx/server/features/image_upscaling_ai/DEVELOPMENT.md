# Development Guide

## Setup Development Environment

### 1. Install Dependencies

```bash
# Install required dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio

# Install Real-ESRGAN (optional)
pip install realesrgan basicsr
```

### 2. Run Setup Script

```bash
python scripts/setup.py
```

This will:
- Check dependencies
- Create directories
- Setup environment file
- Verify installation

### 3. Run Health Check

```bash
python scripts/health_check.py
```

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_basic.py::test_basic_upscaling

# Run with coverage
pytest --cov=image_upscaling_ai tests/
```

### Test Structure

```
tests/
├── test_basic.py          # Basic functionality tests
├── test_realesrgan.py     # Real-ESRGAN tests
├── test_quality.py        # Quality validation tests
└── test_performance.py    # Performance tests
```

## Benchmarking

### Run Benchmark

```bash
python scripts/benchmark.py
```

This will benchmark all available methods and show:
- Average processing time
- Min/Max times
- Quality scores
- Performance comparison

## Development Workflow

### 1. Make Changes

```bash
# Edit code
# ...

# Run tests
pytest tests/

# Check health
python scripts/health_check.py
```

### 2. Profile Performance

```python
from image_upscaling_ai.utils import profile_context

with profile_context("profile.stats"):
    result = await service.upscale_image_enhanced(image, 4.0)

# Analyze
# python -m pstats profile.stats
```

### 3. Validate Configuration

```python
from image_upscaling_ai.utils import ConfigValidator

config = {...}
validation = ConfigValidator.validate_config(config)
if not validation["valid"]:
    print("Issues:", validation["issues"])
```

## Code Quality

### Linting

```bash
# Run linter
pylint image_upscaling_ai/

# Format code
black image_upscaling_ai/
```

### Type Checking

```bash
# Run type checker
mypy image_upscaling_ai/
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Profiler

```python
from image_upscaling_ai.utils import profile_function

@profile_function("debug_profile.stats")
async def debug_function():
    # Your code
    pass
```

## Adding New Features

### 1. Create Module

```python
# models/new_feature.py
class NewFeature:
    def __init__(self):
        pass
```

### 2. Add to __init__.py

```python
# models/__init__.py
from .new_feature import NewFeature
__all__.append("NewFeature")
```

### 3. Write Tests

```python
# tests/test_new_feature.py
def test_new_feature():
    feature = NewFeature()
    assert feature is not None
```

### 4. Update Documentation

Add to relevant documentation files.

## Best Practices

1. **Write Tests**: Always write tests for new features
2. **Document**: Update documentation
3. **Profile**: Profile performance-critical code
4. **Validate**: Use ConfigValidator for configs
5. **Error Handling**: Use ErrorRecovery utilities
6. **Logging**: Use appropriate log levels

## Troubleshooting

### Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode
pip install -e .
```

### Test Failures

```bash
# Run with verbose output
pytest -v tests/

# Run with print statements
pytest -s tests/
```

### Performance Issues

```bash
# Run benchmark
python scripts/benchmark.py

# Profile code
python -m cProfile -o profile.stats your_script.py
```

## Summary

Development tools:
- ✅ Setup script
- ✅ Health check
- ✅ Benchmark script
- ✅ Test suite
- ✅ Profiling tools
- ✅ Configuration validation

Happy coding! 🚀


