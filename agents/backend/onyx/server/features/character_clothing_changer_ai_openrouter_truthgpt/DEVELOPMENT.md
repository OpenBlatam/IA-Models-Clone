# Development Guide

## Project Structure

```
character_clothing_changer_ai_openrouter_truthgpt/
├── api/                    # FastAPI routers
│   ├── clothing_router.py  # Main clothing change endpoints
│   └── health_router.py    # Health check endpoints
├── services/               # Business logic services
│   ├── clothing_service.py # Main orchestration service
│   └── comfyui_service.py  # ComfyUI workflow execution
├── infrastructure/         # External service clients
│   ├── openrouter_client.py
│   ├── truthgpt_client.py
│   ├── truthgpt_status.py
│   └── truthgpt_helpers.py
├── config/                 # Configuration
│   └── settings.py         # Application settings
├── workflows/              # ComfyUI workflow templates
│   └── flux_fill_clothing_changer.json
├── main.py                 # Application entry point
├── ARCHITECTURE.md         # Architecture documentation
└── README.md              # Project overview
```

## Development Setup

### Prerequisites

1. **Python 3.8+**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. **Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# OpenRouter (Optional)
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_ENABLED=true
OPENROUTER_MODEL=openai/gpt-4
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_MAX_TOKENS=2000

# TruthGPT (Optional)
TRUTHGPT_ENABLED=true
TRUTHGPT_ENDPOINT=
TRUTHGPT_TIMEOUT=120.0

# ComfyUI (Required)
COMFYUI_API_URL=http://localhost:8188
COMFYUI_WORKFLOW_PATH=workflows/flux_fill_clothing_changer.json

# Image Processing
MAX_IMAGE_SIZE=10485760
OUTPUT_DIR=outputs
SAVE_TENSORS=true
```

## Running the Application

### Development Mode

```bash
# Using uvicorn with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python directly
python main.py
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Code Style

### Python Style Guide

- Follow PEP 8 conventions
- Use type hints for all function parameters and return values
- Write docstrings for all classes and methods
- Keep functions focused and single-purpose
- Use descriptive variable and function names

### Example Code Structure

```python
from typing import Dict, Any, Optional

class ExampleService:
    """
    Brief description of the service.
    
    More detailed explanation of what the service does
    and how it fits into the overall architecture.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the service.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    async def process_data(
        self,
        input_data: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input data.
        
        Args:
            input_data: The data to process
            options: Optional processing options
            
        Returns:
            Dictionary with processing results
            
        Raises:
            ValueError: If input_data is invalid
        """
        if not input_data:
            raise ValueError("input_data cannot be empty")
        
        # Processing logic here
        return {"result": "processed"}
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_clothing_service.py
```

### Writing Tests

Create test files in a `tests/` directory:

```python
import pytest
from services.clothing_service import ClothingChangeService

@pytest.mark.asyncio
async def test_change_clothing_success():
    """Test successful clothing change"""
    service = ClothingChangeService()
    result = await service.change_clothing(
        image_url="https://example.com/image.png",
        clothing_description="a red dress"
    )
    assert result["success"] is True
    assert "prompt_id" in result
```

## Adding New Features

### 1. Service Layer

When adding a new service:

1. Create the service class in `services/`
2. Add initialization in `__init__`
3. Implement async methods with proper error handling
4. Add type hints and docstrings
5. Update `ARCHITECTURE.md`

### 2. API Endpoints

When adding new endpoints:

1. Create or update router in `api/`
2. Add request/response models
3. Implement endpoint handler
4. Add error handling
5. Update API documentation

### 3. Infrastructure Clients

When adding external service clients:

1. Create client class in `infrastructure/`
2. Implement connection pooling
3. Add error handling and retries
4. Include timeout configuration
5. Add cleanup methods

## Debugging

### Logging

The application uses Python's logging module:

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
```

### Debug Mode

Enable debug mode in `.env`:

```env
DEBUG=true
```

This enables:
- More verbose logging
- Detailed error messages
- Stack traces in responses

### Common Issues

1. **ComfyUI Connection Error**
   - Verify ComfyUI is running: `curl http://localhost:8188/`
   - Check `COMFYUI_API_URL` in environment

2. **OpenRouter API Error**
   - Verify API key is set correctly
   - Check API key has sufficient credits
   - Verify model name is correct

3. **TruthGPT Not Available**
   - Check TruthGPT modules are installed
   - Verify `TRUTHGPT_ENABLED` is set correctly
   - Service will continue without TruthGPT if unavailable

## Performance Optimization

### Async Operations

Always use async/await for I/O operations:

```python
# Good
async def fetch_data(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Avoid
def fetch_data(url: str) -> Dict[str, Any]:
    response = requests.get(url)  # Blocking
    return response.json()
```

### Connection Pooling

Reuse HTTP clients with connection pooling:

```python
class MyClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20
            )
        )
```

### Caching

Cache expensive operations:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_template(path: str) -> Dict[str, Any]:
    # Expensive operation
    return load_json(path)
```

## Error Handling Best Practices

### Validation

Validate inputs early:

```python
def process_request(data: Dict[str, Any]) -> Dict[str, Any]:
    if not data.get("required_field"):
        raise ValueError("required_field is required")
    
    value = data.get("optional_field", "default")
    if not isinstance(value, str):
        raise TypeError("optional_field must be a string")
```

### Error Responses

Return consistent error responses:

```python
def build_error_response(error: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    response = {
        "success": False,
        "error": error
    }
    if context:
        response["context"] = context
    return response
```

### Graceful Degradation

Handle service failures gracefully:

```python
async def optimize_prompt(prompt: str) -> str:
    try:
        return await openrouter_client.optimize(prompt)
    except Exception as e:
        logger.warning(f"Optimization failed: {e}, using original")
        return prompt  # Fallback to original
```

## Documentation

### Code Documentation

- Write docstrings for all public methods
- Include parameter descriptions
- Document return values
- Note any exceptions raised

### API Documentation

FastAPI automatically generates OpenAPI docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Architecture Documentation

Keep `ARCHITECTURE.md` updated when:
- Adding new services
- Changing data flow
- Modifying integration points
- Updating configuration

## Version Control

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add prompt optimization with OpenRouter
fix: Handle ComfyUI connection errors gracefully
docs: Update architecture documentation
refactor: Extract validation logic to separate method
test: Add tests for clothing service
```

### Branch Strategy

- `main`: Production-ready code
- `develop`: Development branch
- `feature/*`: Feature branches
- `fix/*`: Bug fix branches

## Deployment

### Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Configure proper CORS origins
- [ ] Set secure API keys
- [ ] Configure logging levels
- [ ] Set up monitoring
- [ ] Test all integrations
- [ ] Verify ComfyUI connectivity
- [ ] Load test the API

### Environment Variables

Never commit `.env` files. Use environment variables or secrets management:

- Docker: Use `docker-compose.yml` with environment variables
- Kubernetes: Use ConfigMaps and Secrets
- Cloud: Use platform-specific secret management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests
5. Update documentation
6. Submit a pull request

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ComfyUI API](https://github.com/comfyanonymous/ComfyUI)
- [OpenRouter API](https://openrouter.ai/docs)
- [Python Async/Await Guide](https://docs.python.org/3/library/asyncio.html)

