# Services Refactoring - Modular Service Architecture

This document describes the refactoring of services to use the modular architecture.

## 🎯 Refactoring Goals

1. **Separation of Concerns**: Separate API, business logic, and configuration
2. **Use Modular Components**: Leverage all the modular infrastructure
3. **Dependency Injection**: Use DI for better testability
4. **Event-Driven**: Integrate event system
5. **Service Layer**: Use service layer pattern
6. **Repository Pattern**: Use repositories for data access

## 📁 New Service Structure

### LLM Service Structure

```
services/llm_service/
├── main.py                 # FastAPI app entry point
├── core/
│   └── service_core.py    # Business logic (uses modular components)
├── api/
│   └── endpoints.py       # FastAPI endpoints
└── config/
    └── __init__.py        # Configuration loading
```

### Diffusion Service Structure

```
services/diffusion_service/
├── main.py                 # FastAPI app entry point
├── core/
│   └── service_core.py    # Business logic
├── api/
│   └── endpoints.py       # FastAPI endpoints
└── config/
    └── __init__.py        # Configuration loading
```

## ✨ Key Improvements

### 1. Separation of Concerns

**Before**: All logic in `main.py`

**After**:
- `main.py`: FastAPI app setup only
- `core/service_core.py`: Business logic
- `api/endpoints.py`: API endpoints
- `config/`: Configuration management

### 2. Use of Modular Components

**LLM Service Core** now uses:
- ✅ `ModelManager`: Model loading and caching
- ✅ `InferenceEngine`: Optimized inference
- ✅ `EventBus`: Event-driven communication
- ✅ `ServiceRegistry`: Service management
- ✅ `RepositoryManager`: Data access
- ✅ `validate_generation_params`: Input validation
- ✅ `error_handler`, `timing_decorator`: Decorators

**Diffusion Service Core** now uses:
- ✅ `EventBus`: Event-driven communication
- ✅ `error_handler`, `timing_decorator`: Decorators
- ✅ Modular pipeline loading

### 3. Dependency Injection

```python
# Before
_model_cache = {}  # Global state

# After
def get_service_core() -> LLMServiceCore:
    config = get_config()
    return LLMServiceCore(config=config)

@router.post("/generate")
async def generate_text(
    request: TextGenerationRequest,
    service: LLMServiceCore = Depends(get_service_core),
):
    # Use injected service
    result = service.generate_text(...)
```

### 4. Event Integration

```python
# Services now emit events
self.event_bus.publish(
    EventType.MODEL_LOADED,
    {"model_name": model_name, "device": self.device}
)
```

### 5. Configuration Management

```python
# Centralized configuration
config = get_config()  # Loads from YAML or env vars
service = LLMServiceCore(config=config)
```

## 🔧 Service Core Pattern

### LLM Service Core

```python
class LLMServiceCore:
    def __init__(self, config):
        # Initialize modular components
        self.model_manager = ModelManager(...)
        self.repo_manager = RepositoryManager()
        self.service_registry = ServiceRegistry()
        self.event_bus = EventBus()
    
    def generate_text(self, prompt, **kwargs):
        # Use InferenceEngine
        engine = self.get_inference_engine(model_name)
        return engine.generate(prompt, **kwargs)
```

### Benefits

- **Testable**: Easy to mock dependencies
- **Modular**: Uses all modular components
- **Event-Driven**: Emits events for monitoring
- **Configurable**: YAML-based configuration
- **Validated**: Input validation built-in

## 📊 Comparison

| Aspect | Before | After |
|--------|--------|-------|
| Structure | Single file | Modular (core/api/config) |
| Dependencies | Global state | Dependency injection |
| Components | Direct imports | Modular components |
| Events | None | EventBus integrated |
| Validation | Manual | Validators |
| Configuration | Hard-coded | YAML + env vars |
| Testing | Difficult | Easy (DI) |

## 🚀 Usage

### Starting Services

```bash
# LLM Service
python services/llm_service/main.py

# Diffusion Service
python services/diffusion_service/main.py
```

### API Usage

```python
# LLM Service
POST /api/v1/generate
{
    "prompt": "The future of AI",
    "model_name": "gpt2",
    "max_length": 100
}

# Diffusion Service
POST /api/v1/text-to-image
{
    "prompt": "A beautiful landscape",
    "num_inference_steps": 50
}
```

## 🎉 Benefits

- ✅ **Clean Architecture**: Clear separation of concerns
- ✅ **Modular**: Uses all framework components
- ✅ **Testable**: Dependency injection enables testing
- ✅ **Event-Driven**: Integrated event system
- ✅ **Configurable**: YAML-based configuration
- ✅ **Maintainable**: Easy to understand and modify
- ✅ **Extensible**: Easy to add new features

---

**Services are now fully refactored with modular architecture! 🚀**



