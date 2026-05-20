# 🏗️ Architecture Documentation

## Overview

The Character Clothing Changer AI is built with a modular architecture that separates concerns and promotes maintainability.

## Architecture Layers

### 1. API Layer (`api/`)

**Responsibility**: Handle HTTP requests and responses

**Components**:
- **Routers** (`routers/`): Define API endpoints
- **Middleware** (`middleware/`): Request/response processing
- **Utils** (`utils/`): API-specific utilities
- **Dependencies** (`dependencies.py`): Dependency injection

**Key Features**:
- FastAPI-based REST API
- Request validation
- Error handling
- Response formatting

### 2. Core Layer (`core/`)

**Responsibility**: Business logic and orchestration

**Components**:
- **Service** (`clothing_changer_service.py`): Main service class
- **Factories** (`factories/`): Object creation patterns
- **Handlers** (`handlers/`): Request processing handlers
- **Managers** (`managers/`): Resource management
- **Utils** (`utils/`): Core utilities
- **Validators** (`validators.py`): Input validation
- **Exceptions** (`exceptions.py`): Custom exceptions
- **Constants** (`constants.py`): Application constants
- **Advanced Systems**:
  - `workflow.py`: Workflow system with dependencies
  - `pipeline.py`: Data processing pipelines
  - `orchestrator.py`: Service orchestration
  - `state_manager.py`: Application state management
  - `advanced_cache.py`: Advanced caching with multiple strategies
  - `service_base.py`: Base classes for services

**Key Features**:
- Service orchestration
- Model management
- Validation
- Error handling
- Workflow execution
- Pipeline processing
- State management
- Advanced caching

### 3. Model Layer (`models/`)

**Responsibility**: AI model implementations

**Components**:
- **Flux2 Model**: Primary AI model
- **DeepSeek Model**: Fallback AI model
- **Tensor Generator**: ComfyUI tensor generation
- **Prompt Enhancer**: Prompt optimization
- **Quality Metrics**: Result quality assessment

**Key Features**:
- Model abstraction
- Fallback mechanism
- Tensor generation
- Quality assessment

### 4. Configuration Layer (`config/`)

**Responsibility**: Application configuration

**Components**:
- **ClothingChangerConfig**: Main configuration class

**Key Features**:
- Environment variable support
- Configuration validation
- Default values

### 5. Frontend Layer (`static/`)

**Responsibility**: User interface

**Components**:
- **HTML** (`index.html`): Main interface
- **CSS** (`css/`): Styling
- **JavaScript** (`js/`): Application logic

**Key Features**:
- Event-driven architecture
- State management
- Client-side caching
- Error handling

## Data Flow

```
User Request
    ↓
API Router (validation)
    ↓
Service Layer (orchestration)
    ↓
Model Layer (AI processing)
    ↓
Result Builder (formatting)
    ↓
API Response
```

## Design Patterns

### 1. Factory Pattern
- **ModelFactory**: Creates model instances
- **ResultBuilder**: Builds result objects

### 2. Dependency Injection
- FastAPI's `Depends` for service injection
- Singleton pattern for service instance

### 3. Strategy Pattern
- Model selection (Flux2 vs DeepSeek)
- Validation strategies

### 4. Observer Pattern
- Event bus for frontend communication
- State change notifications

### 5. Singleton Pattern
- Service instance management
- Cache management

## Error Handling

### Exception Hierarchy

```
ClothingChangerError (base)
├── ModelError
│   ├── ModelNotInitializedError
│   └── ModelLoadError
├── ValidationError
│   ├── ImageValidationError
│   ├── TextValidationError
│   └── ParameterValidationError
├── ProcessingError
├── TensorGenerationError
├── APIError
└── ConfigurationError
```

### Error Flow

```
Exception Raised
    ↓
Middleware Catches
    ↓
Error Formatted
    ↓
JSON Response
```

## Validation Flow

```
Request Received
    ↓
RequestValidator.validate()
    ↓
ImageValidator.validate()
    ↓
TextValidator.validate()
    ↓
ParameterValidator.validate()
    ↓
All Valid → Process
Any Invalid → Error Response
```

## State Management (Frontend)

```
User Action
    ↓
Event Bus.emit()
    ↓
StateManager.set()
    ↓
UI Updates
```

## Caching Strategy

### Backend
- Model caching
- Embedding cache
- Result cache

### Frontend
- API response cache (TTL-based)
- Image cache
- State cache

## Security Considerations

1. **Input Validation**: All inputs validated
2. **File Size Limits**: Maximum file sizes enforced
3. **Error Messages**: No sensitive data in errors
4. **CORS**: Configured for development

## Performance Optimizations

1. **Model Caching**: Models cached after initialization
2. **Batch Processing**: Support for batch operations
3. **Async Processing**: Non-blocking operations
4. **Client-Side Caching**: Reduce API calls

## Scalability

1. **Modular Architecture**: Easy to scale components
2. **Stateless API**: Can be horizontally scaled
3. **Resource Management**: Proper resource cleanup
4. **Error Recovery**: Fallback mechanisms

## Testing Strategy

1. **Unit Tests**: Individual components
2. **Integration Tests**: Component interactions
3. **E2E Tests**: Full workflow
4. **Performance Tests**: Load testing

## Deployment Considerations

1. **Environment Variables**: Configuration via env vars
2. **Docker Support**: Containerization ready
3. **Logging**: Comprehensive logging
4. **Monitoring**: Health check endpoints

