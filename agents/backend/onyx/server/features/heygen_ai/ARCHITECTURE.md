# HeyGen AI - Architecture Documentation

## Overview

This document describes the improved architecture of HeyGen AI, following clean architecture principles and best practices for deep learning projects.

## Directory Structure

```
heygen_ai/
├── shared/                    # ✅ Shared types, enums, and configurations
│   ├── enums/                 # All enumerations
│   │   ├── avatar_enums.py
│   │   ├── voice_enums.py
│   │   ├── video_enums.py
│   │   └── script_enums.py
│   ├── types/                 # All data types and configs
│   │   ├── avatar_types.py
│   │   ├── voice_types.py
│   │   ├── video_types.py
│   │   └── script_types.py
│   └── interfaces/            # Interface definitions
│
├── core/                      # Core business logic
│   ├── diffusion/             # Diffusion model management
│   │   ├── pipeline_manager.py
│   │   └── scheduler_factory.py
│   ├── face_processing/       # Face detection and enhancement
│   │   └── face_service.py
│   ├── image_processing/       # Image processing utilities
│   │   ├── image_processor.py
│   │   └── prompt_enhancer.py
│   ├── avatar_manager.py       # Avatar generation orchestrator
│   ├── voice_engine.py        # Voice synthesis engine
│   ├── video_renderer.py      # Video rendering engine
│   └── script_generator.py    # Script generation engine
│
├── domain/                    # Domain layer (entities, value objects)
│   ├── entities/
│   ├── value_objects/
│   └── exceptions/
│
├── application/               # Application layer (use cases)
│   └── use_cases/
│
├── infrastructure/            # Infrastructure layer
│   ├── database/
│   ├── cache.py
│   └── external_apis.py
│
├── presentation/              # Presentation layer (API)
│   └── api.py
│
├── models/                    # PyTorch model architectures
├── data/                      # Data processing
├── training/                  # Training utilities
├── utils/                     # Shared utilities
│   └── device_manager.py      # Device management
└── config/                    # Configuration files
```

## Architecture Principles

### 1. Shared Module (`shared/`)

**Purpose**: Central location for all shared types, enums, and configurations.

**Benefits**:
- Single source of truth for types
- Easy to maintain and update
- Prevents duplication
- Clear separation of concerns

**Structure**:
- `enums/`: All enumerations (AvatarStyle, VoiceQuality, etc.)
- `types/`: All data classes and configurations
- `interfaces/`: Interface definitions (future)

### 2. Core Module (`core/`)

**Purpose**: Core business logic organized by domain.

**Structure**:
- **Modular submodules**: Each major component has its own module
  - `diffusion/`: Diffusion pipeline management
  - `face_processing/`: Face detection and enhancement
  - `image_processing/`: Image processing utilities
- **Orchestrators**: High-level managers that coordinate submodules
  - `avatar_manager.py`: Coordinates diffusion, face processing, and image processing
  - `voice_engine.py`: Coordinates TTS engines and audio processing
  - `video_renderer.py`: Coordinates video composition and effects
  - `script_generator.py`: Coordinates LLM models and prompt engineering

### 3. Domain Layer (`domain/`)

**Purpose**: Domain entities and business rules.

**Structure**:
- `entities/`: Domain entities
- `value_objects/`: Value objects
- `exceptions/`: Domain-specific exceptions

### 4. Application Layer (`application/`)

**Purpose**: Application use cases and orchestration.

**Structure**:
- `use_cases/`: Application use cases

### 5. Infrastructure Layer (`infrastructure/`)

**Purpose**: Technical implementations (database, cache, external APIs).

**Structure**:
- `database/`: Database implementations
- `cache.py`: Caching implementation
- `external_apis.py`: External API integrations

### 6. Presentation Layer (`presentation/`)

**Purpose**: API endpoints and presentation logic.

**Structure**:
- `api.py`: API endpoints
- `exception_handlers.py`: Exception handling
- `middleware.py`: Middleware

## Key Improvements

### 1. Centralized Types and Enums

**Before**: Types and enums scattered across multiple files
**After**: All types and enums in `shared/` module

**Example**:
```python
# Before (in avatar_manager.py)
class AvatarStyle(str, Enum):
    REALISTIC = "realistic"
    ...

# After (in shared/enums/avatar_enums.py)
class AvatarStyle(str, Enum):
    REALISTIC = "realistic"
    ...

# Usage (in avatar_manager.py)
from shared import AvatarStyle, AvatarGenerationConfig
```

### 2. Modular Core Components

**Before**: Large monolithic files with all functionality
**After**: Small, focused modules with single responsibilities

**Example**:
```
core/
├── diffusion/              # Diffusion-specific logic
│   ├── pipeline_manager.py
│   └── scheduler_factory.py
├── face_processing/         # Face processing logic
│   └── face_service.py
└── image_processing/       # Image processing logic
    ├── image_processor.py
    └── prompt_enhancer.py
```

### 3. Device Management

**Before**: Device detection duplicated in multiple files
**After**: Centralized in `utils/device_manager.py`

**Example**:
```python
from utils.device_manager import detect_device, get_torch_dtype

device = detect_device()
dtype = get_torch_dtype(device)
```

## Best Practices Applied

1. **Single Responsibility Principle**: Each module has one clear purpose
2. **DRY (Don't Repeat Yourself)**: Shared types prevent duplication
3. **Separation of Concerns**: Clear boundaries between layers
4. **Dependency Inversion**: High-level modules depend on abstractions
5. **Open/Closed Principle**: Easy to extend without modifying existing code

## Usage Examples

### Using Shared Types

```python
from shared import (
    AvatarStyle,
    AvatarQuality,
    AvatarGenerationConfig,
)

config = AvatarGenerationConfig(
    style=AvatarStyle.REALISTIC,
    quality=AvatarQuality.HIGH,
    resolution=Resolution.P1080,
)
```

### Using Core Modules

```python
from core.avatar_manager import AvatarManager
from shared import AvatarGenerationConfig, AvatarStyle

manager = AvatarManager()
config = AvatarGenerationConfig(style=AvatarStyle.REALISTIC)
avatar_path = await manager.generate_avatar("professional headshot", config)
```

### Using Device Management

```python
from utils.device_manager import detect_device, get_device_info

device = detect_device()
info = get_device_info(device)
print(f"Using device: {info['name']}")
```

## Migration Guide

### For Existing Code

1. **Replace local enums with shared enums**:
   ```python
   # Old
   from core.avatar_manager import AvatarStyle
   
   # New
   from shared import AvatarStyle
   ```

2. **Replace local configs with shared types**:
   ```python
   # Old
   from core.avatar_manager import AvatarGenerationConfig
   
   # New
   from shared import AvatarGenerationConfig
   ```

3. **Use device manager utilities**:
   ```python
   # Old
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # New
   from utils.device_manager import detect_device
   device = detect_device()
   ```

## Future Improvements

1. **Interfaces Module**: Define clear interfaces for all services
2. **Repository Pattern**: Implement repository pattern for data access
3. **Dependency Injection**: Add dependency injection container
4. **Event System**: Implement domain events
5. **CQRS**: Separate read and write models

## Conclusion

The new architecture provides:
- ✅ Better organization
- ✅ Clearer separation of concerns
- ✅ Easier maintenance
- ✅ Better testability
- ✅ Improved scalability



