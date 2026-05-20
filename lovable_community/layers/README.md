# Layers Module

Clean Architecture implementation with clear separation of concerns.

**Version:** 1.0.0  
**Author:** Lovable Community Team

## Architecture Overview

This module implements a layered architecture following Clean Architecture principles:

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│  (API Routes, Controllers, Schemas)    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Application Layer                │
│  (Services, Use Cases, DTOs)             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Domain Layer                     │
│  (Entities, Exceptions, Interfaces)      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Infrastructure Layer             │
│  (Repositories, External Services)      │
└─────────────────────────────────────────┘
```

## Module Structure

### Domain Layer (`domain/`)

The domain layer contains the core business logic and rules. It has no dependencies on other layers.

- **Entities** (`entities/`): Core business objects
  - `PublishedChat`
  - `ChatRemix`
  - `ChatVote`
  - `ChatView`
  - `ChatEmbedding`
  - `ChatAIMetadata`

- **Exceptions** (`exceptions/`): Business exceptions
  - `ChatNotFoundError`
  - `InvalidChatError`
  - `DuplicateVoteError`
  - `RemixError`
  - `DatabaseError`

- **Interfaces** (`interfaces/`): Domain contracts
  - `IChatRepository`
  - `IVoteRepository`
  - `IViewRepository`
  - `IRemixRepository`
  - `IRankingService`
  - `IValidator`
  - `IAIProcessor`
  - `IScoreManager`

### Application Layer (`application/`)

The application layer orchestrates domain operations and coordinates between repositories, domain services, and external systems.

- **Services** (`services/`): Application services
  - `ChatService`: Orchestrates chat operations
  - `RankingService`: Handles ranking and scoring

## Type Aliases

The module provides type aliases for better type safety:

- `LayerName`: Alias for `str`, represents a layer name
- `LayerInfo`: Alias for `Dict[str, str]`, represents layer information dictionary
- `ModuleInfo`: Alias for `Dict[str, Any]`, represents module metadata dictionary

```python
from layers import LayerName, LayerInfo, ModuleInfo

def process_layer(name: LayerName) -> LayerInfo:
    # Type-safe function signature
    pass
```

## Usage

### Module Information

```python
from layers import (
    __version__,
    __author__,
    __description__,
    LAYER_DOMAIN,
    LAYER_APPLICATION,
    LAYER_INFRASTRUCTURE,
    LAYER_PRESENTATION,
)

print(f"Version: {__version__}")
print(f"Author: {__author__}")
print(f"Description: {__description__}")

# Use layer constants for reference
print(f"Domain layer: {LAYER_DOMAIN}")
print(f"Application layer: {LAYER_APPLICATION}")

# Use utility functions
from layers import (
    get_layer_info,
    get_module_info,
    get_all_layers,
    get_layer_description,
    get_layer_count,
    is_valid_layer,
    validate_layers,
)

# Get information about all layers
layer_info = get_layer_info()
for layer, description in layer_info.items():
    print(f"{layer}: {description}")

# Get module metadata
module_info = get_module_info()
print(f"Module version: {module_info['version']}")

# Get all layer names
all_layers = get_all_layers()
print(f"Available layers: {all_layers}")

# Validate layer names
if is_valid_layer("domain"):
    print("Valid layer name")

# Get description for a specific layer
description = get_layer_description("domain")
if description:
    print(f"Domain layer: {description}")

# Get total number of layers
layer_count = get_layer_count()
print(f"Total layers: {layer_count}")

# Validate multiple layers at once
validation_results = validate_layers(("domain", "application", "invalid"))
for layer, is_valid in validation_results.items():
    print(f"{layer}: {'valid' if is_valid else 'invalid'}")
```

### Import from Root Module

```python
from layers import (
    PublishedChat,
    ChatService,
    ChatNotFoundError,
    IChatRepository,
)
```

### Import from Specific Layers

```python
# Domain layer
from layers.domain import (
    PublishedChat,
    ChatNotFoundError,
    IChatRepository,
)

# Application layer
from layers.application import (
    ChatService,
    RankingService,
)
```

### Import from Sub-modules

```python
# Domain entities
from layers.domain.entities import PublishedChat, ChatRemix

# Domain exceptions
from layers.domain.exceptions import ChatNotFoundError

# Domain interfaces
from layers.domain.interfaces import IChatRepository

# Application services
from layers.application.services import ChatService
```

## Utility Functions

The module provides several utility functions for working with layers:

### `get_layer_info()`

Returns information about all layers in the architecture.

```python
from layers import get_layer_info

layer_info = get_layer_info()
# Returns:
# {
#     "domain": "Business entities and rules",
#     "application": "Use cases and application services",
#     "infrastructure": "Technical implementations",
#     "presentation": "API layer"
# }
```

### `get_module_info()`

Returns module metadata information.

```python
from layers import get_module_info

module_info = get_module_info()
# Returns:
# {
#     "version": "1.0.0",
#     "author": "Lovable Community Team",
#     "description": "Clean Architecture implementation with layered separation"
# }
```

### `is_valid_layer(layer_name: str)`

Validates if a layer name is valid.

```python
from layers import is_valid_layer, LAYER_DOMAIN

if is_valid_layer(LAYER_DOMAIN):
    print("Valid layer")
    
if is_valid_layer("invalid"):
    print("This won't print")
```

### `get_all_layers()`

Returns a sorted tuple of all valid layer names.

```python
from layers import get_all_layers

layers = get_all_layers()
# Returns: ('application', 'domain', 'infrastructure', 'presentation')

for layer in layers:
    print(f"Layer: {layer}")

# Convert to list if needed
layer_list = list(layers)
```

### `get_layer_description(layer_name: str)`

Returns the description for a specific layer.

```python
from layers import get_layer_description, LAYER_DOMAIN

description = get_layer_description(LAYER_DOMAIN)
# Returns: 'Business entities and rules'

# Returns None for invalid layers
invalid_desc = get_layer_description("invalid")
# Returns: None
```

### `get_layer_count()`

Returns the total number of layers in the architecture.

```python
from layers import get_layer_count

count = get_layer_count()
# Returns: 4
```

### `validate_layers(layer_names: Tuple[str, ...])`

Validates multiple layer names at once.

```python
from layers import validate_layers

results = validate_layers(("domain", "invalid", "application"))
# Returns: {'domain': True, 'invalid': False, 'application': True}

# Check if all are valid
all_valid = all(validate_layers(("domain", "application")).values())
# Returns: True
```

## Performance Optimizations

The module includes several performance optimizations:

### Caching

Utility functions that return constant data are cached using `@lru_cache`:

- `get_layer_info()`: Cached with maxsize=1 (always returns same data)
- `get_module_info()`: Cached with maxsize=1 (always returns same data)
- `get_all_layers()`: Cached with maxsize=1 (always returns same data)
- `get_layer_count()`: Cached with maxsize=1 (always returns same data)
- `get_layer_description()`: Cached with maxsize=4 (one per layer)

### Efficient Validation

- Layer validation uses a `frozenset` for O(1) lookup performance
- Type checking is performed before validation to avoid unnecessary operations

### Immutable Returns

- `get_all_layers()` returns a tuple instead of a list for immutability and better caching

## Principles

### Dependency Rule

- **Domain Layer**: No dependencies on other layers
- **Application Layer**: Depends only on Domain layer
- **Infrastructure Layer**: Depends on Domain and Application layers
- **Presentation Layer**: Depends on all layers

### Benefits

1. **Testability**: Easy to test domain logic in isolation
2. **Maintainability**: Clear separation of concerns
3. **Flexibility**: Easy to swap implementations
4. **Independence**: Domain logic independent of frameworks
5. **Clarity**: Clear boundaries between layers

## Best Practices

1. **Always import from the root module** when possible for consistency
2. **Use domain exceptions** for business rule violations
3. **Depend on interfaces**, not implementations
4. **Keep domain layer pure** - no infrastructure concerns
5. **Use application services** to orchestrate domain operations

## Examples

### Using Domain Entities

```python
from layers import PublishedChat, ChatNotFoundError

try:
    chat = PublishedChat(
        id="chat-123",
        user_id="user-456",
        title="My Chat",
        # ... other fields
    )
except InvalidChatError as e:
    # Handle validation error
    pass
```

### Using Application Services

```python
from layers import ChatService, ChatNotFoundError

service = ChatService(
    chat_repository=chat_repo,
    ranking_service=ranking_service,
)

try:
    chat = service.get_chat("chat-123")
except ChatNotFoundError:
    # Handle not found
    pass
```

### Using Domain Interfaces

```python
from layers import IChatRepository
from typing import Protocol

class MyChatRepository(IChatRepository):
    def get_by_id(self, chat_id: str):
        # Implementation
        pass
```

## Migration Guide

If you're using direct imports from other modules, migrate to the layers module:

```python
# Old
from models import PublishedChat
from exceptions import ChatNotFoundError
from services.chat import ChatService

# New (recommended)
from layers import (
    PublishedChat,
    ChatNotFoundError,
    ChatService,
)
```

## See Also

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)

