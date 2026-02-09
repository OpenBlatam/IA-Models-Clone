# Dependency Container - Guía de Uso

## 📋 Resumen

El `DependencyContainer` mejorado proporciona un sistema robusto de inyección de dependencias con soporte para múltiples scopes, resolución automática de dependencias y operaciones thread-safe.

## 🎯 Características Principales

1. **Múltiples Scopes**: Singleton, Scoped, Transient
2. **Resolución Automática**: Basada en type hints
3. **Thread-Safe**: Operaciones seguras en entornos async
4. **Compatibilidad**: Mantiene compatibilidad con código existente

## 📚 Ejemplos de Uso

### 1. Registro de Servicios

#### Usando Decoradores (Recomendado)

```python
from ..core import singleton, scoped, transient

@singleton('ranking_service')
class RankingService:
    def calculate_score(self, ...):
        ...

@scoped('chat_service')
class ChatService:
    def __init__(self, chat_repository: ChatRepository, ranking_service: RankingService):
        self.chat_repository = chat_repository
        self.ranking_service = ranking_service
```

#### Registro Manual

```python
from ..core import container, ServiceScope

# Singleton - Una instancia para toda la aplicación
container.register_singleton(
    'ranking_service',
    RankingService
)

# Scoped - Una instancia por request/scope
container.register_scoped(
    'chat_service',
    ChatService,
    dependencies=['chat_repository', 'ranking_service']
)

# Transient - Nueva instancia cada vez
container.register_transient(
    'validator',
    ChatValidator
)
```

### 2. Resolución Automática de Dependencias

El contenedor puede detectar automáticamente las dependencias basándose en los type hints:

```python
class ChatService:
    def __init__(
        self,
        chat_repository: ChatRepository,  # Se resuelve automáticamente
        ranking_service: RankingService,    # Se resuelve automáticamente
        validator: ChatValidator             # Se resuelve automáticamente
    ):
        self.chat_repository = chat_repository
        self.ranking_service = ranking_service
        self.validator = validator

# Solo necesitas registrar los servicios
container.register_singleton('chat_repository', ChatRepository)
container.register_singleton('ranking_service', RankingService)
container.register_scoped('chat_service', ChatService)
# Las dependencias se resuelven automáticamente
```

### 3. Obtener Servicios

#### En Contexto Async (Recomendado)

```python
from ..core import container

# Por nombre
chat_service = await container.get('chat_service')

# Por tipo (requiere registro con service_type)
chat_service = await container.get_by_type(ChatService)

# Con scope específico
chat_service = await container.get('chat_service', scope_id='request_123')
```

#### En Contexto Sync (Para compatibilidad)

```python
# Usar get_sync() cuando no estás en contexto async
chat_service = container.get_sync('chat_service')
```

### 4. Uso en FastAPI Dependencies

```python
from fastapi import Depends
from ..core import container

async def get_chat_service():
    """Dependency para obtener ChatService"""
    return await container.get('chat_service', scope_id='request')

@router.post("/chats")
async def create_chat(
    chat_service: ChatService = Depends(get_chat_service)
):
    return await chat_service.create_chat(...)
```

### 5. Gestión de Scopes

```python
# Al inicio de un request
request_id = str(uuid.uuid4())

# Obtener servicios con scope
chat_service = await container.get('chat_service', scope_id=request_id)

# Al final del request, limpiar scope
container.clear_scope(request_id)
```

### 6. Integración con Middleware

```python
from fastapi import Request
from ..core import container

@app.middleware("http")
async def scope_middleware(request: Request, call_next):
    # Crear scope para este request
    scope_id = str(uuid.uuid4())
    request.state.scope_id = scope_id
    
    try:
        response = await call_next(request)
        return response
    finally:
        # Limpiar scope al finalizar request
        container.clear_scope(scope_id)
```

### 7. Ejemplo Completo

```python
from ..core import container, singleton, scoped
from sqlalchemy.orm import Session

# 1. Registrar repositorios (singleton)
@singleton('chat_repository')
class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

# 2. Registrar servicios (scoped)
@scoped('chat_service')
class ChatService:
    def __init__(
        self,
        chat_repository: ChatRepository,
        ranking_service: RankingService
    ):
        self.chat_repository = chat_repository
        self.ranking_service = ranking_service
    
    async def create_chat(self, ...):
        ...

# 3. En FastAPI route
@router.post("/chats")
async def create_chat(
    request: Request,
    ...
):
    # Obtener servicio con scope del request
    scope_id = getattr(request.state, 'scope_id', 'default')
    chat_service = await container.get('chat_service', scope_id=scope_id)
    
    return await chat_service.create_chat(...)
```

## 🔄 Migración desde Código Existente

### Antes (Código Existente)

```python
from ..core import container

container.register_factory('chat_service', lambda: ChatService(...))
service = container.get('chat_service')
```

### Después (Nuevo Código)

```python
from ..core import container

# Opción 1: Mantener compatibilidad
container.register_factory('chat_service', lambda: ChatService(...))
service = container.get_sync('chat_service')

# Opción 2: Migrar a nuevo sistema (recomendado)
container.register_scoped('chat_service', ChatService)
service = await container.get('chat_service')
```

## 🎯 Mejores Prácticas

1. **Usar Decoradores**: `@singleton`, `@scoped`, `@transient` para registro automático
2. **Type Hints**: Siempre usar type hints para resolución automática
3. **Scopes Apropiados**:
   - **Singleton**: Servicios sin estado o configuración global
   - **Scoped**: Servicios que necesitan estado por request
   - **Transient**: Validadores, helpers sin estado
4. **Limpiar Scopes**: Siempre limpiar scopes al finalizar requests
5. **Async First**: Preferir `await container.get()` sobre `get_sync()`

## ⚠️ Consideraciones

- **Thread Safety**: El contenedor es thread-safe para operaciones async
- **Circular Dependencies**: Evitar dependencias circulares entre servicios
- **Scope Lifecycle**: Los scopes deben limpiarse manualmente o mediante middleware
- **Type Resolution**: La resolución por tipo requiere registro explícito con `service_type`

## 📊 Comparación de Scopes

| Scope | Lifetime | Uso Recomendado |
|-------|----------|----------------|
| **Singleton** | Toda la aplicación | Servicios sin estado, configuración |
| **Scoped** | Por request/scope | Servicios con estado por request |
| **Transient** | Cada llamada | Validadores, helpers temporales |

## 🔍 Debugging

```python
# Ver servicios registrados
print(container._services.keys())

# Ver instancias singleton
print([name for name, reg in container._services.items() 
       if reg.scope == ServiceScope.SINGLETON and reg.instance])

# Ver scopes activos
print(container._scoped_instances.keys())
```




