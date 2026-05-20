# Mejoras Arquitectónicas - Multi-Model API

## Resumen Ejecutivo

Este documento describe las mejoras arquitectónicas propuestas para la API Multi-Model, enfocadas en:
- **Separación de responsabilidades** (SoC)
- **Principio de responsabilidad única** (SRP)
- **Inversión de dependencias** (DIP)
- **Patrón Strategy** para estrategias de ejecución
- **Service Layer** para lógica de negocio
- **Dependency Injection** para mejor testabilidad
- **Error Handling centralizado**

## Problemas Actuales Identificados

### 1. Router Monolítico
- **Problema**: `router.py` tiene 972 líneas y maneja múltiples responsabilidades
- **Impacto**: Difícil de mantener, testear y extender
- **Solución**: Separar en múltiples routers y crear service layer

### 2. Lógica de Negocio en Router
- **Problema**: Lógica de ejecución, agregación y validación mezclada con endpoints
- **Impacto**: Violación de SRP, difícil reutilización
- **Solución**: Crear service layer dedicado

### 3. Estrategias de Ejecución como Funciones
- **Problema**: `_execute_parallel`, `_execute_sequential`, `_execute_consensus` son funciones
- **Impacto**: No extensible, difícil agregar nuevas estrategias
- **Solución**: Implementar Strategy Pattern con clases

### 4. Dependencias Globales
- **Problema**: Uso de singletons globales (`get_registry()`, `get_cache()`)
- **Impacto**: Difícil testear, acoplamiento fuerte
- **Solución**: Dependency Injection container

### 5. Manejo de Errores Disperso
- **Problema**: Try-catch repetido en múltiples lugares
- **Impacto**: Inconsistencia, difícil mantener
- **Solución**: Exception handlers centralizados y custom exceptions

### 6. Cache Mezclado con Lógica de Negocio
- **Problema**: Lógica de cache mezclada con ejecución de modelos
- **Impacto**: Violación de SRP
- **Solución**: Decorator pattern o service wrapper

## Arquitectura Propuesta

### Nueva Estructura de Directorios

```
multi_model_api/
├── api/
│   ├── __init__.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── execution.py      # Endpoints de ejecución
│   │   ├── models.py          # Endpoints de modelos
│   │   ├── health.py          # Endpoints de salud
│   │   ├── cache.py           # Endpoints de cache
│   │   └── openrouter.py      # Endpoints de OpenRouter
│   ├── dependencies.py        # Dependency injection
│   ├── exceptions.py          # Custom exceptions
│   ├── exception_handlers.py  # Exception handlers
│   ├── schemas.py
│   └── websocket.py
├── core/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── execution_service.py    # Service principal de ejecución
│   │   ├── model_service.py         # Service de modelos
│   │   ├── cache_service.py         # Service de cache
│   │   └── consensus_service.py     # Service de consenso
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py            # Base strategy interface
│   │   ├── parallel.py         # Parallel strategy
│   │   ├── sequential.py      # Sequential strategy
│   │   └── consensus.py        # Consensus strategy
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── model_repository.py # Repository para modelos
│   ├── ... (resto de core)
└── ...
```

## Mejoras Detalladas

### 1. Service Layer

**Objetivo**: Separar lógica de negocio de la capa de API

**Implementación**:

```python
# core/services/execution_service.py
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models import ModelRegistry
from ..cache import EnhancedCache
from ..strategies import ExecutionStrategy
from ..repositories import ModelRepository
from ...api.schemas import MultiModelRequest, MultiModelResponse

class ExecutionService:
    """Service for executing multi-model requests"""
    
    def __init__(
        self,
        model_repository: ModelRepository,
        cache: EnhancedCache,
        strategy_factory: StrategyFactory
    ):
        self.model_repository = model_repository
        self.cache = cache
        self.strategy_factory = strategy_factory
    
    async def execute(
        self,
        request: MultiModelRequest
    ) -> MultiModelResponse:
        """Execute multi-model request"""
        # 1. Validate request
        self._validate_request(request)
        
        # 2. Check cache
        if request.cache_enabled:
            cached = await self._get_from_cache(request)
            if cached:
                return cached
        
        # 3. Get execution strategy
        strategy = self.strategy_factory.create(request.strategy)
        
        # 4. Execute models
        responses = await strategy.execute(
            models=request.models,
            prompt=request.prompt,
            repository=self.model_repository
        )
        
        # 5. Aggregate responses
        aggregated = await self._aggregate_responses(
            responses,
            request.strategy,
            request.consensus_method
        )
        
        # 6. Build response
        response = self._build_response(request, responses, aggregated)
        
        # 7. Cache result
        if request.cache_enabled:
            await self._cache_result(request, response)
        
        return response
```

### 2. Strategy Pattern para Ejecución

**Objetivo**: Hacer las estrategias de ejecución extensibles y mantenibles

**Implementación**:

```python
# core/strategies/base.py
from abc import ABC, abstractmethod
from typing import List
from ...api.schemas import ModelConfig, ModelResponse
from ..repositories import ModelRepository

class ExecutionStrategy(ABC):
    """Base class for execution strategies"""
    
    @abstractmethod
    async def execute(
        self,
        models: List[ModelConfig],
        prompt: str,
        repository: ModelRepository,
        timeout: Optional[float] = None
    ) -> List[ModelResponse]:
        """Execute models according to strategy"""
        pass

# core/strategies/parallel.py
class ParallelStrategy(ExecutionStrategy):
    """Execute models in parallel"""
    
    async def execute(
        self,
        models: List[ModelConfig],
        prompt: str,
        repository: ModelRepository,
        timeout: Optional[float] = None
    ) -> List[ModelResponse]:
        enabled = [m for m in models if m.is_enabled]
        
        tasks = [
            repository.execute_model(
                model.model_type,
                prompt,
                **build_model_kwargs(model, timeout)
            )
            for model in enabled
        ]
        
        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return self._process_results(enabled, results)

# core/strategies/strategy_factory.py
class StrategyFactory:
    """Factory for creating execution strategies"""
    
    _strategies = {
        "parallel": ParallelStrategy,
        "sequential": SequentialStrategy,
        "consensus": ConsensusStrategy
    }
    
    def create(self, strategy_name: str) -> ExecutionStrategy:
        strategy_class = self._strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return strategy_class()
```

### 3. Repository Pattern

**Objetivo**: Abstraer acceso a datos de modelos

**Implementación**:

```python
# core/repositories/model_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from ...api.schemas import ModelType, ModelResponse

class ModelRepository(ABC):
    """Repository for model operations"""
    
    @abstractmethod
    async def execute_model(
        self,
        model_type: ModelType,
        prompt: str,
        **kwargs
    ) -> ModelResponse:
        """Execute a model"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[ModelMetadata]:
        """Get available models"""
        pass

# core/repositories/registry_repository.py
class RegistryModelRepository(ModelRepository):
    """Repository implementation using ModelRegistry"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    async def execute_model(
        self,
        model_type: ModelType,
        prompt: str,
        **kwargs
    ) -> ModelResponse:
        return await self.registry.execute_model(
            model_type,
            prompt,
            **kwargs
        )
    
    def get_available_models(self) -> List[ModelMetadata]:
        return self.registry.get_available_models()
```

### 4. Dependency Injection

**Objetivo**: Mejorar testabilidad y reducir acoplamiento

**Implementación**:

```python
# api/dependencies.py
from functools import lru_cache
from fastapi import Depends
from ..core.models import ModelRegistry, get_registry
from ..core.cache import EnhancedCache, get_cache
from ..core.repositories import RegistryModelRepository, ModelRepository
from ..core.services import ExecutionService
from ..core.strategies import StrategyFactory

@lru_cache()
def get_model_repository(
    registry: ModelRegistry = Depends(get_model_registry)
) -> ModelRepository:
    """Get model repository"""
    return RegistryModelRepository(registry)

@lru_cache()
def get_strategy_factory() -> StrategyFactory:
    """Get strategy factory"""
    return StrategyFactory()

@lru_cache()
def get_execution_service(
    repository: ModelRepository = Depends(get_model_repository),
    cache: EnhancedCache = Depends(get_cache_instance),
    factory: StrategyFactory = Depends(get_strategy_factory)
) -> ExecutionService:
    """Get execution service"""
    return ExecutionService(repository, cache, factory)
```

### 5. Exception Handling Centralizado

**Objetivo**: Manejo consistente de errores

**Implementación**:

```python
# api/exceptions.py
class MultiModelAPIException(Exception):
    """Base exception for Multi-Model API"""
    pass

class ModelExecutionException(MultiModelAPIException):
    """Exception during model execution"""
    pass

class RateLimitExceededException(MultiModelAPIException):
    """Rate limit exceeded"""
    pass

class CacheException(MultiModelAPIException):
    """Cache operation failed"""
    pass

# api/exception_handlers.py
from fastapi import Request
from fastapi.responses import JSONResponse
from .exceptions import (
    MultiModelAPIException,
    ModelExecutionException,
    RateLimitExceededException
)

async def multi_model_exception_handler(
    request: Request,
    exc: MultiModelAPIException
) -> JSONResponse:
    """Handle Multi-Model API exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
            "path": request.url.path
        }
    )

async def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitExceededException
) -> JSONResponse:
    """Handle rate limit exceptions"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "RateLimitExceeded",
            "message": str(exc),
            "retry_after": getattr(exc, "retry_after", 60)
        },
        headers={
            "Retry-After": str(getattr(exc, "retry_after", 60))
        }
    )
```

### 6. Routers Separados

**Objetivo**: Dividir router monolítico en routers especializados

**Implementación**:

```python
# api/routers/execution.py
from fastapi import APIRouter, Depends, Request
from ..dependencies import get_execution_service
from ..services import ExecutionService
from ..schemas import MultiModelRequest, MultiModelResponse

router = APIRouter(prefix="/multi-model", tags=["Execution"])

@router.post("/execute", response_model=MultiModelResponse)
async def execute_multi_model(
    request: MultiModelRequest,
    http_request: Request,
    service: ExecutionService = Depends(get_execution_service)
):
    """Execute multiple AI models"""
    return await service.execute(request)

# api/routers/models.py
router = APIRouter(prefix="/multi-model/models", tags=["Models"])

@router.get("", response_model=ModelsListResponse)
async def list_models(
    repository: ModelRepository = Depends(get_model_repository)
):
    """List all available models"""
    return await repository.get_available_models()

# api/__init__.py
from .routers import execution, models, health, cache, openrouter

def get_all_routers():
    """Get all routers"""
    return [
        execution.router,
        models.router,
        health.router,
        cache.router,
        openrouter.router
    ]
```

### 7. Cache Service Wrapper

**Objetivo**: Separar lógica de cache de lógica de negocio

**Implementación**:

```python
# core/services/cache_service.py
class CacheService:
    """Service for cache operations"""
    
    def __init__(self, cache: EnhancedCache):
        self.cache = cache
    
    async def get_cached_response(
        self,
        request: MultiModelRequest
    ) -> Optional[MultiModelResponse]:
        """Get cached response if available"""
        if not request.cache_enabled:
            return None
        
        cache_key = self._generate_key(request)
        cached = await self.cache.get(cache_key)
        
        if cached:
            return MultiModelResponse(**cached, cache_hit=True)
        return None
    
    async def cache_response(
        self,
        request: MultiModelRequest,
        response: MultiModelResponse
    ):
        """Cache response"""
        if not request.cache_enabled:
            return
        
        cache_key = self._generate_key(request)
        ttl = request.cache_ttl or 3600
        
        await self.cache.set(
            cache_key,
            response.dict(),
            ttl=ttl
        )
```

## Plan de Implementación

### Fase 1: Foundation (Semana 1)
1. ✅ Crear estructura de directorios
2. ✅ Implementar custom exceptions
3. ✅ Implementar exception handlers
4. ✅ Crear dependency injection

### Fase 2: Service Layer (Semana 2)
1. ✅ Implementar ModelRepository
2. ✅ Implementar ExecutionService
3. ✅ Implementar CacheService
4. ✅ Migrar lógica de negocio

### Fase 3: Strategy Pattern (Semana 3)
1. ✅ Crear base strategy interface
2. ✅ Implementar ParallelStrategy
3. ✅ Implementar SequentialStrategy
4. ✅ Implementar ConsensusStrategy
5. ✅ Crear StrategyFactory

### Fase 4: Router Refactoring (Semana 4)
1. ✅ Dividir router en múltiples routers
2. ✅ Actualizar endpoints para usar services
3. ✅ Actualizar tests
4. ✅ Documentación

## Beneficios Esperados

### Mantenibilidad
- **-70% complejidad por archivo**: Routers más pequeños y enfocados
- **Mejor organización**: Separación clara de responsabilidades
- **Más legible**: Código más claro y fácil de entender

### Testabilidad
- **Unit tests más fáciles**: Services y repositories son testeables independientemente
- **Mocking simplificado**: Dependency injection facilita mocks
- **Mejor cobertura**: Componentes aislados son más fáciles de testear

### Extensibilidad
- **Nuevas estrategias**: Fácil agregar nuevas estrategias de ejecución
- **Nuevos endpoints**: Agregar endpoints no afecta código existente
- **Nuevos repositorios**: Fácil cambiar implementación de datos

### Performance
- **Sin impacto negativo**: Refactoring no afecta performance
- **Mejor cacheo**: Service layer puede optimizar cacheo
- **Mejor paralelismo**: Strategy pattern permite optimizaciones específicas

## Métricas de Éxito

- ✅ Reducción de complejidad ciclomática: 50%
- ✅ Reducción de líneas por archivo: 60%
- ✅ Aumento de cobertura de tests: 80%+
- ✅ Tiempo de desarrollo de nuevas features: -40%

## Notas de Implementación

1. **Backward Compatibility**: Mantener API pública compatible
2. **Testing**: Escribir tests antes de refactorizar
3. **Documentación**: Actualizar documentación en cada fase
4. **Code Review**: Revisión cuidadosa de cada cambio
5. **Gradual Migration**: Migrar endpoints uno por uno

## Referencias

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)

