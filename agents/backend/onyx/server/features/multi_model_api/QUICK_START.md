# Quick Start - Nueva Arquitectura

## Instalación Rápida

### 1. Registrar Exception Handlers

```python
from fastapi import FastAPI
from multi_model_api import register_exception_handlers

app = FastAPI()

# Registrar exception handlers
register_exception_handlers(app)
```

### 2. Usar Nuevos Routers

```python
from multi_model_api import execution_router, models_router

app.include_router(execution_router)
app.include_router(models_router)
```

### 3. O Usar Router Legacy (Backward Compatible)

```python
from multi_model_api import router

app.include_router(router)  # Funciona igual que antes
```

## Ejemplo Completo

```python
from fastapi import FastAPI
from multi_model_api import (
    execution_router,
    models_router,
    register_exception_handlers,
    init_sentry,
    MetricsMiddleware,
    LoggingMiddleware
)

app = FastAPI(
    title="Multi-Model API",
    version="2.7.0"
)

# 1. Registrar exception handlers
register_exception_handlers(app)

# 2. Inicializar Sentry (opcional)
init_sentry(dsn=os.getenv("SENTRY_DSN"))

# 3. Agregar middleware
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)

# 4. Incluir routers
app.include_router(execution_router)
app.include_router(models_router)

# O usar router legacy
# app.include_router(router)
```

## Uso de Services Directamente

```python
from multi_model_api import get_execution_service
from multi_model_api.api.schemas import MultiModelRequest, ModelConfig, ModelType

# Obtener service
execution_service = get_execution_service()

# Crear request
request = MultiModelRequest(
    prompt="Hello, world!",
    models=[
        ModelConfig(
            model_type=ModelType.GPT_51,
            is_enabled=True
        )
    ],
    strategy="parallel"
)

# Ejecutar
response = await execution_service.execute(request)
print(response.aggregated_response)
```

## Agregar Nueva Estrategia

```python
from multi_model_api import StrategyFactory, ExecutionStrategy
from multi_model_api.api.schemas import ModelConfig, ModelResponse

class CustomStrategy(ExecutionStrategy):
    async def execute(self, models, prompt, execute_func, **kwargs):
        # Tu implementación personalizada
        responses = []
        for model in models:
            if model.is_enabled:
                response = await execute_func(model, prompt, **kwargs)
                responses.append(response)
        return responses

# Registrar
StrategyFactory.register_strategy("custom", CustomStrategy)

# Usar
strategy = StrategyFactory.create("custom")
```

## Migración Gradual

Puedes migrar gradualmente sin romper nada:

1. **Fase 1**: Registrar exception handlers
   ```python
   register_exception_handlers(app)
   ```

2. **Fase 2**: Agregar nuevos routers (coexisten con el viejo)
   ```python
   app.include_router(execution_router)  # Nuevo
   app.include_router(router)  # Viejo (todavía funciona)
   ```

3. **Fase 3**: Migrar endpoints uno por uno

4. **Fase 4**: Remover router legacy cuando todo esté migrado

## Beneficios Inmediatos

✅ **Exception Handling Mejorado**: Errores más consistentes y útiles
✅ **Código Más Limpio**: Routers más pequeños y enfocados
✅ **Mejor Testabilidad**: Services fáciles de mockear
✅ **Extensibilidad**: Fácil agregar nuevas estrategias

## Próximos Pasos

Ver [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) para migración completa.




