# Mejoras Avanzadas del Framework

## Resumen

Se han agregado módulos avanzados para mejorar el rendimiento, seguridad, caché, operaciones asíncronas e integración de servicios.

## Nuevos Módulos

### 1. Cache Manager (`shared/ml/cache/`)

Sistema de caché avanzado con soporte para TTL y LRU.

**Características:**
- **LRUCache**: Implementación de caché LRU con tamaño máximo configurable
- **CacheEntry**: Entradas de caché con metadatos (TTL, contador de accesos)
- **CacheManager**: Gestor principal con múltiples estrategias
- **Decorador `@cached`**: Para cachear resultados de funciones automáticamente

**Ejemplo de uso:**
```python
from shared.ml import CacheManager

# Crear cache manager
cache = CacheManager(cache_type="lru", max_size=100, default_ttl=3600)

# Uso básico
cache.set("key", "value", ttl=60)
value = cache.get("key")

# Decorador
@cache.cached(ttl=300)
def expensive_function(x, y):
    return x + y
```

### 2. Security Utilities (`shared/ml/security/`)

Utilidades de seguridad y validación para operaciones ML.

**Componentes:**
- **InputSanitizer**: Sanitización de entradas de usuario
  - `sanitize_text()`: Limpia texto de caracteres peligrosos
  - `sanitize_prompt()`: Sanitiza prompts de generación
  - `validate_model_name()`: Valida nombres de modelos
- **RateLimiter**: Limitación de tasa para endpoints
  - Control de requests por ventana de tiempo
  - Tracking por identificador
- **ResourceLimiter**: Límites de recursos
  - Monitoreo de memoria CPU/GPU
  - Validación de límites configurados

**Ejemplo de uso:**
```python
from shared.ml import InputSanitizer, RateLimiter

# Sanitizar input
sanitizer = InputSanitizer()
clean_prompt = sanitizer.sanitize_prompt(user_input)

# Rate limiting
limiter = RateLimiter(max_requests=100, window_seconds=60)
if limiter.is_allowed(user_id):
    # Procesar request
    pass
```

### 3. Performance Optimizer (`shared/ml/performance/`)

Optimizaciones avanzadas de rendimiento.

**Componentes:**
- **ModelOptimizer**: Optimización de modelos para inferencia
  - `optimize_for_inference()`: Compilación, xformers, flash attention
  - `fuse_modules()`: Fusión de módulos
  - `enable_torchscript()`: Habilitar TorchScript
- **MemoryOptimizer**: Optimización de memoria
  - `enable_gradient_checkpointing()`: Ahorro de memoria en training
  - `clear_cache()`: Limpiar caché GPU
  - `get_memory_usage()`: Estadísticas de memoria

**Ejemplo de uso:**
```python
from shared.ml import ModelOptimizer, MemoryOptimizer

# Optimizar modelo
optimizer = ModelOptimizer()
optimized_model = optimizer.optimize_for_inference(
    model,
    compile_model=True,
    enable_xformers=True,
)

# Gestión de memoria
memory_opt = MemoryOptimizer()
memory_opt.enable_gradient_checkpointing(model)
stats = memory_opt.get_memory_usage()
```

### 4. Async Operations (`shared/ml/async_ops/`)

Ejecución asíncrona de operaciones bloqueantes.

**Componentes:**
- **AsyncExecutor**: Ejecutor asíncrono con ThreadPool/ProcessPool
  - `run()`: Ejecutar función asíncronamente
  - `run_batch()`: Ejecutar batch de operaciones
- **AsyncModelInference**: Wrapper para inferencia asíncrona
  - `predict()`: Predicción asíncrona
  - `predict_batch()`: Batch de predicciones
- **Decorador `@asyncify`**: Convertir función síncrona a asíncrona

**Ejemplo de uso:**
```python
from shared.ml import AsyncExecutor, AsyncModelInference, asyncify

# Executor manual
executor = AsyncExecutor(max_workers=4)
result = await executor.run(blocking_function, arg1, arg2)

# Inferencia asíncrona
async_inference = AsyncModelInference(model)
predictions = await async_inference.predict_batch(input_batch)

# Decorador
@asyncify
def blocking_function(x):
    return x * 2
```

### 5. Service Integration (`shared/ml/integration/`)

Integración y orquestación de servicios.

**Componentes:**
- **ServiceOrchestrator**: Orquestador de múltiples servicios
  - Registro de servicios con dependencias
  - Ejecución de workflows
  - Resolución de referencias entre servicios
- **PipelineBuilder**: Constructor de pipelines de servicios
  - API fluida para construir pipelines
  - Referencias a resultados anteriores

**Ejemplo de uso:**
```python
from shared.ml import ServiceOrchestrator, PipelineBuilder

# Registrar servicios
orchestrator = ServiceOrchestrator()
orchestrator.register_service("llm", llm_service)
orchestrator.register_service("diffusion", diffusion_service)

# Construir pipeline
pipeline = (
    PipelineBuilder(orchestrator)
    .add_step("llm", "generate", {"prompt": "..."})
    .add_step("diffusion", "generate_image", {
        "prompt": "$llm.output"  # Referencia a resultado anterior
    })
    .build()
)

# Ejecutar
results = pipeline()
```

## Integración con Servicios Existentes

Estos módulos se integran perfectamente con los servicios existentes:

### LLM Service
```python
from shared.ml import InputSanitizer, CacheManager, AsyncExecutor

class LLMServiceCore:
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.cache = CacheManager()
        self.executor = AsyncExecutor()
    
    async def generate(self, prompt: str):
        # Sanitizar
        clean_prompt = self.sanitizer.sanitize_prompt(prompt)
        
        # Verificar cache
        cache_key = self.cache.make_key("generate", clean_prompt)
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Generar (async)
        result = await self.executor.run(self._generate_sync, clean_prompt)
        
        # Cachear
        self.cache.set(cache_key, result, ttl=3600)
        return result
```

### Diffusion Service
```python
from shared.ml import ModelOptimizer, MemoryOptimizer

class DiffusionServiceCore:
    def __init__(self):
        self.model_optimizer = ModelOptimizer()
        self.memory_optimizer = MemoryOptimizer()
    
    def load_model(self, model_name: str):
        model = load_pipeline(model_name)
        
        # Optimizar para inferencia
        model = self.model_optimizer.optimize_for_inference(
            model,
            compile_model=True,
            enable_xformers=True,
        )
        
        return model
```

## Beneficios

1. **Rendimiento**: Caché reduce latencia, optimizaciones mejoran throughput
2. **Seguridad**: Sanitización y rate limiting protegen contra abusos
3. **Escalabilidad**: Operaciones asíncronas permiten mejor concurrencia
4. **Integración**: Orquestación facilita workflows complejos
5. **Mantenibilidad**: Módulos especializados y bien organizados

## Próximos Pasos

- Integrar estos módulos en los servicios existentes
- Agregar tests unitarios para cada módulo
- Documentar casos de uso avanzados
- Crear ejemplos de integración completa



