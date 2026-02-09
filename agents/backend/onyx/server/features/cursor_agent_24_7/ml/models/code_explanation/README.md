# Code Explanation Model - Arquitectura Modular

## 📁 Estructura Modular

El modelo de explicación de código está completamente modularizado en componentes separados:

```
code_explanation/
├── __init__.py          # Exportaciones del módulo
├── model.py             # Modelo principal (orquestador)
├── model_loader.py       # Carga de modelo y tokenizer
├── validator.py          # Validación de inputs
├── prompt_builder.py     # Construcción de prompts
├── cache.py              # Gestión de caché
├── batch_processor.py    # Procesamiento en batch
└── stats.py              # Estadísticas del modelo
```

## 🧩 Componentes

### 1. `model.py` - Modelo Principal
**Responsabilidad**: Orquestar todos los componentes y proporcionar la API pública.

**Características**:
- Integra todos los módulos
- Proporciona métodos públicos: `generate()`, `explain_code()`, `explain_batch()`
- Gestiona el ciclo de vida del modelo
- Mantiene compatibilidad con `BaseModel`

### 2. `model_loader.py` - Carga de Modelos
**Responsabilidad**: Cargar y gestionar el modelo y tokenizer.

**Características**:
- Carga lazy del modelo
- Optimizaciones automáticas (torch.compile, mixed precision)
- Gestión de dispositivos (CPU/GPU)
- Guardado y carga de modelos

### 3. `validator.py` - Validación
**Responsabilidad**: Validar todos los inputs y parámetros.

**Métodos**:
- `validate_code()`: Valida código de entrada
- `validate_generation_params()`: Valida parámetros de generación
- `validate_config()`: Valida configuración del modelo
- `validate_batch_codes()`: Valida lista de códigos

### 4. `prompt_builder.py` - Construcción de Prompts
**Responsabilidad**: Construir prompts para diferentes niveles de detalle.

**Características**:
- Templates configurables
- Soporte para diferentes lenguajes
- Niveles de detalle: brief, medium, detailed
- Ajuste automático de max_length según detalle

### 5. `cache.py` - Gestión de Caché
**Responsabilidad**: Gestionar caché de explicaciones.

**Características**:
- Soporte para caché externo e interno
- TTL configurable
- Integración con estadísticas
- Fallback automático

### 6. `batch_processor.py` - Procesamiento en Batch
**Responsabilidad**: Procesar múltiples códigos eficientemente.

**Características**:
- Procesamiento en batch optimizado
- Validación automática
- Caché integrado
- Manejo de errores robusto

### 7. `stats.py` - Estadísticas
**Responsabilidad**: Trackear estadísticas de uso.

**Métricas**:
- Total de requests
- Cache hits/misses
- Errores
- Cache hit rate

## 🔄 Flujo de Uso

### Uso Básico
```python
from ml.models.code_explanation import CodeExplanationModel

config = {
    "model_name": "t5-small",
    "max_length": 512,
    "max_target_length": 128
}

model = CodeExplanationModel(config)
explanation = model.generate("def hello(): print('world')")
```

### Con Opciones Avanzadas
```python
# Explicación con nivel de detalle
explanation = model.explain_code(
    code="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
    language="python",
    detail_level="detailed"
)

# Procesamiento en batch
codes = ["def func1(): pass", "def func2(): return 42"]
explanations = model.explain_batch(codes, temperature=0.5)
```

### Con Caché Externo
```python
from ml.models.code_explanation import CodeExplanationModel, ExplanationCache

# Usar caché externo (Redis, etc.)
external_cache = get_redis_cache()
cache = ExplanationCache(
    enabled=True,
    ttl=3600,
    cache_instance=external_cache
)

config = {
    "model_name": "t5-small",
    "cache_instance": cache
}
model = CodeExplanationModel(config)
```

## 📊 Estadísticas

```python
# Obtener estadísticas del modelo
stats = model.get_stats()
# {
#     "total_requests": 100,
#     "cache_hits": 45,
#     "cache_misses": 55,
#     "errors": 0,
#     "cache_hit_rate": 45.0,
#     ...
# }

# Obtener estadísticas del caché
cache_stats = model.get_cache_stats()
# {
#     "enabled": True,
#     "external_cache_available": True,
#     "internal_cache_size": 10,
#     ...
# }
```

## 🎯 Ventajas de la Arquitectura Modular

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad única
2. **Testabilidad**: Cada componente se puede testear independientemente
3. **Reutilización**: Los componentes se pueden usar en otros contextos
4. **Mantenibilidad**: Cambios en un módulo no afectan a otros
5. **Extensibilidad**: Fácil agregar nuevas funcionalidades
6. **Debugging**: Más fácil identificar problemas en módulos específicos

## 🔧 Extensión

Para agregar nuevas funcionalidades:

1. **Nuevo componente**: Crear un nuevo módulo en `code_explanation/`
2. **Integración**: Importar y usar en `model.py`
3. **Exportación**: Agregar a `__init__.py`

Ejemplo:
```python
# code_explanation/explanation_formatter.py
class ExplanationFormatter:
    def format(self, explanation: str, style: str) -> str:
        ...

# model.py
from .explanation_formatter import ExplanationFormatter
self.formatter = ExplanationFormatter()

# Usar en generate()
explanation = self.formatter.format(explanation, style="markdown")
```

## ✅ Compatibilidad

- **Backward Compatible**: El archivo `code_explanation.py` mantiene compatibilidad
- **API Pública**: Sin cambios en la API pública
- **Módulos Opcionales**: Los componentes se pueden usar independientemente

