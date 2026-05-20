# Deep Learning Generator - Ultra-Modular Architecture

## Overview

Refactorizado con arquitectura ultra-modular siguiendo mejores prácticas de Deep Learning y Python.

## Estructura Modular

```
deep_learning_generator/
├── __init__.py          # Public API exports
├── constants.py         # Configuration constants
├── validators.py        # Validation functions
├── factory.py           # Factory pattern implementation
└── integration.py      # External system integration
```

## Módulos

### 1. `constants.py`
Centraliza todas las constantes y configuraciones:
- `SUPPORTED_FRAMEWORKS`: Lista de frameworks soportados
- `SUPPORTED_MODEL_TYPES`: Tipos de modelos soportados
- `DEFAULT_CONFIG`: Configuración por defecto
- `CAPABILITIES`: Capacidades del generador
- `VERSION`: Versión del módulo

### 2. `validators.py`
Funciones puras de validación:
- `validate_config_dict()`: Valida diccionarios de configuración
- `validate_framework()`: Valida frameworks
- `validate_model_type()`: Valida tipos de modelos
- `validate_generator_config()`: Validación completa
- `ValidationError`: Excepción personalizada

### 3. `factory.py`
Implementación del patrón Factory:
- `GeneratorFactory`: Clase factory con lazy loading
- `get_factory()`: Singleton factory instance
- Manejo robusto de errores
- Integración de características avanzadas

### 4. `integration.py`
Integración con sistemas externos:
- `AdvancedFeaturesIntegrator`: Integración de pipelines avanzados
- `get_pipeline_info()`: Información de pipelines disponibles
- `integrate_pipelines()`: Integración de pipelines en configuración

### 5. `config_builder.py` ⭐ NEW
Fluent API para construir configuraciones:
- `ConfigBuilder`: Builder pattern con API fluida
- `create_config_builder()`: Factory para crear builders
- Métodos encadenables: `with_framework()`, `with_model_type()`, etc.
- Soporte para configuración personalizada

### 6. `presets.py` ⭐ NEW
Presets pre-configurados para casos comunes:
- `PresetManager`: Gestor de presets
- `get_preset()`: Obtener un preset por nombre
- `list_presets()`: Listar todos los presets disponibles
- 10+ presets incluidos: transformer, cnn, llm, diffusion, gan, vae, etc.

### 7. `cache.py` ⭐ NEW
Utilidades de caché:
- `GeneratorCache`: Sistema de caché para generadores
- `cached_generator()`: Decorator para cachear generadores
- `clear_all_caches()`: Limpiar todos los caches
- Optimización de rendimiento

### 8. `monitoring.py` ⭐ NEW
Métricas y monitoreo:
- `GeneratorMetrics`: Tracking de métricas de uso
- `get_metrics()`: Obtener métricas globales
- `record_generator_creation()`: Registrar creación de generadores
- Estadísticas de uso, errores, y éxito

### 9. `optimizer.py` ⭐ NEW
Optimización automática de configuraciones:
- `ConfigOptimizer`: Optimización basada en hardware
- `optimize_for_hardware()`: Optimizar según GPU/RAM/CPU
- `optimize_for_model_type()`: Optimizar según tipo de modelo
- `auto_tune_batch_size()`: Auto-tuning de batch size
- `optimize_learning_rate()`: Optimización de learning rate

### 10. `serialization.py` ⭐ NEW
Serialización de configuraciones:
- `ConfigSerializer`: Serializar/deserializar configuraciones
- `save_config()`: Guardar configuración a archivo (JSON/YAML)
- `load_config()`: Cargar configuración desde archivo
- Soporte para JSON y YAML

### 11. `testing.py` ⭐ NEW
Utilidades de testing:
- `GeneratorTester`: Suite de testing para generadores
- `test_config_validation()`: Test de validación
- `test_generator_creation()`: Test de creación
- `test_preset_loading()`: Test de presets
- `run_all_tests()`: Ejecutar todos los tests

### 12. `plugins.py` ⭐ NEW
Sistema de plugins:
- `GeneratorPlugin`: Clase base para plugins
- `PluginManager`: Gestor de plugins
- `register_plugin()`: Registrar plugins personalizados
- Hooks: `before_create()`, `after_create()`, `validate_config()`

### 13. `benchmark.py` ⭐ NEW
Benchmarking y comparación de rendimiento:
- `BenchmarkRunner`: Ejecutar benchmarks
- `BenchmarkResult`: Resultados de benchmarks
- `benchmark()`: Benchmark de funciones
- `compare_configs()`: Comparar configuraciones por rendimiento

### 14. `recommender.py` ⭐ NEW
Recomendaciones inteligentes:
- `ConfigRecommender`: Sistema de recomendaciones
- `recommend_for_use_case()`: Recomendaciones por caso de uso
- `recommend_based_on_data()`: Recomendaciones basadas en datos
- `recommend_for_budget()`: Recomendaciones por presupuesto
- 10+ casos de uso predefinidos

### 15. `comparator.py` ⭐ NEW
Comparación y análisis de configuraciones:
- `ConfigComparator`: Comparar configuraciones
- `compare()`: Comparar dos configuraciones
- `find_differences()`: Encontrar diferencias entre múltiples configs
- `merge_configs()`: Fusionar configuraciones con diferentes estrategias

### 16. `versioning.py` ⭐ NEW
Versionado de configuraciones:
- `ConfigVersion`: Versión de configuración con metadata
- `ConfigVersionManager`: Gestor de versiones
- `save_config_version()`: Guardar versión
- `get_config_versions()`: Obtener todas las versiones
- `get_latest_config_version()`: Obtener última versión

## Uso

### Básico

```python
from core.deep_learning_generator import create_generator

# Crear generador
generator = create_generator(
    framework="pytorch",
    model_type="transformer",
    enable_advanced_features=True
)
```

### Usando Presets ⭐ NEW

```python
from core.deep_learning_generator import create_generator, get_preset, list_presets

# Ver presets disponibles
print(list_presets())

# Usar preset
config = get_preset("llm_pytorch")
generator = create_generator(**config)
```

### Usando Config Builder (Fluent API) ⭐ NEW

```python
from core.deep_learning_generator import create_generator, create_config_builder

config = (create_config_builder()
          .with_framework("pytorch")
          .with_model_type("transformer")
          .with_gpu(True)
          .with_mixed_precision(True)
          .with_batch_size(32)
          .with_learning_rate(1e-4)
          .with_epochs(10)
          .with_early_stopping(True, patience=5)
          .build())

generator = create_generator(**config)
```

### Con Validación

```python
from core.deep_learning_generator import (
    create_generator,
    validate_generator_config,
    ValidationError
)

# Validar configuración
config = {"framework": "pytorch", "model_type": "transformer"}
is_valid, error = validate_generator_config(config)

if is_valid:
    generator = create_generator(**config)
else:
    print(f"Error: {error}")
```

### Con Monitoreo ⭐ NEW

```python
from core.deep_learning_generator import (
    create_generator,
    get_metrics,
    record_generator_creation
)

try:
    generator = create_generator(framework="pytorch", model_type="transformer")
    record_generator_creation("pytorch", "transformer", success=True)
except Exception as e:
    record_generator_creation("pytorch", "transformer", success=False, error=str(e))

# Ver métricas
metrics = get_metrics()
print(metrics.get_stats())
```

### Usando Factory Directamente

```python
from core.deep_learning_generator.factory import get_factory

factory = get_factory()
if factory.is_available:
    generator = factory.create(
        framework="pytorch",
        model_type="transformer"
    )
```

## Mejoras Implementadas

1. **Separación de Responsabilidades**: Cada módulo tiene un propósito único
2. **Factory Pattern**: Creación centralizada y controlada de instancias
3. **Validación Robusta**: Validación completa con mensajes de error claros
4. **Type Hints**: Type hints completos en todas las funciones
5. **Error Handling**: Manejo robusto de errores con logging
6. **Backward Compatibility**: Compatibilidad con código existente
7. **Lazy Loading**: Carga perezosa de dependencias
8. **Singleton Pattern**: Factory singleton para eficiencia

## Beneficios

- **Mantenibilidad**: Código más fácil de mantener y extender
- **Testabilidad**: Cada módulo puede ser testeado independientemente
- **Escalabilidad**: Fácil agregar nuevas características
- **Claridad**: Código más claro y legible
- **Robustez**: Mejor manejo de errores y validación
- **Productividad**: Presets y builders aceleran el desarrollo
- **Observabilidad**: Métricas y monitoreo integrados
- **Rendimiento**: Sistema de caché para optimización

## Nuevas Características ⭐

### 1. Config Builder (Fluent API)
API fluida para construir configuraciones complejas de forma legible y encadenable.

### 2. Presets
10+ presets pre-configurados para casos comunes:
- `transformer_pytorch`, `transformer_tensorflow`
- `cnn_pytorch`, `llm_pytorch`, `diffusion_pytorch`
- `gan_pytorch`, `vae_pytorch`, `vision_transformer`
- `production_ready`, `fast_prototyping`

### 3. Caché
Sistema de caché para optimizar creación de generadores y validaciones.

### 4. Monitoreo
Tracking completo de métricas:
- Creaciones exitosas/fallidas
- Uso por framework y tipo de modelo
- Historial de errores
- Tasas de éxito

### 5. Optimizer ⭐ NEW
Optimización automática de configuraciones:
- Optimización basada en hardware (GPU, RAM, CPU)
- Auto-tuning de batch size y learning rate
- Optimización específica por tipo de modelo

### 6. Serialization ⭐ NEW
Importar/exportar configuraciones:
- Guardar configuraciones a archivos (JSON/YAML)
- Cargar configuraciones desde archivos
- Persistencia de configuraciones

### 7. Testing ⭐ NEW
Suite completa de testing:
- Tests de validación de configuración
- Tests de creación de generadores
- Tests de presets
- Ejecución automática de todos los tests

### 8. Plugins ⭐ NEW
Sistema extensible de plugins:
- Crear plugins personalizados
- Hooks para modificar comportamiento
- Validación personalizada
- Extensibilidad sin modificar código base

### 9. Benchmark ⭐ NEW
Benchmarking de rendimiento:
- Medir tiempo de ejecución
- Comparar configuraciones
- Estadísticas detalladas (avg, min, max, std dev)
- Análisis de rendimiento

### 10. Recommender ⭐ NEW
Recomendaciones inteligentes:
- Recomendaciones por caso de uso (10+ casos)
- Recomendaciones basadas en datos
- Recomendaciones por presupuesto
- Configuraciones optimizadas automáticamente

### 11. Comparator ⭐ NEW
Comparación de configuraciones:
- Comparar dos configuraciones
- Encontrar diferencias
- Fusionar configuraciones
- Análisis de similitud

### 12. Versioning ⭐ NEW
Versionado de configuraciones:
- Guardar versiones con metadata
- Historial de versiones
- Comparar versiones
- Gestión de versiones

Ver `EXAMPLES.md` para más ejemplos de uso.

