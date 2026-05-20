# Integración con Optimization Core

## Resumen

La API de TruthGPT ahora está integrada con `optimization_core` para aprovechar las nuevas carpetas y estructuras de optimización avanzadas. Esta integración es transparente y mantiene compatibilidad completa con la API existente.

## Cambios Realizados

### 1. Módulo de Adaptadores (`optimizers/adapters.py`)

Se creó un nuevo módulo de adaptadores que:

- Detecta automáticamente si `optimization_core` está disponible
- Proporciona una interfaz unificada entre TruthGPT API y optimization_core
- Mantiene compatibilidad hacia atrás con PyTorch como fallback

**Características principales:**

```python
from truthgpt_api.optimizers.adapters import (
    OptimizationCoreAdapter,
    is_optimization_core_available,
    get_optimization_core_path,
    create_optimizer_from_core
)

# Verificar disponibilidad
if is_optimization_core_available():
    print("optimization_core está disponible!")
    print(f"Ruta: {get_optimization_core_path()}")
```

### 2. Optimizadores Actualizados

Todos los optimizadores principales han sido actualizados para usar `optimization_core` cuando está disponible:

- **Adam** (`optimizers/adam.py`)
- **SGD** (`optimizers/sgd.py`)
- **RMSprop** (`optimizers/rmsprop.py`)
- **Adagrad** (`optimizers/adagrad.py`)
- **AdamW** (`optimizers/adamw.py`)

**Nuevo parámetro opcional:**

Todos los optimizadores ahora aceptan un parámetro `use_optimization_core` (por defecto `True`):

```python
from truthgpt_api.optimizers import Adam

# Usar optimization_core si está disponible (por defecto)
optimizer = Adam(learning_rate=0.001, use_optimization_core=True)

# Forzar uso de PyTorch
optimizer = Adam(learning_rate=0.001, use_optimization_core=False)
```

### 3. Detección Automática

El sistema detecta automáticamente:

- Si `optimization_core` está disponible en el sistema
- Qué módulos de optimization_core están disponibles:
  - `optimizers/tensorflow/` - Optimizadores estilo TensorFlow
  - `optimizers/core/` - Optimizadores core
  - `optimizers/truthgpt/` - Optimizadores específicos de TruthGPT

### 4. Fallback Automático

Si `optimization_core` no está disponible o falla, el sistema automáticamente:

- Usa los optimizadores de PyTorch como fallback
- Mantiene la misma interfaz de API
- No requiere cambios en el código del usuario

## Uso

### Uso Básico (Sin Cambios)

El código existente sigue funcionando sin modificaciones:

```python
import truthgpt as tg

model = tg.Sequential([
    tg.layers.Dense(128, activation='relu'),
    tg.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tg.optimizers.Adam(learning_rate=0.001),
    loss=tg.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
```

### Verificar Uso de Optimization Core

Puedes verificar si se está usando `optimization_core`:

```python
optimizer = tg.optimizers.Adam(learning_rate=0.001)
config = optimizer.get_config()

if config.get('using_optimization_core'):
    print("✅ Usando optimization_core para mejor rendimiento!")
    print(f"Configuración: {config.get('optimization_core', {})}")
else:
    print("⚠️ Usando PyTorch como fallback")
```

### Uso Avanzado con Adaptadores

Para control más fino:

```python
from truthgpt_api.optimizers.adapters import OptimizationCoreAdapter

# Crear adaptador personalizado
adapter = OptimizationCoreAdapter(
    optimizer_type='adam',
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    use_core=True
)

# Usar con parámetros
optimizer = adapter(model.parameters())
```

## Estructura de Optimization Core

La integración busca optimizadores en las siguientes ubicaciones de `optimization_core`:

```
optimization_core/
├── optimizers/
│   ├── tensorflow/          # Optimizadores estilo TensorFlow
│   │   ├── tensorflow_inspired_optimizer.py
│   │   └── advanced_tensorflow_optimizer.py
│   ├── core/                # Optimizadores core
│   │   ├── unified_optimizer.py
│   │   └── generic_optimizer.py
│   ├── truthgpt/            # Optimizadores específicos TruthGPT
│   │   ├── truthgpt_dynamo_optimizer.py
│   │   └── truthgpt_inductor_optimizer.py
│   ├── specialized/         # Optimizadores especializados
│   ├── production/          # Optimizadores de producción
│   ├── kv_cache/            # Optimizadores de KV cache
│   └── quantum/              # Optimizadores cuánticos
```

## Compatibilidad

### ✅ Compatible con:

- Código existente de TruthGPT API
- Todos los optimizadores actuales (Adam, SGD, RMSprop, Adagrad, AdamW)
- PyTorch como fallback
- API de TensorFlow-like

### 🔄 Cambios Transparentes:

- Los optimizadores mantienen la misma interfaz
- El comportamiento es idéntico si `optimization_core` no está disponible
- No se requieren cambios en el código existente

## Beneficios

1. **Mejor Rendimiento**: Aprovecha optimizaciones avanzadas de `optimization_core`
2. **Transparente**: No requiere cambios en el código existente
3. **Flexible**: Permite elegir entre `optimization_core` y PyTorch
4. **Robusto**: Fallback automático si algo falla
5. **Extensible**: Fácil agregar nuevos optimizadores de `optimization_core`

## Troubleshooting

### optimization_core no se detecta

1. Verifica que `optimization_core` esté en el path de Python
2. Verifica que `optimization_core/__init__.py` exista
3. Usa `get_optimization_core_path()` para ver la ruta detectada

### Errores de importación

Si hay errores al importar desde `optimization_core`:

- El sistema automáticamente usa PyTorch como fallback
- No se requiere acción del usuario
- El código sigue funcionando normalmente

### Forzar uso de PyTorch

Si necesitas forzar el uso de PyTorch:

```python
optimizer = tg.optimizers.Adam(
    learning_rate=0.001,
    use_optimization_core=False
)
```

## Próximos Pasos

- Integración con más optimizadores de `optimization_core`
- Soporte para optimizadores especializados (quantum, kv_cache, etc.)
- Métricas de rendimiento comparativas
- Documentación de optimizadores avanzados

## Referencias

- [optimization_core/REFACTORING_COMPLETE_SUMMARY.md](../../optimization_core/REFACTORING_COMPLETE_SUMMARY.md)
- [optimization_core/DIRECTORY_STRUCTURE_GUIDE.md](../../optimization_core/DIRECTORY_STRUCTURE_GUIDE.md)

