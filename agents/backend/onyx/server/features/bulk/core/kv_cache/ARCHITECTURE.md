# 🏗️ Arquitectura del KV Cache - Versión 3.3.0

## 📐 Visión General de la Arquitectura

El sistema KV Cache sigue una arquitectura modular en capas con separación clara de responsabilidades.

## 🎯 Principios Arquitectónicos

1. **Modularidad**: Cada componente es independiente y reutilizable
2. **Composición sobre Herencia**: Uso de composición para flexibilidad
3. **Interfaces Claras**: ABCs y Protocols para contratos claros
4. **Separación de Concerns**: Cada módulo tiene una responsabilidad única
5. **Dependency Injection**: Componentes inyectados, no hardcodeados
6. **Open/Closed Principle**: Abierto para extensión, cerrado para modificación

## 📁 Estructura de Módulos

```
kv_cache/
├── 📦 Foundation Layer (Fundación)
│   ├── types.py           # Type aliases y type hints
│   ├── constants.py        # Constantes centralizadas
│   ├── interfaces.py       # ABCs y Protocols
│   ├── exceptions.py       # Jerarquía de excepciones
│   └── config.py           # Configuración centralizada
│
├── 🏗️ Core Layer (Núcleo)
│   ├── base.py             # BaseKVCache - Clase base
│   ├── cache_storage.py    # CacheStorage - Almacenamiento
│   ├── stats.py            # CacheStatsTracker - Estadísticas
│   └── strategies/         # Estrategias de eviction
│       ├── base.py         # BaseEvictionStrategy
│       ├── lru.py          # LRUEvictionStrategy
│       ├── lfu.py          # LFUEvictionStrategy
│       ├── adaptive.py     # AdaptiveEvictionStrategy
│       └── factory.py      # create_eviction_strategy
│
├── ⚙️ Processing Layer (Procesamiento)
│   ├── quantization.py     # Quantizer - Cuantización
│   ├── compression.py     # Compressor - Compresión
│   ├── memory_manager.py  # MemoryManager - Gestión de memoria
│   └── optimizations.py   # FastQuantizer, FastCompressor
│
├── 🔧 Utility Layer (Utilidades)
│   ├── device_manager.py  # DeviceManager - Gestión de dispositivos
│   ├── validators.py      # CacheValidator - Validación
│   ├── error_handler.py   # ErrorHandler - Manejo de errores
│   ├── profiler.py        # CacheProfiler - Profiling
│   └── utils.py           # Funciones utilitarias
│
├── 🎨 Adapter Layer (Adaptadores)
│   └── adapters/
│       ├── adaptive_cache.py  # AdaptiveKVCache
│       └── paged_cache.py     # PagedKVCache
│
├── 🚀 Advanced Layer (Avanzado)
│   ├── batch_operations.py      # BatchCacheOperations
│   ├── monitoring.py            # CacheMonitor, CacheMetrics
│   ├── transformers_integration.py  # TransformersKVCache
│   └── persistence.py           # CachePersistence
│
├── 🛠️ Development Layer (Desarrollo)
│   ├── decorators.py      # Decoradores útiles
│   ├── helpers.py         # Funciones helper
│   ├── builders.py         # Builder pattern
│   ├── prelude.py         # Setup y inicialización
│   ├── performance.py     # Análisis de rendimiento
│   ├── testing.py         # Utilidades de testing
│   └── examples.py        # Ejemplos de uso
│
└── 📚 Documentation
    └── *.md               # Documentación completa
```

## 🔄 Flujo de Datos

### 1. Inicialización

```
Config → BaseKVCache → Componentes Modulares
  ↓
  ├─ DeviceManager (resuelve device)
  ├─ CacheStorage (almacenamiento)
  ├─ CacheStatsTracker (estadísticas)
  ├─ CacheValidator (validación)
  ├─ ErrorHandler (errores)
  ├─ CacheProfiler (profiling)
  ├─ Quantizer/Compressor (si habilitados)
  ├─ MemoryManager (memoria)
  └─ EvictionStrategy (evicción)
```

### 2. Operación Forward

```
Input (key, value)
  ↓
CacheValidator.validate() → Validación
  ↓
DeviceManager.transfer() → Transferencia a device
  ↓
CacheStorage.get() → Intentar obtener del cache
  ↓
  ├─ Hit → CacheStatsTracker.record_hit() → Return cached
  └─ Miss → CacheStatsTracker.record_miss()
       ↓
    Quantizer.quantize() → Cuantización (si habilitada)
       ↓
    Compressor.compress() → Compresión (si habilitada)
       ↓
    MemoryManager.should_evict() → Verificar memoria
       ↓
    EvictionStrategy.select_eviction_candidates() → Evicción si necesario
       ↓
    CacheStorage.put() → Almacenar
       ↓
    Return new
```

### 3. Operación Put

```
put(position, key, value)
  ↓
CacheValidator.validate() → Validación
  ↓
ErrorHandler.handle_oom() → Manejo de errores con retry
  ↓
DeviceManager.transfer() → Transferencia optimizada
  ↓
Quantizer.quantize() → Cuantización
  ↓
Compressor.compress() → Compresión
  ↓
MemoryManager.should_evict() → Verificar memoria
  ↓
EvictionStrategy (si necesario) → Evicción
  ↓
CacheStorage.put() → Almacenar
  ↓
CacheStatsTracker.update() → Actualizar estadísticas
```

## 🎨 Patrones de Diseño

### 1. **Strategy Pattern** (Estrategias de Evicción)
- `BaseEvictionStrategy` define la interfaz
- `LRUEvictionStrategy`, `LFUEvictionStrategy`, `AdaptiveEvictionStrategy` implementan
- `Factory Pattern` (`create_eviction_strategy`) crea instancias

### 2. **Factory Pattern** (Creación de Componentes)
- `create_eviction_strategy()` crea estrategias
- `CacheConfigBuilder` construye configuraciones
- Funciones `create_*_config()` para presets

### 3. **Composition Pattern** (Composición de Componentes)
- `BaseKVCache` compone múltiples componentes
- No usa herencia profunda
- Fácil intercambio de implementaciones

### 4. **Observer Pattern** (Estadísticas y Monitoreo)
- `CacheStatsTracker` observa operaciones
- `CacheMonitor` observa métricas
- Callbacks y eventos para extensibilidad

### 5. **Decorator Pattern** (Funcionalidad Adicional)
- Decoradores en `decorators.py`
- Extienden funcionalidad sin modificar clases
- `@profile_cache_operation`, `@retry_on_failure`, etc.

### 6. **Builder Pattern** (Construcción Flexible)
- `CacheConfigBuilder` para configuraciones complejas
- Fluent API encadenable
- Presets para casos comunes

## 🔌 Interfaces y Contratos

### Interfaces Principales

```python
# IQuantizer - Contrato para cuantización
class IQuantizer(ABC):
    def quantize(key, value, dtype) -> TensorPair: ...

# ICompressor - Contrato para compresión
class ICompressor(ABC):
    def compress(key, value, dtype) -> TensorPair: ...

# IStorage - Contrato para almacenamiento
class IStorage(ABC):
    def get(position) -> TensorPair | None: ...
    def put(position, key, value) -> None: ...
    def size() -> int: ...
    def clear() -> None: ...

# IMemoryManager - Contrato para gestión de memoria
class IMemoryManager(ABC):
    def should_evict(cache_size) -> bool: ...
    def collect_garbage() -> None: ...
    def get_memory_stats() -> StatsDict: ...
```

### Protocols

```python
# Para componentes que pueden ser perfilados
class Profilable(Protocol):
    def get_stats() -> StatsDict: ...

# Para componentes que pueden ser monitoreados
class Monitorable(Protocol):
    def update_metrics(stats: StatsDict) -> None: ...
```

## 🧩 Componentes Clave

### BaseKVCache (Orquestador Principal)

**Responsabilidades**:
- Coordinar todos los componentes
- Proporcionar API pública
- Manejar flujo de datos
- Gestionar errores

**Composición**:
```python
BaseKVCache:
  ├─ DeviceManager        # Gestión de dispositivos
  ├─ CacheStorage         # Almacenamiento
  ├─ CacheStatsTracker    # Estadísticas
  ├─ CacheValidator       # Validación
  ├─ ErrorHandler         # Errores
  ├─ CacheProfiler        # Profiling
  ├─ Quantizer?           # Cuantización (opcional)
  ├─ Compressor?          # Compresión (opcional)
  ├─ MemoryManager        # Memoria
  └─ EvictionStrategy     # Evicción
```

### CacheStorage (Almacenamiento)

**Responsabilidades**:
- Thread-safe storage
- Gestión de acceso (times, counts)
- Cálculo de memoria
- Operaciones CRUD

**Diseño**:
- Thread-safe con `threading.Lock`
- Type aliases para claridad
- Métodos optimizados

### MemoryManager (Gestión de Memoria)

**Responsabilidades**:
- Monitoreo de memoria (GPU/CPU)
- Decisión de evicción
- Garbage collection
- Estadísticas de memoria

**Diseño**:
- Usa `IMemoryManager` interface
- Soporta GPU y CPU
- Thresholds configurables

### EvictionStrategy (Estrategias)

**Responsabilidades**:
- Seleccionar entradas para evicción
- Algoritmos LRU, LFU, Adaptive
- Extensible para nuevas estrategias

**Diseño**:
- Strategy Pattern
- Factory para creación
- Base abstracta clara

## 🔄 Flujos de Extensión

### Agregar Nueva Estrategia de Evicción

1. Crear clase heredando de `BaseEvictionStrategy`
2. Implementar `select_eviction_candidates()`
3. Agregar al enum `CacheStrategy`
4. Actualizar factory `create_eviction_strategy()`

### Agregar Nueva Optimización

1. Crear clase implementando interfaz (`IQuantizer`, `ICompressor`)
2. Agregar a `optimizations.py` si es fast version
3. Usar en `BaseKVCache._init_components()`
4. Configurar via `KVCacheConfig`

### Agregar Nuevo Adapter

1. Crear clase heredando de `BaseKVCache`
2. Extender con nueva funcionalidad
3. Agregar a `adapters/`
4. Exportar en `__init__.py`

## 📊 Dependencias entre Capas

```
Foundation → Core → Processing → Utility → Advanced → Development
    ↓          ↓         ↓          ↓          ↓           ↓
  Types    BaseCache  Quantize   Device    Monitoring  Decorators
Constants  Storage   Compress   Validate  Persistence Helpers
Interfaces Strategies Memory    Error     Transformers Builders
Exceptions Factory    Optimize  Profile   Batch       Testing
Config                              Utils
```

## 🎯 Mejores Prácticas Arquitectónicas

1. **Single Responsibility**: Cada módulo una responsabilidad
2. **Dependency Inversion**: Depender de abstracciones, no implementaciones
3. **Open/Closed**: Abierto para extensión, cerrado para modificación
4. **Interface Segregation**: Interfaces pequeñas y específicas
5. **Composition over Inheritance**: Favorecer composición
6. **DRY (Don't Repeat Yourself)**: Reutilizar código común
7. **Separation of Concerns**: Separar responsabilidades claramente

## 🔒 Thread Safety

- `CacheStorage`: Thread-safe con `Lock`
- `CacheStatsTracker`: Thread-safe con `Lock`
- `MemoryManager`: Thread-safe con `Lock`
- `ErrorHandler`: Thread-safe (estadísticas)
- Operaciones críticas protegidas

## 🚀 Optimizaciones Arquitectónicas

1. **Lazy Initialization**: Componentes inicializados bajo demanda
2. **Caching**: Cache de escalas en quantización
3. **Batching**: Operaciones batch cuando es posible
4. **Async Ready**: Estructura preparada para async (future)
5. **Memory Efficient**: Gestión explícita de memoria GPU

## 📈 Escalabilidad

### Horizontal (Más Caches)
- Múltiples instancias independientes
- Sin estado compartido
- Thread-safe por instancia

### Vertical (Más Capacidad)
- Configuración de `max_tokens`
- Estrategias adaptativas
- Compresión y cuantización

### Distribuido (Futuro)
- `enable_distributed` en config
- Backend configurable (`nccl`, `gloo`)
- Preparado para multi-node

## 🎓 Conclusiones

La arquitectura actual es:
- ✅ **Modular**: Componentes independientes
- ✅ **Extensible**: Fácil agregar nuevas funcionalidades
- ✅ **Mantenible**: Separación clara de concerns
- ✅ **Testeable**: Componentes aislados
- ✅ **Performante**: Optimizaciones integradas
- ✅ **Production-Ready**: Thread-safe, error handling, logging

---

**Versión**: 3.3.0  
**Arquitectura**: Modular en Capas  
**Estado**: ✅ Production-Ready



