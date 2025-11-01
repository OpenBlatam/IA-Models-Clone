# 🚀 Ultra-Adaptive KV Cache Engine - Modular Edition

Sistema modular de Key-Value Cache para Transformers y LLMs, completamente refactorizado siguiendo mejores prácticas de PyTorch y desarrollo de software.

## 📦 Estructura Modular

```
kv_cache/
├── __init__.py              # Package exports
├── config.py                 # Configuration classes
├── quantization.py           # Quantization module
├── compression.py            # Compression module
├── memory_manager.py         # Memory management
├── strategies/               # Eviction strategies
│   ├── __init__.py
│   ├── base.py              # Base interface
│   ├── lru.py               # LRU strategy
│   ├── lfu.py               # LFU strategy
│   ├── adaptive.py          # Adaptive strategy
│   └── factory.py           # Strategy factory
└── MODULAR_STRUCTURE.md      # Detailed architecture docs
```

## 🎯 Principios de Diseño

1. **Separación de Responsabilidades**: Cada módulo tiene una única responsabilidad
2. **Composición sobre Herencia**: Uso de composición para flexibilidad
3. **Interfaces Claras**: ABC para estrategias, interfaces bien definidas
4. **Extensibilidad**: Fácil agregar nuevas funcionalidades sin modificar código existente

## 🔧 Módulos Principales

### Config (`config.py`)
- `CacheStrategy`: Estrategias de cache (LRU, LFU, Adaptive)
- `CacheMode`: Modos de operación (Training, Inference, etc.)
- `KVCacheConfig`: Configuración completa con validación

### Quantization (`quantization.py`)
- `Quantizer`: Cuantización INT8/INT4
- Soporte para mixed precision
- Manejo robusto de errores

### Compression (`compression.py`)
- `Compressor`: Compresión SVD/low-rank/sparse
- Múltiples métodos de compresión
- Soporte para mixed precision

### Memory Manager (`memory_manager.py`)
- `MemoryManager`: Gestión de memoria GPU/CPU
- Monitoreo de memoria
- Garbage collection

### Strategies (`strategies/`)
- `BaseEvictionStrategy`: Interface base
- `LRUEvictionStrategy`: Least Recently Used
- `LFUEvictionStrategy`: Least Frequently Used
- `AdaptiveEvictionStrategy`: Combinación adaptativa
- `factory.create_eviction_strategy()`: Factory para crear estrategias

## 📝 Ejemplo de Uso

```python
from kv_cache import KVCacheConfig, CacheStrategy, CacheMode
from kv_cache.quantization import Quantizer
from kv_cache.compression import Compressor
from kv_cache.memory_manager import MemoryManager
from kv_cache.strategies import create_eviction_strategy

# Configuración
config = KVCacheConfig(
    max_tokens=4096,
    cache_strategy=CacheStrategy.ADAPTIVE,
    cache_mode=CacheMode.INFERENCE,
    use_quantization=True,
    use_compression=True,
)

# Componentes modulares
quantizer = Quantizer(bits=8, use_amp=True)
compressor = Compressor(ratio=0.3, method="svd")
memory_manager = MemoryManager(config, device)
eviction_strategy = create_eviction_strategy(config.cache_strategy)

# Usar componentes independientemente
key = torch.randn(1, 8, 128, 64)
value = torch.randn(1, 8, 128, 64)

# Quantize
key_q, value_q = quantizer.quantize(key, value)

# Compress
key_c, value_c = compressor.compress(key_q, value_q)

# Check memory
if memory_manager.should_evict(cache_size=1000):
    memory_manager.collect_garbage()
```

## ✅ Beneficios

1. **Modularidad**: Código organizado en módulos claros
2. **Testabilidad**: Cada módulo puede testearse independientemente
3. **Mantenibilidad**: Cambios localizados, debugging más fácil
4. **Extensibilidad**: Fácil agregar nuevas funcionalidades
5. **Reutilización**: Módulos pueden usarse en otros contextos

## 🔄 Migración desde Versión Monolítica

El código original sigue funcionando. Los módulos modulares están disponibles como una API alternativa más limpia y mantenible.

---

**Versión**: 2.1.0 (Modular Architecture)  
**Documentación**: Ver `MODULAR_STRUCTURE.md` para detalles completos

