# 📚 Referencia Completa de API - Ultra Adaptive KV Cache Engine

## 📋 Tabla de Contenidos

- [Clases Principales](#clases-principales)
- [Configuración](#configuración)
- [Métodos Principales](#métodos-principales)
- [Advanced Features](#advanced-features)
- [Utilidades](#utilidades)

## 🏗️ Clases Principales

### BaseKVCache

Base class para todas las implementaciones de KV Cache.

```python
class BaseKVCache(nn.Module):
    """Base KV cache implementation."""
    
    def __init__(self, config: KVCacheConfig):
        """Initialize cache.
        
        Args:
            config: Cache configuration
        """
```

**Métodos Públicos:**
- `store(key: torch.Tensor, value: torch.Tensor) -> None`
- `retrieve(key: torch.Tensor) -> Optional[torch.Tensor]`
- `clear() -> None`
- `get_stats() -> Dict[str, Any]`

### AdaptiveKVCache

Cache adaptativo que ajusta estrategia automáticamente.

```python
class AdaptiveKVCache(BaseKVCache):
    """Adaptive cache with automatic strategy adjustment."""
    
    def adjust_strategy(self) -> None:
        """Ajusta estrategia basado en patrones de uso."""
    
    def get_current_strategy(self) -> CacheStrategy:
        """Obtiene estrategia actual."""
```

### UltraAdaptiveKVCacheEngine

Engine principal con características avanzadas.

```python
class UltraAdaptiveKVCacheEngine:
    """Ultra-adaptive KV cache engine with advanced features."""
    
    def __init__(self, config: KVCacheConfig):
        """Initialize engine.
        
        Args:
            config: Engine configuration
        
        Raises:
            ValueError: Si la configuración es inválida
        """
```

## ⚙️ Configuración

### KVCacheConfig

```python
@dataclass
class KVCacheConfig:
    """Configuración completa del KV Cache."""
    
    # Core
    num_heads: int = 8
    head_dim: int = 64
    max_tokens: int = 4096
    block_size: int = 128
    
    # Strategy
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    cache_mode: CacheMode = CacheMode.INFERENCE
    
    # Optimization
    use_compression: bool = True
    compression_ratio: float = 0.3
    use_quantization: bool = False
    quantization_bits: int = 8
    
    # Memory
    max_memory_mb: Optional[int] = None
    enable_gc: bool = True
    gc_threshold: float = 0.8
    
    # Performance
    pin_memory: bool = True
    non_blocking: bool = True
    dtype: torch.dtype = torch.float16
    
    # Advanced
    enable_persistence: bool = False
    persistence_path: Optional[str] = None
    enable_prefetch: bool = True
    prefetch_size: int = 4
    enable_profiling: bool = False
    enable_distributed: bool = False
    distributed_backend: str = "nccl"
    multi_tenant: bool = False
    tenant_isolation: bool = True
```

### CacheStrategy

```python
class CacheStrategy(Enum):
    """Estrategias de cache disponibles."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    PAGED = "paged"
    COMPRESSED = "compressed"
    QUANTIZED = "quantized"
```

### CacheMode

```python
class CacheMode(Enum):
    """Modos de operación del cache."""
    TRAINING = "training"
    INFERENCE = "inference"
    BULK = "bulk"
    STREAMING = "streaming"
    INTERACTIVE = "interactive"
```

## 🔧 Métodos Principales

### UltraAdaptiveKVCacheEngine

#### process_request

```python
async def process_request(
    self,
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """Procesa un request.
    
    Args:
        request: Request dict con:
            - text: Texto a procesar
            - priority: Prioridad (opcional)
            - session_id: ID de sesión (opcional)
    
    Returns:
        Dict con:
            - result: Resultado del procesamiento
            - cache_hit: Si fue cache hit
            - latency: Latencia en ms
            - metadata: Metadata adicional
    
    Raises:
        ValueError: Si request es inválido
        RuntimeError: Si hay error en procesamiento
    
    Example:
        >>> engine = UltraAdaptiveKVCacheEngine(config)
        >>> result = await engine.process_request({
        ...     'text': 'Hello world',
        ...     'priority': 1
        ... })
        >>> print(result['cache_hit'])
    """
```

#### process_batch_optimized

```python
async def process_batch_optimized(
    self,
    requests: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """Procesa batch de requests optimizado.
    
    Args:
        requests: Lista de requests
        batch_size: Tamaño del batch
    
    Returns:
        Lista de resultados
    
    Example:
        >>> requests = [
        ...     {'text': f'Query {i}', 'priority': 1}
        ...     for i in range(100)
        ... ]
        >>> results = await engine.process_batch_optimized(requests, batch_size=20)
    """
```

#### get_stats

```python
def get_stats(self) -> Dict[str, Any]:
    """Obtiene estadísticas del engine.
    
    Returns:
        Dict con métricas:
            - total_requests: Total de requests
            - cache_hits: Cache hits
            - cache_misses: Cache misses
            - hit_rate: Tasa de hits
            - avg_latency: Latencia promedio
            - p50_latency: Latencia P50
            - p95_latency: Latencia P95
            - p99_latency: Latencia P99
            - memory_usage: Uso de memoria
            - gpu_utilization: Utilización de GPU
    
    Example:
        >>> stats = engine.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """
```

#### persist

```python
def persist(self, path: Optional[str] = None) -> bool:
    """Persiste el cache a disco.
    
    Args:
        path: Ruta personalizada (opcional)
    
    Returns:
        True si fue exitoso
    
    Raises:
        IOError: Si hay error de I/O
    
    Example:
        >>> engine.persist()
        >>> # O con ruta personalizada
        >>> engine.persist('/backup/cache.pt')
    """
```

#### load

```python
def load(self, path: Optional[str] = None) -> bool:
    """Carga el cache desde disco.
    
    Args:
        path: Ruta del archivo (opcional)
    
    Returns:
        True si fue exitoso
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        IOError: Si hay error de I/O
    """
```

#### clear_cache

```python
def clear_cache(self) -> None:
    """Limpia todo el cache."""
```

#### optimize

```python
def optimize(
    self,
    optimization_goal: str = "latency"
) -> Dict[str, Any]:
    """Optimiza el cache.
    
    Args:
        optimization_goal: Objetivo ("latency" | "memory" | "throughput")
    
    Returns:
        Dict con resultados de optimización
    
    Example:
        >>> result = engine.optimize("latency")
        >>> print(f"Improvement: {result['improvement_percentage']}%")
    """
```

## 🚀 Advanced Features

### WorkloadPredictor

```python
class WorkloadPredictor:
    """Predice patrones de carga de trabajo."""
    
    def predict_next_workload(self) -> Dict[str, Any]:
        """Predice próxima carga de trabajo."""
    
    def update_pattern(self, pattern: Dict[str, Any]) -> None:
        """Actualiza patrón observado."""
```

### CachePrefetcher

```python
class CachePrefetcher:
    """Prefetch inteligente."""
    
    def prefetch(self, keys: List[torch.Tensor]) -> None:
        """Prefetch keys."""
    
    def get_prefetch_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de prefetch."""
```

### AutoScaler

```python
class AutoScaler:
    """Auto-escalado de recursos."""
    
    def scale_up(self) -> bool:
        """Escala hacia arriba."""
    
    def scale_down(self) -> bool:
        """Escala hacia abajo."""
    
    def get_recommended_scaling(self) -> Dict[str, Any]:
        """Obtiene recomendación de escalado."""
```

### MultiGPULoadBalancer

```python
class MultiGPULoadBalancer:
    """Balanceador de carga multi-GPU."""
    
    def balance_load(self) -> None:
        """Balancea carga entre GPUs."""
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas por GPU."""
```

## 🛠️ Utilidades

### Factory Functions

```python
def create_kv_cache_engine(
    config: Optional[KVCacheConfig] = None,
    strategy: Optional[CacheStrategy] = None
) -> UltraAdaptiveKVCacheEngine:
    """Crea engine con configuración por defecto.
    
    Args:
        config: Configuración personalizada
        strategy: Estrategia específica
    
    Returns:
        Engine configurado
    
    Example:
        >>> engine = create_kv_cache_engine(strategy=CacheStrategy.ADAPTIVE)
    """
```

### Integration Helpers

```python
class TruthGPTIntegration:
    """Integración con TruthGPT."""
    
    @staticmethod
    def create_engine_for_truthgpt(
        max_tokens: int = 16384
    ) -> UltraAdaptiveKVCacheEngine:
        """Crea engine optimizado para TruthGPT."""
```

---

**Más información:**
- [Guía de Desarrollo](DEVELOPMENT_GUIDE.md)
- [Guía de Testing](TESTING_GUIDE.md)
- [README KV Cache](README_ULTRA_ADAPTIVE_KV_CACHE.md)



