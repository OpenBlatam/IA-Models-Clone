# 🔄 Especificación de Polyglot Core - Optimization Core

## 📋 Resumen

Este documento especifica el módulo `polyglot_core` que proporciona una API unificada para acceder a múltiples backends (Rust, C++, Go, etc.) de manera transparente.

## 🎯 Objetivos

1. **API Unificada**: Una sola API Python para todos los backends
2. **Auto-Selección**: Selección automática del mejor backend disponible
3. **Transparencia**: El usuario no necesita saber qué backend se usa
4. **Flexibilidad**: Permite forzar un backend específico si se desea
5. **Fallback Automático**: Fallback a Python si otros backends no están disponibles

## 🏗️ Arquitectura

### Diagrama de Flujo

```
Usuario
  │
  ├─> KVCache() [backend=AUTO]
  │     │
  │     ├─> get_available_backends()
  │     ├─> get_best_backend("kv_cache")
  │     ├─> Backend.RUST seleccionado
  │     └─> PyKVCache (Rust) creado
  │
  ├─> Compressor() [backend=AUTO]
  │     │
  │     ├─> get_best_backend("compression")
  │     ├─> Backend.RUST seleccionado
  │     └─> PyCompressor (Rust) creado
  │
  └─> Attention() [backend=AUTO]
        │
        ├─> get_best_backend("attention")
        ├─> Backend.CPP seleccionado
        └─> FlashAttention (C++) creado
```

## 📦 Componentes Principales

### Backend Detection

**Especificación**:

```python
class Backend(Enum):
    """Backend enumeration."""
    AUTO = "auto"
    RUST = "rust"
    CPP = "cpp"
    GO = "go"
    JULIA = "julia"
    SCALA = "scala"
    ELIXIR = "elixir"
    PYTHON = "python"

@dataclass
class BackendInfo:
    """Information about a backend."""
    name: str
    available: bool
    version: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    performance_score: float = 0.0  # 0.0-1.0
    error: Optional[str] = None

def check_rust_backend() -> BackendInfo:
    """
    Check if Rust backend is available.
    
    Returns:
        BackendInfo with availability status
    """
    try:
        import truthgpt_rust
        return BackendInfo(
            name="rust",
            available=True,
            version=getattr(truthgpt_rust, "__version__", "unknown"),
            capabilities=["kv_cache", "compression", "tokenization", "data_loading"],
            performance_score=0.95
        )
    except ImportError as e:
        return BackendInfo(
            name="rust",
            available=False,
            error=str(e),
            performance_score=0.0
        )

def check_cpp_backend() -> BackendInfo:
    """
    Check if C++ backend is available.
    
    Returns:
        BackendInfo with availability status
    """
    try:
        import _cpp_core
        return BackendInfo(
            name="cpp",
            available=True,
            version=getattr(_cpp_core, "__version__", "unknown"),
            capabilities=["attention", "cuda_kernels", "inference", "memory_mgmt"],
            performance_score=0.98
        )
    except ImportError as e:
        return BackendInfo(
            name="cpp",
            available=False,
            error=str(e),
            performance_score=0.0
        )

def check_go_backend() -> BackendInfo:
    """
    Check if Go backend is available.
    
    Returns:
        BackendInfo with availability status
    """
    try:
        from go_core.client import GoClient
        # Try to connect to Go service
        client = GoClient()
        if client.is_available():
            return BackendInfo(
                name="go",
                available=True,
                version=client.get_version(),
                capabilities=["http_api", "grpc", "messaging", "distributed"],
                performance_score=0.90
            )
    except Exception as e:
        pass
    
    return BackendInfo(
        name="go",
        available=False,
        error="Go service not available",
        performance_score=0.0
    )

def get_available_backends() -> List[BackendInfo]:
    """
    Get list of all available backends.
    
    Returns:
        List of BackendInfo for all backends
    """
    return [
        check_rust_backend(),
        check_cpp_backend(),
        check_go_backend(),
        # Python is always available
        BackendInfo(
            name="python",
            available=True,
            version=sys.version,
            capabilities=["kv_cache", "compression", "attention", "inference"],
            performance_score=0.50
        )
    ]

def is_backend_available(backend: Backend) -> bool:
    """
    Check if a specific backend is available.
    
    Args:
        backend: Backend to check
    
    Returns:
        True if available
    """
    if backend == Backend.PYTHON:
        return True
    
    backends = get_available_backends()
    backend_info = next(
        (b for b in backends if b.name == backend.value),
        None
    )
    
    return backend_info.available if backend_info else False
```

### Backend Selection

**Especificación**:

```python
# Feature to backend preference mapping
FEATURE_BACKEND_MAP = {
    "kv_cache": [Backend.RUST, Backend.CPP, Backend.GO, Backend.PYTHON],
    "compression": [Backend.RUST, Backend.CPP, Backend.GO, Backend.PYTHON],
    "tokenization": [Backend.RUST, Backend.PYTHON],
    "attention": [Backend.CPP, Backend.RUST, Backend.PYTHON],
    "cuda_kernels": [Backend.CPP, Backend.PYTHON],
    "inference": [Backend.CPP, Backend.RUST, Backend.PYTHON],
    "http_api": [Backend.GO, Backend.RUST, Backend.PYTHON],
    "grpc": [Backend.GO, Backend.RUST, Backend.PYTHON],
    "messaging": [Backend.GO, Backend.ELIXIR, Backend.PYTHON],
    "distributed": [Backend.GO, Backend.SCALA, Backend.PYTHON],
}

def get_best_backend(feature: str) -> Backend:
    """
    Get best available backend for a feature.
    
    Args:
        feature: Feature name (e.g., "kv_cache", "attention")
    
    Returns:
        Best available backend for the feature
    
    Raises:
        ValueError: If feature not supported
    """
    if feature not in FEATURE_BACKEND_MAP:
        raise ValueError(f"Unknown feature: {feature}")
    
    preferred_backends = FEATURE_BACKEND_MAP[feature]
    available_backends = get_available_backends()
    
    # Find first available backend in preference order
    for backend in preferred_backends:
        backend_info = next(
            (b for b in available_backends if b.name == backend.value),
            None
        )
        
        if backend_info and backend_info.available:
            return backend
    
    # Fallback to Python (always available)
    return Backend.PYTHON

def select_backend(
    feature: str,
    preferred: Optional[Backend] = None
) -> Backend:
    """
    Select backend for a feature.
    
    Args:
        feature: Feature name
        preferred: Preferred backend (None = auto-select)
    
    Returns:
        Selected backend
    
    Raises:
        BackendNotAvailableError: If preferred backend not available
    """
    if preferred is None or preferred == Backend.AUTO:
        return get_best_backend(feature)
    
    if not is_backend_available(preferred):
        raise BackendNotAvailableError(
            f"Backend {preferred} not available for feature {feature}"
        )
    
    return preferred
```

### Unified Components

#### KVCache

**Especificación**:

```python
class KVCache:
    """
    Unified KV Cache interface.
    
    Automatically selects best backend or allows manual selection.
    """
    
    def __init__(
        self,
        max_size: int = 8192,
        backend: Backend = Backend.AUTO,
        **kwargs
    ):
        """
        Initialize KV Cache.
        
        Args:
            max_size: Maximum cache size
            backend: Backend to use (AUTO = best available)
            **kwargs: Backend-specific parameters
        """
        self.max_size = max_size
        
        # Select backend
        if backend == Backend.AUTO:
            backend = get_best_backend("kv_cache")
        
        self.backend = backend
        
        # Create backend-specific implementation
        self._impl = self._create_impl(backend, max_size, **kwargs)
    
    def _create_impl(
        self,
        backend: Backend,
        max_size: int,
        **kwargs
    ) -> Any:
        """Create backend-specific implementation."""
        if backend == Backend.RUST:
            from rust_core import PyKVCache
            return PyKVCache(max_size=max_size, **kwargs)
        
        elif backend == Backend.CPP:
            from cpp_core import KVCache as CppKVCache
            return CppKVCache(max_size=max_size, **kwargs)
        
        elif backend == Backend.GO:
            from go_core.client import GoKVCacheClient
            return GoKVCacheClient(max_size=max_size, **kwargs)
        
        else:  # Python fallback
            from polyglot_core.cache import PythonKVCache
            return PythonKVCache(max_size=max_size, **kwargs)
    
    def put(
        self,
        layer_idx: int,
        position: int,
        data: bytes
    ) -> None:
        """
        Put data into cache.
        
        Args:
            layer_idx: Layer index
            position: Position in sequence
            data: Data to cache
        """
        self._impl.put(layer_idx, position, data)
    
    def get(
        self,
        layer_idx: int,
        position: int
    ) -> Optional[bytes]:
        """
        Get data from cache.
        
        Args:
            layer_idx: Layer index
            position: Position in sequence
        
        Returns:
            Cached data or None if not found
        """
        return self._impl.get(layer_idx, position)
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._impl.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics:
            {
                "size": int,
                "max_size": int,
                "hits": int,
                "misses": int,
                "hit_rate": float,
                "memory_usage_bytes": int
            }
        """
        return self._impl.get_stats()
```

#### Compressor

**Especificación**:

```python
class Compressor:
    """
    Unified compression interface.
    
    Supports LZ4, Zstd, and other algorithms.
    """
    
    def __init__(
        self,
        algorithm: str = "lz4",
        backend: Backend = Backend.AUTO,
        **kwargs
    ):
        """
        Initialize compressor.
        
        Args:
            algorithm: Compression algorithm (lz4, zstd, gzip)
            backend: Backend to use (AUTO = best available)
            **kwargs: Backend-specific parameters
        """
        self.algorithm = algorithm
        
        if backend == Backend.AUTO:
            backend = get_best_backend("compression")
        
        self.backend = backend
        self._impl = self._create_impl(backend, algorithm, **kwargs)
    
    def _create_impl(
        self,
        backend: Backend,
        algorithm: str,
        **kwargs
    ) -> Any:
        """Create backend-specific implementation."""
        if backend == Backend.RUST:
            from rust_core import PyCompressor
            return PyCompressor(algorithm=algorithm, **kwargs)
        
        elif backend == Backend.CPP:
            from cpp_core import Compressor as CppCompressor
            return CppCompressor(algorithm=algorithm, **kwargs)
        
        else:  # Python fallback
            from polyglot_core.compression import PythonCompressor
            return PythonCompressor(algorithm=algorithm, **kwargs)
    
    def compress(self, data: bytes) -> bytes:
        """
        Compress data.
        
        Args:
            data: Data to compress
        
        Returns:
            Compressed data
        """
        return self._impl.compress(data)
    
    def decompress(self, data: bytes) -> bytes:
        """
        Decompress data.
        
        Args:
            data: Compressed data
        
        Returns:
            Decompressed data
        """
        return self._impl.decompress(data)
```

#### Attention

**Especificación**:

```python
class Attention:
    """
    Unified attention interface.
    
    Supports Flash Attention and standard attention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        backend: Backend = Backend.AUTO,
        **kwargs
    ):
        """
        Initialize attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            backend: Backend to use (AUTO = best available)
            **kwargs: Backend-specific parameters
        """
        self.d_model = d_model
        self.n_heads = n_heads
        
        if backend == Backend.AUTO:
            backend = get_best_backend("attention")
        
        self.backend = backend
        self._impl = self._create_impl(backend, d_model, n_heads, **kwargs)
    
    def _create_impl(
        self,
        backend: Backend,
        d_model: int,
        n_heads: int,
        **kwargs
    ) -> Any:
        """Create backend-specific implementation."""
        if backend == Backend.CPP:
            from cpp_core import FlashAttention
            return FlashAttention(
                d_model=d_model,
                n_heads=n_heads,
                **kwargs
            )
        
        elif backend == Backend.RUST:
            from rust_core import PyAttention
            return PyAttention(
                d_model=d_model,
                n_heads=n_heads,
                **kwargs
            )
        
        else:  # Python fallback
            from polyglot_core.attention import PythonAttention
            return PythonAttention(
                d_model=d_model,
                n_heads=n_heads,
                **kwargs
            )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute attention.
        
        Args:
            q: Query tensor [batch, seq_len, d_model]
            k: Key tensor [batch, seq_len, d_model]
            v: Value tensor [batch, seq_len, d_model]
            **kwargs: Additional parameters
        
        Returns:
            Attention output [batch, seq_len, d_model]
        """
        return self._impl.forward(q, k, v, **kwargs)
```

## 🔄 Fallback Mechanism

**Especificación**:

```python
def create_with_fallback(
    component_type: str,
    preferred_backends: List[Backend],
    **kwargs
) -> Any:
    """
    Create component with automatic fallback.
    
    Args:
        component_type: Type of component
        preferred_backends: List of preferred backends (in order)
        **kwargs: Component-specific parameters
    
    Returns:
        Component instance
    
    Raises:
        ComponentCreationError: If all backends fail
    """
    errors = []
    
    for backend in preferred_backends:
        if not is_backend_available(backend):
            continue
        
        try:
            return _create_component(component_type, backend, **kwargs)
        except Exception as e:
            errors.append((backend, str(e)))
            logger.warning(
                f"Failed to create {component_type} with {backend}: {e}"
            )
            continue
    
    # Try Python fallback
    try:
        return _create_component(component_type, Backend.PYTHON, **kwargs)
    except Exception as e:
        errors.append((Backend.PYTHON, str(e)))
        raise ComponentCreationError(
            f"Failed to create {component_type} with all backends. Errors: {errors}"
        )
```

## 📊 Usage Examples

### Auto-Selection

```python
# Automatically selects best available backend
cache = KVCache(max_size=8192)
compressor = Compressor(algorithm="lz4")
attention = Attention(d_model=768, n_heads=12)
```

### Manual Selection

```python
# Force specific backend
cache = KVCache(max_size=8192, backend=Backend.RUST)
compressor = Compressor(algorithm="lz4", backend=Backend.CPP)
```

### Check Availability

```python
# Check available backends
backends = get_available_backends()
for backend in backends:
    print(f"{backend.name}: {backend.available}")

# Check specific backend
if is_backend_available(Backend.RUST):
    cache = KVCache(backend=Backend.RUST)
```

## 🧪 Testing

### Test Backend Detection

```python
def test_backend_detection():
    """Test backend detection."""
    backends = get_available_backends()
    assert len(backends) > 0
    assert any(b.name == "python" and b.available for b in backends)
```

### Test Auto-Selection

```python
def test_auto_selection():
    """Test automatic backend selection."""
    cache = KVCache(max_size=1024)
    assert cache.backend in [Backend.RUST, Backend.CPP, Backend.GO, Backend.PYTHON]
```

### Test Fallback

```python
def test_fallback():
    """Test fallback mechanism."""
    # Force unavailable backend, should fallback to Python
    if not is_backend_available(Backend.RUST):
        cache = KVCache(max_size=1024, backend=Backend.AUTO)
        assert cache.backend == Backend.PYTHON
```

---

**Versión**: 1.0.0  
**Última actualización**: Enero 2025




