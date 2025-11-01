# 🧪 Guía de Testing - Ultra Adaptive KV Cache Engine

## 📋 Tabla de Contenidos

- [Setup de Testing](#setup-de-testing)
- [Tests Unitarios](#tests-unitarios)
- [Tests de Integración](#tests-de-integración)
- [Tests de Performance](#tests-de-performance)
- [Tests de Carga](#tests-de-carga)
- [CI/CD Integration](#cicd-integration)

## 🔧 Setup de Testing

### Instalación de Dependencias

```bash
pip install pytest pytest-asyncio pytest-cov pytest-benchmark
pip install torch pytest-mock
```

### Estructura de Tests

```
bulk/core/
├── ultra_adaptive_kv_cache_engine.py
├── test_ultra_adaptive_kv_cache.py
└── tests/
    ├── __init__.py
    ├── test_base_cache.py
    ├── test_adaptive_cache.py
    ├── test_engine.py
    ├── test_integration.py
    ├── test_performance.py
    └── fixtures/
        └── cache_fixtures.py
```

## 🧪 Tests Unitarios

### Test Base Cache

```python
# tests/test_base_cache.py
import pytest
import torch
from ultra_adaptive_kv_cache_engine import (
    BaseKVCache,
    KVCacheConfig,
    CacheStrategy
)

@pytest.fixture
def config():
    return KVCacheConfig(
        num_heads=8,
        head_dim=64,
        max_tokens=1024,
        cache_strategy=CacheStrategy.LRU
    )

@pytest.fixture
def cache(config):
    return BaseKVCache(config)

def test_cache_initialization(cache):
    """Test inicialización del cache."""
    assert cache is not None
    assert cache.config.max_tokens == 1024
    assert len(cache.cache) == 0

def test_store_and_retrieve(cache):
    """Test almacenar y recuperar."""
    key = torch.randn(1, 8, 64)
    value = torch.randn(1, 8, 64)
    
    cache.store(key, value)
    retrieved = cache.retrieve(key)
    
    assert retrieved is not None
    assert torch.allclose(retrieved, value)

def test_cache_eviction(cache):
    """Test evicción cuando el cache está lleno."""
    # Llenar cache
    for i in range(1024):
        key = torch.randn(1, 8, 64)
        value = torch.randn(1, 8, 64)
        cache.store(key, value)
    
    # Cache debería haber evictado entradas
    assert len(cache.cache) <= 1024

def test_cache_clear(cache):
    """Test limpiar cache."""
    cache.store(torch.randn(1, 8, 64), torch.randn(1, 8, 64))
    cache.clear()
    assert len(cache.cache) == 0
```

### Test Adaptive Cache

```python
# tests/test_adaptive_cache.py
import pytest
from ultra_adaptive_kv_cache_engine import AdaptiveKVCache, CacheStrategy

def test_adaptive_strategy_adjustment():
    """Test que la estrategia se ajusta automáticamente."""
    config = KVCacheConfig(cache_strategy=CacheStrategy.ADAPTIVE)
    cache = AdaptiveKVCache(config)
    
    # Simular diferentes patrones de acceso
    for i in range(100):
        key = torch.randn(1, 8, 64)
        value = torch.randn(1, 8, 64)
        cache.store(key, value)
        cache.retrieve(key)  # Acceso frecuente
    
    # Verificar que la estrategia se ajustó
    assert cache.current_strategy is not None
```

### Test Engine

```python
# tests/test_engine.py
import pytest
import asyncio
from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine

@pytest.fixture
def engine():
    config = KVCacheConfig(
        max_tokens=4096,
        enable_prefetch=True
    )
    return UltraAdaptiveKVCacheEngine(config)

@pytest.mark.asyncio
async def test_process_request(engine):
    """Test procesamiento de request."""
    request = {
        'text': 'Test query',
        'priority': 1
    }
    
    result = await engine.process_request(request)
    
    assert 'result' in result
    assert 'cache_hit' in result
    assert 'latency' in result

@pytest.mark.asyncio
async def test_concurrent_requests(engine):
    """Test requests concurrentes."""
    requests = [
        {'text': f'Query {i}', 'priority': 1}
        for i in range(100)
    ]
    
    tasks = [engine.process_request(req) for req in requests]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 100
    assert all('result' in r for r in results)

@pytest.mark.asyncio
async def test_cache_hit(engine):
    """Test cache hit."""
    request = {'text': 'Same query', 'priority': 1}
    
    # Primera llamada - cache miss
    result1 = await engine.process_request(request)
    assert result1['cache_hit'] == False
    
    # Segunda llamada - cache hit
    result2 = await engine.process_request(request)
    assert result2['cache_hit'] == True
    assert result2['latency'] < result1['latency']
```

## 🔗 Tests de Integración

### Test End-to-End

```python
# tests/test_integration.py
import pytest
import asyncio
import tempfile
import os
from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig

@pytest.fixture
def temp_cache_dir():
    """Crea directorio temporal para cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.mark.asyncio
async def test_persistence_workflow(temp_cache_dir):
    """Test completo de persistencia."""
    config = KVCacheConfig(
        max_tokens=4096,
        enable_persistence=True,
        persistence_path=temp_cache_dir
    )
    
    # Crear engine
    engine1 = UltraAdaptiveKVCacheEngine(config)
    
    # Procesar requests
    for i in range(10):
        await engine1.process_request({
            'text': f'Query {i}',
            'priority': 1
        })
    
    # Persistir
    engine1.persist()
    
    # Crear nuevo engine y cargar
    engine2 = UltraAdaptiveKVCacheEngine(config)
    engine2.load()
    
    # Verificar que el cache se restauró
    result = await engine2.process_request({
        'text': 'Query 0',
        'priority': 1
    })
    assert result['cache_hit'] == True

@pytest.mark.asyncio
async def test_multi_gpu_integration():
    """Test integración multi-GPU."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Requiere al menos 2 GPUs")
    
    config = KVCacheConfig(
        max_tokens=8192,
        enable_distributed=True
    )
    
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Requests distribuidos
    requests = [
        {'text': f'Query {i}', 'priority': 1}
        for i in range(100)
    ]
    
    results = await asyncio.gather(*[
        engine.process_request(req) for req in requests
    ])
    
    assert len(results) == 100
    # Verificar que se usaron múltiples GPUs
    stats = engine.get_stats()
    assert stats.get('gpu_count', 0) >= 2
```

## ⚡ Tests de Performance

### Benchmark Tests

```python
# tests/test_performance.py
import pytest
import time
import numpy as np
from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_latency_benchmark(benchmark):
    """Benchmark de latencia."""
    config = KVCacheConfig(max_tokens=4096)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    request = {'text': 'Test query', 'priority': 1}
    
    result = benchmark.pedantic(
        lambda: asyncio.run(engine.process_request(request)),
        iterations=1000,
        rounds=10
    )
    
    assert result['latency'] < 100  # ms

@pytest.mark.asyncio
async def test_throughput_benchmark():
    """Benchmark de throughput."""
    config = KVCacheConfig(max_tokens=8192)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    requests = [
        {'text': f'Query {i}', 'priority': 1}
        for i in range(1000)
    ]
    
    start = time.time()
    results = await asyncio.gather(*[
        engine.process_request(req) for req in requests
    ])
    duration = time.time() - start
    
    throughput = len(requests) / duration
    assert throughput > 50  # req/s

@pytest.mark.asyncio
async def test_memory_efficiency():
    """Test eficiencia de memoria."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    config = KVCacheConfig(max_tokens=16384)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Procesar muchos requests
    for i in range(1000):
        await engine.process_request({
            'text': f'Query {i}',
            'priority': 1
        })
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # No debería aumentar más de 2GB
    assert memory_increase < 2048
```

## 📊 Tests de Carga

### Stress Tests

```python
# tests/test_load.py
import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine

@pytest.mark.asyncio
async def test_high_concurrency():
    """Test alta concurrencia."""
    config = KVCacheConfig(max_tokens=16384)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # 1000 requests concurrentes
    requests = [
        {'text': f'Query {i}', 'priority': 1}
        for i in range(1000)
    ]
    
    tasks = [engine.process_request(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verificar que no hay errores
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 0

@pytest.mark.asyncio
async def test_sustained_load():
    """Test carga sostenida."""
    config = KVCacheConfig(max_tokens=8192)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # 10 minutos de carga continua
    duration = 600  # segundos
    start = time.time()
    request_count = 0
    
    while time.time() - start < duration:
        await engine.process_request({
            'text': f'Query {request_count}',
            'priority': 1
        })
        request_count += 1
        
        # Pequeña pausa para evitar saturación
        await asyncio.sleep(0.01)
    
    # Verificar que el sistema sigue funcionando
    stats = engine.get_stats()
    assert stats['total_requests'] == request_count
    assert stats.get('error_count', 0) == 0
```

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=bulk/core --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  
  - repo: local
    hooks:
      - id: tests
        name: tests
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## 📈 Métricas de Testing

### Coverage Goals

- **Unit Tests**: >90%
- **Integration Tests**: >80%
- **Critical Paths**: 100%

### Performance Goals

- **Latency P50**: <100ms
- **Latency P95**: <500ms
- **Throughput**: >50 req/s
- **Memory**: <2GB para 10K requests

---

**Más información:**
- [Guía de Desarrollo](DEVELOPMENT_GUIDE.md)
- [README KV Cache](README_ULTRA_ADAPTIVE_KV_CACHE.md)

