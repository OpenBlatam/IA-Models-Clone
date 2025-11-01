# 🧪 Testing End-to-End - Blatam Academy Features

## 🎯 Estrategia de Testing E2E

### Arquitectura de Tests E2E

```
tests/
├── e2e/
│   ├── conftest.py          # Configuración compartida
│   ├── test_api_flows.py    # Flujos de API completos
│   ├── test_cache_flows.py  # Flujos de cache
│   ├── test_integration.py  # Integración entre servicios
│   └── fixtures/
│       ├── test_data.py
│       └── mock_services.py
```

## 🔧 Setup de E2E Testing

### conftest.py

```python
import pytest
import asyncio
from httpx import AsyncClient
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_cache_engine():
    """Create test cache engine."""
    config = KVCacheConfig(
        max_tokens=4096,
        cache_strategy=CacheStrategy.ADAPTIVE,
        enable_persistence=False  # Sin persistencia en tests
    )
    engine = UltraAdaptiveKVCacheEngine(config)
    yield engine
    await engine.cleanup()

@pytest.fixture
async def api_client():
    """Create API test client."""
    async with AsyncClient(base_url="http://localhost:8002") as client:
        yield client
```

## 📝 Tests de Flujos Completos

### Test de Flujo de API Completo

```python
import pytest

@pytest.mark.asyncio
async def test_complete_query_flow(api_client, test_cache_engine):
    """Test flujo completo de query."""
    # 1. Enviar query inicial
    response = await api_client.post("/api/query", json={
        "query": "What is AI?",
        "context": "technology"
    })
    assert response.status_code == 200
    initial_result = response.json()
    assert "result" in initial_result
    
    # 2. Verificar cache miss (primera vez)
    cache_stats = await api_client.get("/api/stats")
    assert cache_stats.json()["cache_misses"] > 0
    
    # 3. Enviar misma query (debe ser cache hit)
    response2 = await api_client.post("/api/query", json={
        "query": "What is AI?",
        "context": "technology"
    })
    assert response2.status_code == 200
    
    # 4. Verificar cache hit
    cache_stats2 = await api_client.get("/api/stats")
    assert cache_stats2.json()["cache_hits"] > cache_stats.json()["cache_hits"]
```

### Test de Batch Processing

```python
@pytest.mark.asyncio
async def test_batch_processing_flow(api_client):
    """Test procesamiento de batch completo."""
    # Preparar batch de requests
    requests = [
        {"query": f"Query {i}", "context": "test"}
        for i in range(100)
    ]
    
    # Procesar batch
    response = await api_client.post("/api/batch", json={
        "requests": requests,
        "batch_size": 10
    })
    
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 100
    assert all("result" in r for r in results)
    
    # Verificar estadísticas
    stats = await api_client.get("/api/stats")
    assert stats.json()["total_requests"] >= 100
```

### Test de Cache Persistence

```python
@pytest.mark.asyncio
async def test_cache_persistence_flow(test_cache_engine):
    """Test persistencia de cache."""
    # 1. Procesar request
    request1 = {"query": "test query", "context": "test"}
    result1 = await test_cache_engine.process_request(request1)
    
    # 2. Guardar cache
    await test_cache_engine.save_cache("/tmp/test_cache.pkl")
    
    # 3. Crear nuevo engine y cargar cache
    engine2 = UltraAdaptiveKVCacheEngine(config)
    await engine2.load_cache("/tmp/test_cache.pkl")
    
    # 4. Verificar cache hit
    result2 = await engine2.process_request(request1)
    assert result2["cache_hit"] == True
    assert result2["result"] == result1["result"]
```

## 🔄 Tests de Integración entre Servicios

### Test de API Gateway → BUL Flow

```python
@pytest.mark.asyncio
async def test_api_gateway_to_bul_flow():
    """Test flujo completo API Gateway -> BUL."""
    # 1. Request a API Gateway
    gateway_client = AsyncClient(base_url="http://api-gateway:8000")
    response = await gateway_client.post("/api/v1/query", json={
        "query": "test",
        "api_key": "test-key"
    })
    
    assert response.status_code == 200
    
    # 2. Verificar que llegó a BUL
    stats = await api_client.get("/api/stats")
    assert stats.json()["total_requests"] > 0
```

### Test de Multi-Service Integration

```python
@pytest.mark.asyncio
async def test_multi_service_flow(api_client):
    """Test flujo que involucra múltiples servicios."""
    # 1. Query a BUL
    query_response = await api_client.post("/api/query", json={
        "query": "user profile",
        "user_id": 123
    })
    
    # 2. Verificar que se guardó en DB
    db_response = await api_client.get(f"/api/history/{123}")
    assert db_response.status_code == 200
    
    # 3. Verificar que se cacheó en Redis
    cache_check = await api_client.get("/api/cache/check", params={"key": "user:123"})
    assert cache_check.status_code == 200
```

## 🎭 Tests con Mocks y Fixtures

### Fixtures de Test Data

```python
# fixtures/test_data.py
import pytest

@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        {"query": "What is AI?", "context": "technology"},
        {"query": "How does ML work?", "context": "technology"},
        {"query": "Explain neural networks", "context": "technology"}
    ]

@pytest.fixture
def mock_external_api(monkeypatch):
    """Mock external API calls."""
    async def mock_call(*args, **kwargs):
        return {"result": "mocked response"}
    
    monkeypatch.setattr("external_api.call", mock_call)
    return mock_call
```

## 🚀 Performance Testing E2E

### Load Testing

```python
import asyncio
import time

@pytest.mark.asyncio
async def test_load_handling(api_client):
    """Test manejo de carga."""
    async def send_request(query_id):
        response = await api_client.post("/api/query", json={
            "query": f"Query {query_id}",
            "context": "test"
        })
        return response.status_code == 200
    
    # Enviar 1000 requests concurrentes
    start = time.time()
    tasks = [send_request(i) for i in range(1000)]
    results = await asyncio.gather(*tasks)
    
    duration = time.time() - start
    success_rate = sum(results) / len(results)
    
    assert success_rate > 0.95  # 95% success rate
    assert duration < 60  # Completar en menos de 60s
```

### Stress Testing

```python
@pytest.mark.asyncio
async def test_stress_handling(api_client):
    """Test de stress."""
    # Enviar requests continuamente por 5 minutos
    end_time = time.time() + 300
    
    errors = 0
    requests = 0
    
    while time.time() < end_time:
        try:
            response = await api_client.post("/api/query", json={
                "query": "stress test",
                "context": "test"
            })
            if response.status_code != 200:
                errors += 1
            requests += 1
        except:
            errors += 1
        
        await asyncio.sleep(0.1)
    
    error_rate = errors / requests if requests > 0 else 0
    assert error_rate < 0.01  # Menos del 1% de errores
```

## ✅ Checklist de E2E Testing

### Setup
- [ ] Configuración de test environment
- [ ] Fixtures compartidas
- [ ] Mock services configurados
- [ ] Test data preparado

### Tests
- [ ] Flujos completos de API
- [ ] Integración entre servicios
- [ ] Cache persistence
- [ ] Error handling
- [ ] Performance bajo carga

### CI/CD
- [ ] E2E tests en pipeline
- [ ] Tests ejecutándose en cada PR
- [ ] Resultados reportados

---

**Más información:**
- [Testing Guide](bulk/core/TESTING_GUIDE.md)
- [CI/CD Setup](CI_CD_SETUP.md)
- [Performance Tuning](PERFORMANCE_TUNING.md)

