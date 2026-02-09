# Guía de Troubleshooting - Arquitectura V8.0

## 📋 Problemas Comunes y Soluciones

### 🔴 Problema: Dependencias Circulares

#### Síntomas
```
ImportError: cannot import name 'X' from partially initialized module
```

#### Causa
Dos módulos se importan mutuamente.

#### Solución

**Opción 1: Mover import dentro de función**
```python
# ❌ Antes: Import al inicio
from features.recommendations import SkincareRecommender

class AnalysisService:
    def __init__(self):
        self.recommender = SkincareRecommender()

# ✅ Después: Import lazy
class AnalysisService:
    def __init__(self):
        from features.recommendations import SkincareRecommender
        self.recommender = SkincareRecommender()
```

**Opción 2: Usar TYPE_CHECKING**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from features.recommendations import SkincareRecommender

class AnalysisService:
    def __init__(self, recommender: 'SkincareRecommender'):
        self.recommender = recommender
```

**Opción 3: Reorganizar dependencias**
- Mover código compartido a `shared/`
- Usar interfaces en lugar de implementaciones concretas

---

### 🔴 Problema: Service Factory No Encuentra Servicio

#### Síntomas
```
ValueError: Service 'recommendations.skincare' not registered
```

#### Causa
Servicio no registrado o registro en orden incorrecto.

#### Solución

**Verificar registro:**
```python
# En features/recommendations/__init__.py
def register_recommendation_services(service_factory):
    service_factory.register(
        name="recommendations.skincare",
        factory=lambda: SkincareRecommender(...),
        dependencies=["product_repository"]  # Verificar que existe
    )
```

**Verificar orden de registro:**
```python
# En composition_root.py
# 1. Registrar adapters primero
ServiceRegistration.register_adapters(...)

# 2. Registrar repositories
ServiceRegistration.register_repositories(...)

# 3. Registrar domain services
ServiceRegistration.register_domain_services(...)

# 4. Registrar feature services (dependen de los anteriores)
register_recommendation_services(service_factory)
```

**Verificar dependencias:**
```python
# Asegurar que dependencias están registradas antes
factory.register("product_repository", ...)
factory.register("recommendations.skincare", ..., dependencies=["product_repository"])
```

---

### 🔴 Problema: Tests Fracasan Después de Migración

#### Síntomas
```
AttributeError: 'Mock' object has no attribute 'X'
```

#### Causa
Mocks no actualizados para nuevas interfaces.

#### Solución

**Actualizar mocks:**
```python
# ❌ Antes: Mock de implementación concreta
mock_repo = Mock(AnalysisRepository)

# ✅ Después: Mock de interfaz
mock_repo = Mock(IAnalysisRepository)
mock_repo.get_by_id.return_value = analysis
```

**Verificar métodos mockeados:**
```python
# Asegurar que todos los métodos necesarios están mockeados
mock_repo = Mock(IAnalysisRepository)
mock_repo.get_by_id = AsyncMock(return_value=analysis)
mock_repo.save = AsyncMock()
```

---

### 🔴 Problema: Performance Degradado

#### Síntomas
- Requests más lentos
- Alto uso de CPU/memoria
- Timeouts frecuentes

#### Causa
- Falta de caching
- N+1 queries
- Procesamiento bloqueante

#### Solución

**Agregar caching:**
```python
# ✅ Agregar cache decorator
from functools import lru_cache

@lru_cache(maxsize=100)
def get_product_recommendations(skin_type: str) -> List[Product]:
    return expensive_calculation(skin_type)
```

**Optimizar queries:**
```python
# ❌ Antes: N+1 queries
for analysis in analyses:
    user = await repository.get_user(analysis.user_id)  # Query por cada análisis

# ✅ Después: Batch query
user_ids = [a.user_id for a in analyses]
users = await repository.get_users_batch(user_ids)  # Una query
user_map = {u.id: u for u in users}
```

**Usar async correctamente:**
```python
# ❌ Antes: Bloqueante
def process_image(image_data: bytes):
    result = cv2.imread(image_data)  # Bloquea
    return result

# ✅ Después: Async con thread pool
async def process_image(image_data: bytes):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        cv2.imread,
        image_data
    )
    return result
```

---

### 🔴 Problema: Circuit Breaker Siempre Abierto

#### Síntomas
```
CircuitBreakerOpenError: Circuit breaker 'X' is OPEN
```

#### Causa
- Threshold muy bajo
- Timeout muy corto
- Servicio realmente fallando

#### Solución

**Ajustar configuración:**
```python
# Aumentar threshold
config = CircuitBreakerConfig(
    failure_threshold=10,  # En lugar de 5
    recovery_timeout=120.0  # En lugar de 60.0
)
```

**Verificar servicio:**
```python
# Verificar que servicio está funcionando
health = await service.health_check()
if health["status"] != "healthy":
    # Servicio realmente tiene problemas
    pass
```

**Resetear circuit breaker:**
```python
# En desarrollo/testing
circuit_breaker.reset()
```

---

### 🔴 Problema: Imports No Funcionan

#### Síntomas
```
ModuleNotFoundError: No module named 'features.analysis'
```

#### Causa
- `__init__.py` faltante
- PYTHONPATH incorrecto
- Estructura de directorios incorrecta

#### Solución

**Verificar `__init__.py`:**
```bash
# Asegurar que existe
ls features/analysis/__init__.py
```

**Verificar PYTHONPATH:**
```python
# En main.py o setup
import sys
sys.path.insert(0, '/path/to/dermatology_ai')
```

**Verificar estructura:**
```
features/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   └── services/
│       ├── __init__.py
│       └── image_analysis.py
```

---

### 🔴 Problema: Memory Leaks

#### Síntomas
- Memoria crece constantemente
- Servidor se vuelve lento
- OOM (Out of Memory) errors

#### Causa
- Cachés sin límite
- Referencias circulares
- Event listeners no removidos

#### Solución

**Limitar tamaño de cache:**
```python
# ✅ Cache con límite
from functools import lru_cache

@lru_cache(maxsize=100)  # Limitar tamaño
def expensive_function(x):
    return compute(x)
```

**Limpiar referencias:**
```python
# ✅ Cleanup en shutdown
async def shutdown():
    # Limpiar caches
    cache.clear()
    
    # Cerrar conexiones
    await pool.close()
    
    # Remover listeners
    event_bus.remove_all_listeners()
```

**Usar weak references:**
```python
# ✅ Para referencias opcionales
import weakref

class Service:
    def __init__(self):
        self._callbacks = weakref.WeakSet()  # No previene GC
```

---

### 🔴 Problema: Tests Lentos

#### Síntomas
- Suite de tests tarda mucho
- Tests individuales lentos

#### Causa
- Tests haciendo I/O real
- Tests no aislados
- Setup/teardown costosos

#### Solución

**Mockear I/O:**
```python
# ❌ Antes: I/O real
async def test_analyze_image():
    image_data = open("test_image.jpg", "rb").read()
    result = await service.analyze_image(image_data)

# ✅ Después: Mock
async def test_analyze_image():
    mock_processor = Mock(IImageProcessor)
    mock_processor.process.return_value = {"score": 75.0}
    service = ImageAnalysisService(mock_processor)
    result = await service.analyze_image(b"fake_data")
```

**Usar fixtures:**
```python
# ✅ Fixtures reutilizables
@pytest.fixture
async def analysis_service():
    mock_processor = Mock(IImageProcessor)
    return ImageAnalysisService(mock_processor)

async def test_analyze(analysis_service):
    result = await analysis_service.analyze_image(b"data")
```

**Paralelizar tests:**
```bash
# Ejecutar tests en paralelo
pytest -n auto
```

---

### 🔴 Problema: Logs Excesivos

#### Síntomas
- Logs muy grandes
- Performance degradado
- Difícil encontrar información

#### Causa
- Logging en loops
- Log level muy bajo
- Información duplicada

#### Solución

**Ajustar log level:**
```python
# ✅ Logging apropiado
logger.debug("Detailed debug info")  # Solo en desarrollo
logger.info("Important event")  # Producción
logger.warning("Potential issue")
logger.error("Error occurred", exc_info=True)
```

**Evitar logging en loops:**
```python
# ❌ Antes: Log por cada iteración
for item in items:
    logger.info(f"Processing {item}")  # Muchos logs

# ✅ Después: Log resumen
logger.info(f"Processing {len(items)} items")
for item in items:
    process(item)
logger.info("Processing complete")
```

**Usar structured logging:**
```python
# ✅ Logging estructurado
logger.info(
    "Analysis completed",
    extra={
        "analysis_id": analysis.id,
        "duration": duration,
        "score": analysis.score
    }
)
```

---

## 🔍 Debugging Tips

### 1. Activar Debug Mode

```python
# En config/settings.py
DEBUG = True
LOG_LEVEL = "DEBUG"
```

### 2. Usar Debugger

```python
# Agregar breakpoint
import pdb; pdb.set_trace()

# O usar debugger del IDE
breakpoint()
```

### 3. Logging Detallado

```python
# Agregar logging temporal
logger.debug(f"Variable value: {variable}")
logger.debug(f"Function called with: {args}, {kwargs}")
```

### 4. Dependency Graph

```python
# Ver dependencias
graph = composition_root.get_dependency_graph()
print(json.dumps(graph, indent=2))
```

### 5. Health Checks

```python
# Verificar salud de servicios
health = await composition_root.health_check()
print(json.dumps(health, indent=2))
```

---

## 📞 Obtener Ayuda

### 1. Revisar Documentación
- `ARCHITECTURE_IMPROVEMENTS_V8.md`
- `ARCHITECTURE_MIGRATION_GUIDE.md`
- `ARCHITECTURE_CODE_EXAMPLES.md`

### 2. Revisar Logs
```bash
# Ver logs recientes
tail -f logs/app.log

# Buscar errores
grep ERROR logs/app.log
```

### 3. Verificar Tests
```bash
# Ejecutar tests relevantes
pytest tests/features/analysis/

# Con verbose
pytest -v tests/features/analysis/
```

### 4. Revisar Métricas
```bash
# Ver métricas de Prometheus
curl http://localhost:8006/metrics
```

---

**Versión:** 1.0.0  
**Fecha:** 2024




