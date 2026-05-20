# Tests para bulk_chat

## Estructura de Tests

```
tests/
├── __init__.py
├── conftest.py                    # Configuración de pytest
├── test_api.py                    # Tests de API endpoints
├── test_chat_engine.py            # Tests del chat engine
├── test_cache.py                  # Tests del cache
├── test_transaction_manager.py    # Tests del transaction manager
├── test_saga_orchestrator.py      # Tests del saga orchestrator
├── test_distributed_coordinator.py # Tests del distributed coordinator
├── test_service_mesh.py           # Tests del service mesh
├── test_adaptive_throttler.py     # Tests del adaptive throttler
├── test_backpressure_manager.py   # Tests del backpressure manager
├── test_rate_limiter.py           # Tests del rate limiter
├── test_session_storage.py        # Tests del session storage
├── test_plugins.py                # Tests del plugin manager
├── test_webhooks.py               # Tests del webhook manager
├── test_templates.py              # Tests del template manager
├── test_auth.py                   # Tests del auth manager
├── test_health_monitor.py         # Tests del health monitor
├── test_metrics.py                # Tests del metrics collector
├── test_conversation_analyzer.py  # Tests del conversation analyzer
├── test_exporters.py              # Tests del conversation exporters
├── test_circuit_breaker.py        # Tests del circuit breaker
├── test_task_queue.py             # Tests del task queue
├── test_feature_flags.py          # Tests del feature flags
├── test_validation.py             # Tests del validation engine
├── test_search_engine.py          # Tests del search engine
├── test_message_queue.py          # Tests del message queue
├── test_notifications.py          # Tests del notification manager
├── test_workflow.py               # Tests del workflow engine
├── test_event_system.py           # Tests del event system
├── test_clustering.py             # Tests del cluster manager
├── test_advanced_rate_limiter.py  # Tests del advanced rate limiter
├── test_intelligent_cache.py      # Tests del intelligent cache
├── test_smart_retry.py            # Tests del smart retry manager
├── test_distributed_lock.py       # Tests del distributed lock manager
├── test_queue_manager.py          # Tests del queue manager
├── test_resource_pool.py          # Tests del resource pool
├── test_connection_manager.py     # Tests del connection manager
├── test_batch_processor.py        # Tests del batch processor
├── test_performance_monitor.py    # Tests del performance monitor
├── test_auto_scaler.py            # Tests del auto scaler
├── test_health_checker_v2.py     # Tests del health checker V2
├── test_cache_warmer.py           # Tests del cache warmer
├── test_event_bus.py              # Tests del event bus
├── test_event_scheduler.py        # Tests del event scheduler
├── test_graceful_degradation.py   # Tests del graceful degradation
├── test_load_shedder.py           # Tests del load shedder
├── test_conflict_resolver.py      # Tests del conflict resolver
├── test_state_machine.py          # Tests del state machine manager
├── test_workflow_engine_v2.py     # Tests del workflow engine V2
├── test_adaptive_optimizer.py     # Tests del adaptive optimizer
├── test_feature_toggle.py          # Tests del feature toggle manager
├── test_rate_limiter_v2.py        # Tests del rate limiter V2
├── test_circuit_breaker_v2.py     # Tests del circuit breaker V2
├── test_sentiment_analyzer.py      # Tests del sentiment analyzer
├── test_task_manager.py           # Tests del task manager
├── test_resource_monitor.py       # Tests del resource monitor
├── test_push_notifications.py     # Tests del push notifications
├── test_distributed_sync.py       # Tests del distributed synchronization
├── test_query_analyzer.py         # Tests del query analyzer
├── test_file_manager.py           # Tests del file manager
├── test_data_compression.py       # Tests del data compression
├── test_config_manager.py         # Tests del config manager
├── test_mfa_authentication.py     # Tests del MFA authentication
├── test_user_behavior_analyzer.py # Tests del user behavior analyzer
├── test_event_stream.py           # Tests del event stream
├── test_security_analyzer.py      # Tests del security analyzer
├── test_session_manager.py        # Tests del session manager
├── test_realtime_metrics.py       # Tests del real-time metrics
├── test_integration.py            # Tests de integración
├── run_tests.sh                   # Script para Linux/Mac
└── run_tests.bat                  # Script para Windows
```

## Instalación de Dependencias

```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

O instalar todo desde requirements.txt:
```bash
pip install -r requirements.txt
```

## Estadísticas de Tests

- **Total de archivos de test:** 70+
- **Tests unitarios:** ~500+
- **Tests de integración:** 25+
- **Cobertura objetivo:** 90%+

## Módulos Cubiertos

### ✅ Core Systems
- Chat Engine, Chat Session, Session Storage
- Response Cache, Intelligent Cache
- Metrics Collector, Performance Monitor

### ✅ Advanced Systems  
- Transaction Manager, Saga Orchestrator
- Distributed Coordinator, Service Mesh
- Adaptive Throttler, Backpressure Manager

### ✅ Processing & Queues
- Task Queue, Message Queue, Queue Manager
- Batch Processor, Smart Retry Manager
- Distributed Lock Manager

### ✅ Resource Management
- Resource Pool, Connection Manager
- Auto Scaler, Load Shedder
- Cache Warmer

### ✅ Workflows & State
- Workflow Engine, Workflow Engine V2
- State Machine Manager
- Event System, Event Bus, Event Scheduler

### ✅ Resilience & Recovery
- Circuit Breaker, Graceful Degradation
- Conflict Resolver
- Health Monitor, Health Checker V2

### ✅ API & Endpoints
- REST API Endpoints
- Rate Limiter (basic & advanced)
- Webhooks, Notifications

### ✅ Utilities
- Plugins, Templates, Feature Flags, Feature Toggle
- Validation Engine, Search Engine
- Conversation Analyzer, Exporters
- Clustering, Auth Manager
- Sentiment Analyzer, Task Manager
- File Manager, Data Compression

### ✅ Monitoring & Analysis
- Resource Monitor, Query Analyzer
- Performance Monitor, Health Monitors
- Auto Scaler, Cache Warmer

### ✅ Advanced Features
- Rate Limiter V2, Circuit Breaker V2
- Adaptive Optimizer
- Push Notifications, Distributed Sync
- Config Manager, MFA Authentication
- User Behavior Analyzer, Security Analyzer
- Session Manager, Real-Time Metrics
- Event Stream

## Ejecutar Tests

### Opción 1: Todos los tests
```bash
pytest tests/ -v
```

### Opción 2: Con coverage
```bash
pytest tests/ -v --cov=bulk_chat --cov-report=term-missing
```

### Opción 3: Con reporte HTML
```bash
pytest tests/ --cov=bulk_chat --cov-report=html
# Abrir htmlcov/index.html en el navegador
```

### Opción 4: Tests específicos
```bash
# Solo tests de API
pytest tests/test_api.py -v

# Solo tests de integración
pytest tests/test_integration.py -v -m integration

# Solo tests unitarios
pytest tests/ -v -m unit

# Excluir tests lentos
pytest tests/ -v -m "not slow"
```

### Opción 5: Usar scripts
```bash
# Linux/Mac
chmod +x tests/run_tests.sh
./tests/run_tests.sh

# Windows
tests\run_tests.bat
```

## Markers Disponibles

- `@pytest.mark.asyncio` - Tests asíncronos
- `@pytest.mark.slow` - Tests lentos
- `@pytest.mark.integration` - Tests de integración
- `@pytest.mark.unit` - Tests unitarios

## Ejemplos de Uso

### Ejecutar solo tests rápidos
```bash
pytest tests/ -v -m "not slow"
```

### Ejecutar con más detalle
```bash
pytest tests/ -v -s  # Muestra prints
```

### Ejecutar tests en paralelo
```bash
pytest tests/ -n auto  # Requiere pytest-xdist
```

### Ejecutar tests con profiling
```bash
pytest tests/ --profile
```

## Configuración

El archivo `pytest.ini` contiene la configuración principal:

- **Test discovery**: Busca archivos `test_*.py` y `*_test.py`
- **Coverage**: Configurado para reportar coverage
- **Markers**: Definidos para categorizar tests
- **Async mode**: Configurado para tests asíncronos

## Coverage

Para ver el reporte de coverage:

```bash
# Terminal
pytest tests/ --cov=bulk_chat --cov-report=term-missing

# HTML (recomendado)
pytest tests/ --cov=bulk_chat --cov-report=html
open htmlcov/index.html  # Mac
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

## Troubleshooting

### Error: Module not found
Asegúrate de estar en el directorio correcto y que las dependencias estén instaladas:
```bash
cd agents/backend/onyx/server/features/bulk_chat
pip install -r requirements.txt
```

### Error: Async tests no funcionan
Asegúrate de tener `pytest-asyncio` instalado:
```bash
pip install pytest-asyncio
```

### Tests muy lentos
Usa markers para ejecutar solo tests rápidos:
```bash
pytest tests/ -v -m "not slow"
```

## Contribuir

Al agregar nuevos tests:

1. Crea archivos `test_*.py` en el directorio `tests/`
2. Usa fixtures de `conftest.py` cuando sea posible
3. Marca tests lentos con `@pytest.mark.slow`
4. Marca tests de integración con `@pytest.mark.integration`
5. Asegúrate de que todos los tests pasen antes de hacer commit

## Estructura de un Test

```python
import pytest
from ..core.module import Module

@pytest.fixture
def module():
    """Create module for testing."""
    return Module()

@pytest.mark.asyncio
async def test_module_functionality(module):
    """Test module functionality."""
    result = await module.do_something()
    assert result is not None
```

