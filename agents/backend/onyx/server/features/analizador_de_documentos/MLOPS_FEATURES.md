# Características MLOps - Versión 1.9.0

## 🎯 Nuevas Características MLOps Implementadas

### 1. Sistema de Versionado de Modelos (`ModelVersionManager`)

Gestión completa de versiones de modelos fine-tuned.

**Características:**
- Versionado semántico (1.0.0, 1.1.0, etc.)
- Historial completo de versiones
- Rollback de modelos
- Comparación de versiones
- Gestión de estado (training, ready, deployed, archived, failed)

**Uso:**
```python
from core.model_versioning import get_model_version_manager

manager = get_model_version_manager()

# Registrar nueva versión
version = manager.register_version(
    "1.0.0",
    "models/v1.0.0",
    metrics={"accuracy": 0.95, "f1": 0.93},
    description="Primera versión estable"
)

# Desplegar versión
manager.deploy_version("1.0.0")

# Hacer rollback
manager.rollback("0.9.0")

# Comparar versiones
comparison = manager.compare_versions("1.0.0", "1.1.0")
```

**API:**
```bash
POST /api/analizador-documentos/versions/
GET /api/analizador-documentos/versions/
POST /api/analizador-documentos/versions/{version}/deploy
POST /api/analizador-documentos/versions/{version}/rollback
GET /api/analizador-documentos/versions/{version1}/{version2}/compare
GET /api/analizador-documentos/versions/current
```

### 2. Sistema de MLOps (`MLOpsManager`)

Monitoreo y gestión de modelos en producción.

**Características:**
- Monitoreo de rendimiento en tiempo real
- Detección de drift de modelos
- Health checks automáticos
- Métricas de rendimiento (accuracy, latency, throughput, error rate)
- Alertas de degradación

**Uso:**
```python
from core.mlops import get_mlops_manager

manager = get_mlops_manager()

# Registrar rendimiento
manager.record_performance(
    model_id="model_v1",
    accuracy=0.95,
    latency=0.5,
    throughput=100.0,
    error_rate=0.02
)

# Obtener salud del modelo
health = manager.get_model_health("model_v1")

# Detectar drift
drift_info = manager.detect_drift("model_v1", window_size=100)

# Obtener estadísticas
stats = manager.get_performance_stats("model_v1", hours=24)
```

**API:**
```bash
POST /api/analizador-documentos/mlops/performance
GET /api/analizador-documentos/mlops/health/{model_id}
GET /api/analizador-documentos/mlops/stats/{model_id}
GET /api/analizador-documentos/mlops/drift/{model_id}
POST /api/analizador-documentos/mlops/thresholds/{model_id}
```

### 3. Sistema de A/B Testing (`ABTestingManager`)

Tests A/B para comparar diferentes modelos y configuraciones.

**Características:**
- Creación y gestión de tests A/B
- Asignación de tráfico configurable
- Análisis de resultados
- Determinación automática de ganador
- Tracking de métricas por variante

**Uso:**
```python
from core.ab_testing import get_ab_testing_manager

manager = get_ab_testing_manager()

# Crear test A/B
test = manager.create_test(
    test_id="test_001",
    name="Test Modelo v1 vs v2",
    variants=[
        {
            "variant_id": "v1",
            "model_config": {"model": "bert-base"},
            "traffic_percentage": 50.0
        },
        {
            "variant_id": "v2",
            "model_config": {"model": "roberta-base"},
            "traffic_percentage": 50.0
        }
    ]
)

# Seleccionar variante para request
variant_id = manager.select_variant("test_001")

# Registrar resultado
manager.record_result(
    test_id="test_001",
    variant_id=variant_id,
    success=True,
    latency=0.5,
    accuracy=0.95
)

# Obtener resultados
results = manager.get_test_results("test_001")
```

**API:**
```bash
POST /api/analizador-documentos/ab-testing/tests
GET /api/analizador-documentos/ab-testing/tests/{test_id}/variant
POST /api/analizador-documentos/ab-testing/tests/{test_id}/result
GET /api/analizador-documentos/ab-testing/tests/{test_id}/results
POST /api/analizador-documentos/ab-testing/tests/{test_id}/stop
```

### 4. Sistema de Seguimiento de Costos (`CostTracker`)

Tracking completo de costos de API y modelos.

**Características:**
- Tracking de costos por operación
- Costos por modelo y usuario
- Estimación de costos
- Reportes de costos diarios
- Análisis de costos por período

**Uso:**
```python
from core.cost_tracker import get_cost_tracker

tracker = get_cost_tracker()

# Registrar costo
tracker.record_cost(
    operation="classification",
    tokens_used=1000,
    model="gpt-4",
    user_id="user_123"
)

# Obtener costo total
total = tracker.get_total_cost(start_date="2024-01-01", end_date="2024-01-31")

# Costos por operación
costs_by_op = tracker.get_cost_by_operation()

# Costos por modelo
costs_by_model = tracker.get_cost_by_model()

# Costos diarios
daily_costs = tracker.get_daily_cost(days=7)

# Estimar costo
estimated = tracker.estimate_cost("classification", estimated_tokens=5000)
```

**API:**
```bash
POST /api/analizador-documentos/costs/record
GET /api/analizador-documentos/costs/total
GET /api/analizador-documentos/costs/by-operation
GET /api/analizador-documentos/costs/by-model
GET /api/analizador-documentos/costs/by-user
GET /api/analizador-documentos/costs/daily
GET /api/analizador-documentos/costs/estimate
```

## 📊 Resumen de Endpoints

### Versionado de Modelos
- `POST /api/analizador-documentos/versions/` - Registrar versión
- `GET /api/analizador-documentos/versions/` - Listar versiones
- `POST /api/analizador-documentos/versions/{version}/deploy` - Desplegar versión
- `POST /api/analizador-documentos/versions/{version}/rollback` - Rollback
- `GET /api/analizador-documentos/versions/{version}` - Obtener versión
- `GET /api/analizador-documentos/versions/{version1}/{version2}/compare` - Comparar
- `GET /api/analizador-documentos/versions/current` - Versión actual

### MLOps
- `POST /api/analizador-documentos/mlops/performance` - Registrar rendimiento
- `GET /api/analizador-documentos/mlops/health/{model_id}` - Salud del modelo
- `GET /api/analizador-documentos/mlops/stats/{model_id}` - Estadísticas
- `GET /api/analizador-documentos/mlops/drift/{model_id}` - Detectar drift
- `POST /api/analizador-documentos/mlops/thresholds/{model_id}` - Configurar umbrales

### A/B Testing
- `POST /api/analizador-documentos/ab-testing/tests` - Crear test
- `GET /api/analizador-documentos/ab-testing/tests/{test_id}/variant` - Obtener variante
- `POST /api/analizador-documentos/ab-testing/tests/{test_id}/result` - Registrar resultado
- `GET /api/analizador-documentos/ab-testing/tests/{test_id}/results` - Resultados
- `POST /api/analizador-documentos/ab-testing/tests/{test_id}/stop` - Detener test

### Cost Tracking
- `POST /api/analizador-documentos/costs/record` - Registrar costo
- `GET /api/analizador-documentos/costs/total` - Costo total
- `GET /api/analizador-documentos/costs/by-operation` - Por operación
- `GET /api/analizador-documentos/costs/by-model` - Por modelo
- `GET /api/analizador-documentos/costs/by-user` - Por usuario
- `GET /api/analizador-documentos/costs/daily` - Costos diarios
- `GET /api/analizador-documentos/costs/estimate` - Estimar costo

## 🎯 Estadísticas Finales

- **60+ endpoints API** en 27 grupos
- **31 módulos core** principales
- **Sistema MLOps completo**
- **Versionado de modelos**
- **A/B Testing**
- **Cost Tracking**

---

**Versión**: 1.9.0  
**Estado**: ✅ **SISTEMA MLOPS ENTERPRISE COMPLETO**
















