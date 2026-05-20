# Guía Rápida de Referencia - Addiction Recovery AI

## 🚀 Inicio Rápido

### Estructura de Archivos Clave

```
addiction_recovery_ai/
├── api/
│   ├── routes/                # Endpoints modulares
│   │   ├── assessment/        # Evaluaciones
│   │   ├── progress/          # Seguimiento de progreso
│   │   ├── relapse/           # Prevención de recaídas
│   │   └── support/           # Soporte y coaching
│   ├── health.py              # Health checks
│   ├── recovery_api_refactored.py # ✅ API principal (canonical)
│   ├── recovery_api.py        # ⚠️ Deprecated (monolithic, 4932+ lines)
│   └── websocket_api.py       # WebSocket
│
├── core/
│   ├── app_factory.py         # Factory para crear la app
│   ├── lifespan.py            # Lifespan manager
│   └── models/                # Modelos de ML
│
├── config/
│   └── app_config.py          # Configuración centralizada
│
├── services/
│   └── functions/             # Funciones puras de negocio
│
├── schemas/                   # Modelos Pydantic
│   ├── assessment.py
│   ├── progress.py
│   └── relapse.py
│
├── middleware/                # Middleware
│   ├── error_handler.py
│   ├── performance.py
│   └── rate_limit.py
│
└── main.py                    # Punto de entrada
```

## 📝 Patrones de Uso

### Endpoints Principales

```python
# Assessment (Evaluación)
POST /api/assessment/create
GET  /api/assessment/{assessment_id}
GET  /api/assessment/user/{user_id}

# Progress (Progreso)
POST /api/progress/entry
GET  /api/progress/user/{user_id}
GET  /api/progress/stats/{user_id}

# Relapse Prevention (Prevención de Recaídas)
POST /api/relapse/risk-assessment
GET  /api/relapse/risk/{user_id}
POST /api/relapse/prevention-plan

# Support (Soporte)
POST /api/support/chat
GET  /api/support/resources
POST /api/support/emergency
```

### Uso de Schemas

```python
from schemas.assessment import AssessmentCreate, AssessmentResponse
from schemas.progress import ProgressEntry, ProgressStats

# Crear evaluación
assessment = AssessmentCreate(
    user_id="user123",
    substance_type="alcohol",
    severity_score=7.5
)

# Crear entrada de progreso
progress = ProgressEntry(
    user_id="user123",
    days_sober=30,
    mood_score=0.8,
    cravings_level=0.3
)
```

### Servicios y Funciones

```python
from services.functions.assessment_functions import (
    calculate_severity_score,
    generate_assessment_report
)

from services.functions.progress_functions import (
    calculate_recovery_stage,
    predict_relapse_risk
)

# Calcular severidad
severity = calculate_severity_score(assessment_data)

# Generar reporte
report = generate_assessment_report(assessment_id)
```

## 🎯 Endpoints Principales

### Assessment API
- `POST /api/assessment/create` - Crear evaluación
- `GET /api/assessment/{id}` - Obtener evaluación
- `GET /api/assessment/user/{user_id}` - Evaluaciones de usuario
- `PUT /api/assessment/{id}` - Actualizar evaluación

### Progress API
- `POST /api/progress/entry` - Registrar progreso
- `GET /api/progress/user/{user_id}` - Historial de progreso
- `GET /api/progress/stats/{user_id}` - Estadísticas
- `GET /api/progress/trends/{user_id}` - Tendencias

### Relapse Prevention API
- `POST /api/relapse/risk-assessment` - Evaluar riesgo
- `GET /api/relapse/risk/{user_id}` - Riesgo actual
- `POST /api/relapse/prevention-plan` - Plan de prevención
- `GET /api/relapse/history/{user_id}` - Historial

### Support API
- `POST /api/support/chat` - Chat de soporte
- `GET /api/support/resources` - Recursos
- `POST /api/support/emergency` - Emergencia
- `GET /api/support/coaching/{user_id}` - Coaching

## 🔧 Configuración

### Variables de Entorno

```env
# Server
HOST=0.0.0.0
PORT=8020
DEBUG=True

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# Redis
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
```

### Configuración de la App

```python
from config.app_config import get_config

config = get_config()
print(config.host)  # 0.0.0.0
print(config.port)  # 8020
```

## 📚 Modelos Principales

### Assessment Models
- `AssessmentCreate` - Crear evaluación
- `AssessmentResponse` - Respuesta de evaluación
- `AssessmentUpdate` - Actualizar evaluación

### Progress Models
- `ProgressEntry` - Entrada de progreso
- `ProgressStats` - Estadísticas
- `ProgressTrend` - Tendencias

### Relapse Models
- `RiskAssessment` - Evaluación de riesgo
- `PreventionPlan` - Plan de prevención
- `RelapseEvent` - Evento de recaída

## 🧪 Testing

### Test de Endpoint

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
```

### Test de Servicio

```python
from services.functions.assessment_functions import calculate_severity_score

def test_severity_calculation():
    data = {"symptoms": [1, 2, 3, 4, 5]}
    score = calculate_severity_score(data)
    assert 0 <= score <= 10
```

## 🚨 Troubleshooting

### Error: Puerto en uso
```bash
# Cambiar puerto
uvicorn main:app --port 8021
```

### Error: Importación fallida
```bash
# Verificar que estás en el directorio correcto
cd agents/backend/onyx/server/features/addiction_recovery_ai
```

### Error: Variables de entorno
```bash
# Verificar .env existe
ls -la .env

# Cargar variables manualmente
export HOST=0.0.0.0
export PORT=8020
```

### Error: Dependencias faltantes
```bash
pip install -r requirements.txt
```

## 📖 Documentación Completa

- `START.md` - Inicio rápido
- `INSTALLATION_GUIDE.md` - Instalación detallada
- `ARCHITECTURE_QUICK_START.md` - Arquitectura
- `API_QUICK_START.md` - Uso de API
- `README.md` - Documentación principal

## 🔗 Enlaces Útiles

- **API Docs**: http://localhost:8020/docs
- **Health Check**: http://localhost:8020/health
- **ReDoc**: http://localhost:8020/redoc

---

**Última actualización**: 2025  
**Versión**: 3.4.0






