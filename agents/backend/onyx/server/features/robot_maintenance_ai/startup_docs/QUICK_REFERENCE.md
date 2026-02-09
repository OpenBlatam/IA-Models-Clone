# Guía Rápida de Referencia - Robot Maintenance AI

## 🚀 Estructura del Proyecto

```
robot_maintenance_ai/
├── api/
│   └── maintenance_api.py      # Endpoints FastAPI
├── config/
│   └── maintenance_config.py   # Configuración (OpenRouter, NLP, ML)
├── core/
│   ├── maintenance_tutor.py     # Clase principal (RobotMaintenanceTutor)
│   ├── maintenance_trainer.py   # Clase alternativa (MaintenanceTrainer)
│   ├── nlp_processor.py         # Procesamiento NLP
│   ├── ml_predictor.py         # Predicciones ML
│   └── conversation_manager.py # Gestión de conversaciones
├── utils/
│   ├── helpers.py              # Utilidades
│   ├── retry_handler.py        # Retry con backoff
│   ├── cache_manager.py        # Gestión de caché
│   ├── validators.py           # Validación de inputs
│   ├── metrics.py              # Sistema de métricas
│   └── metrics_decorator.py   # Decorador para métricas
├── examples/
│   └── basic_usage.py          # Ejemplos de uso
├── startup_docs/               # Documentación de inicio
├── main.py                     # Punto de entrada
├── requirements.txt            # Dependencias
└── README.md                   # Documentación completa
```

## 📝 Endpoints Principales

### POST /api/robot-maintenance/ask
Hacer una pregunta de mantenimiento

**Request:**
```json
{
  "question": "¿Cómo cambio el aceite?",
  "robot_type": "robots_industriales",
  "maintenance_type": "lubricacion",
  "sensor_data": {"temperature": 28.5}
}
```

### POST /api/robot-maintenance/procedure
Obtener procedimiento detallado

**Request:**
```json
{
  "procedure": "lubricación",
  "robot_type": "robots_industriales",
  "difficulty": "intermedio"
}
```

### POST /api/robot-maintenance/diagnose
Diagnosticar problema

**Request:**
```json
{
  "symptoms": "Ruidos extraños",
  "robot_type": "robots_industriales",
  "sensor_data": {"temperature": 85.0, "vibration": 6.5}
}
```

### POST /api/robot-maintenance/predict
Predecir mantenimiento con ML

**Request:**
```json
{
  "robot_type": "robots_industriales",
  "sensor_data": {"temperature": 28.5, "runtime_hours": 8500},
  "historical_data": []
}
```

### POST /api/robot-maintenance/checklist
Generar checklist de mantenimiento

**Request:**
```json
{
  "robot_type": "robots_industriales",
  "maintenance_type": "preventivo"
}
```

### GET /api/robot-maintenance/robot-types
Listar tipos de robots soportados

### GET /api/robot-maintenance/metrics
Obtener métricas del sistema

**Response:**
```json
{
  "success": true,
  "data": {
    "uptime_seconds": 3600,
    "total_requests": 150,
    "total_errors": 2,
    "cache_hits": 45,
    "cache_misses": 105,
    "cache_hit_rate": 0.3,
    "endpoints": {
      "ask": {
        "count": 100,
        "errors": 1,
        "avg_duration": 1.2,
        "min_duration": 0.5,
        "max_duration": 3.0,
        "error_rate": 0.01
      }
    }
  }
}
```

### GET /api/robot-maintenance/cache/stats
Obtener estadísticas de caché

**Response:**
```json
{
  "success": true,
  "data": {
    "cache_enabled": true,
    "size": 50,
    "max_size": 1000,
    "ttl": 3600
  }
}
```

### POST /api/robot-maintenance/metrics/reset
Resetear todas las métricas

**Response:**
```json
{
  "success": true,
  "message": "Metrics reset successfully"
}
```

### GET /api/robot-maintenance/conversation/{conversation_id}/export/json
Exportar conversación como JSON

**Response:** Archivo JSON descargable

### GET /api/robot-maintenance/conversation/{conversation_id}/export/csv
Exportar conversación como CSV

**Response:** Archivo CSV descargable

### GET /api/robot-maintenance/conversation/{conversation_id}/report
Generar reporte de mantenimiento desde conversación

**Query Parameters:**
- `robot_type` (opcional): Tipo de robot
- `maintenance_type` (opcional): Tipo de mantenimiento

**Response:**
```json
{
  "success": true,
  "data": {
    "generated_at": "2024-01-01T12:00:00",
    "robot_type": "robots_industriales",
    "maintenance_type": "preventivo",
    "total_messages": 10,
    "summary": {
      "questions_asked": 5,
      "answers_provided": 5,
      "topics_discussed": []
    },
    "conversation": [...]
  }
}
```

### GET /api/robot-maintenance/health
Health check del servicio

## 💻 Uso en Código Python

### Uso Básico

```python
import asyncio
from robot_maintenance_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    # Hacer pregunta
    response = await tutor.ask_maintenance_question(
        question="¿Cómo cambio el aceite?",
        robot_type="robots_industriales"
    )
    print(response["answer"])
    
    await tutor.close()

asyncio.run(main())
```

### Diagnóstico con ML

```python
diagnosis = await tutor.diagnose_problem(
    symptoms="Ruidos extraños y temperatura elevada",
    robot_type="robots_industriales",
    sensor_data={
        "temperature": 85.0,
        "vibration": 6.5,
        "current": 12.3
    }
)
```

### Predicción de Mantenimiento

```python
prediction = await tutor.predict_maintenance_schedule(
    robot_type="robots_industriales",
    sensor_data={
        "temperature": 28.5,
        "vibration": 0.15,
        "runtime_hours": 8500,
        "battery_level": 75.0
    }
)
```

## ⚙️ Configuración

### Variables de Entorno

```bash
OPENROUTER_API_KEY=tu-api-key
OPENROUTER_DEFAULT_MODEL=openai/gpt-4-turbo
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_MAX_TOKENS=3000
```

### Configuración Personalizada

```python
from robot_maintenance_ai import MaintenanceConfig, OpenRouterConfig, NLPConfig, MLConfig

openrouter_config = OpenRouterConfig(
    api_key="tu-api-key",
    default_model="openai/gpt-4-turbo",
    temperature=0.7
)

nlp_config = NLPConfig(
    language="es",
    use_spacy=True,
    use_transformers=True
)

ml_config = MLConfig(
    enable_predictive_maintenance=True,
    enable_anomaly_detection=True
)

config = MaintenanceConfig(
    openrouter=openrouter_config,
    nlp=nlp_config,
    ml=ml_config
)
```

## 🔧 Tipos de Robots Soportados

- `robots_industriales`
- `robots_medicos`
- `robots_servicio`
- `robots_agricolas`
- `maquinaria_cnc`
- `sistemas_automatizados`

## 📊 Categorías de Mantenimiento

- `preventivo`
- `correctivo`
- `predictivo`
- `diagnostico`
- `calibracion`
- `lubricacion`
- `reemplazo_piezas`
- `actualizacion_software`

## 🎯 Niveles de Dificultad

- `principiante`
- `intermedio`
- `avanzado`
- `experto`

## 🛠️ Troubleshooting

### Error: API Key no configurada
```bash
export OPENROUTER_API_KEY="tu-api-key"
```

### Error: Modelo spaCy no encontrado
```bash
python -m spacy download es_core_news_sm
```

### Error: Dependencias faltantes
```bash
pip install -r requirements.txt
```

### Error: Puerto en uso
Cambia el puerto en `main.py` o usa:
```bash
uvicorn main:app --port 8001
```

## 📚 Recursos Adicionales

- **README.md**: Documentación completa
- **examples/basic_usage.py**: Ejemplos de código
- **API Docs**: http://localhost:8000/docs
- **START.md**: Guía de inicio rápido

