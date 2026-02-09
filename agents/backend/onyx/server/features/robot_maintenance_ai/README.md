# Robot Maintenance AI

Sistema de IA para enseñanza de mantenimiento de robots y máquinas que utiliza OpenRouter, NLP y ML con las mejores librerías disponibles.

## 📚 Documentación Rápida

- **[START.md](startup_docs/START.md)** - Inicio rápido del sistema
- **[QUICK_REFERENCE.md](startup_docs/QUICK_REFERENCE.md)** - Referencia rápida para desarrolladores

## 🎯 Características

- **Enseñanza Inteligente**: Sistema tutor que enseña procedimientos de mantenimiento de robots y máquinas
- **Procesamiento de Lenguaje Natural (NLP)**: Usa spaCy y transformers para entender consultas de mantenimiento
- **Machine Learning (ML)**: Predicción de mantenimiento predictivo usando scikit-learn
- **Integración OpenRouter**: Acceso a modelos de lenguaje avanzados
- **Diagnóstico Inteligente**: Análisis de síntomas y recomendaciones
- **Predicción de Mantenimiento**: ML para predecir cuándo se necesita mantenimiento
- **API REST**: Endpoints fáciles de usar para integración
- **Sistema de Caché**: Caché en memoria con TTL y LRU para mejorar rendimiento
- **Retry con Backoff Exponencial**: Reintentos automáticos con backoff exponencial para mayor robustez
- **Validación de Inputs**: Validación completa de todos los inputs con mensajes de error claros
- **Métricas y Monitoreo**: Sistema de métricas para monitorear rendimiento y uso de la API
- **Manejo de Errores Mejorado**: Manejo robusto de errores con códigos HTTP apropiados
- **Rate Limiting**: Sistema de rate limiting para proteger la API (100 req/min por IP)
- **Logging Mejorado**: Sistema de logging configurable con soporte para archivos
- **Tests Unitarios**: Suite completa de tests para validación, caché y rate limiting
- **Documentación API**: Referencia completa de la API con ejemplos
- **Request Logging Middleware**: Middleware para logging automático de todas las peticiones
- **Health Check Detallado**: Health check con información completa del sistema
- **Exportación de Conversaciones**: Exportar conversaciones en JSON y CSV
- **Generación de Reportes**: Generar reportes de mantenimiento desde conversaciones
- **CORS Configurado**: Soporte CORS para integración frontend
- **Docker Support**: Dockerfile y docker-compose para despliegue fácil
- **Configuración YAML**: Soporte para archivos de configuración YAML
- **Scripts de Inicio**: Scripts para iniciar el servidor fácilmente
- **Persistencia de Datos**: Base de datos SQLite para conversaciones y registros
- **WebSockets**: Actualizaciones en tiempo real mediante WebSockets
- **Optimizaciones de Rendimiento**: Decoradores de timing y procesamiento por lotes
- **Utilidades de Seguridad**: Sanitización de inputs y validación mejorada
- **Sistema de Autenticación**: API keys para control de acceso
- **Sistema de Notificaciones**: Notificaciones para alertas y actualizaciones
- **API de Analytics**: Dashboard completo con métricas y estadísticas avanzadas
- **Búsqueda Avanzada**: Búsqueda en conversaciones y registros de mantenimiento
- **Operaciones por Lotes**: Procesamiento eficiente de múltiples operaciones
- **Gestión de Plugins**: API completa para gestión y ejecución de plugins
- **Sistema de Alertas**: Alertas inteligentes basadas en análisis de sensores y ML
- **Recomendaciones Inteligentes**: Sistema de recomendaciones de mantenimiento basado en IA
- **Gestión de Incidencias**: Sistema completo de tickets e incidencias de mantenimiento
- **Comparación y Benchmarking**: Comparación de robots y análisis de rendimiento
- **Reportes Avanzados**: Generación de reportes personalizados (resumen, detallado, predictivo, costos)
- **Aprendizaje Continuo**: Sistema de feedback y mejora continua de modelos ML
- **Dashboard en Tiempo Real**: Dashboard completo con widgets y métricas en tiempo real
- **Sistema de Webhooks**: Integraciones externas mediante webhooks con eventos del sistema
- **Exportación Avanzada**: Exportación en múltiples formatos (JSON, CSV, Excel) con filtros
- **Configuración Dinámica**: Gestión de configuración en tiempo de ejecución con validación
- **Monitoreo Avanzado**: Monitoreo completo del sistema con alertas automáticas y métricas de recursos
- **Sistema de Auditoría**: Registro completo de actividades del sistema con análisis y estadísticas
- **Plantillas de Mantenimiento**: Sistema de plantillas reutilizables para procedimientos de mantenimiento
- **Validación Avanzada**: Validación de datos con reglas personalizables y validación por lotes

## 📋 Requisitos

- Python 3.8+
- Clave API de Open Router (`OPENROUTER_API_KEY`)

## 🚀 Instalación

### Opción 1: Instalación Local

1. Instala las dependencias:

```bash
pip install -r requirements.txt
```

2. Instala el modelo de spaCy en español (opcional pero recomendado):

```bash
python -m spacy download es_core_news_sm
```

3. Configura la variable de entorno:

```bash
export OPENROUTER_API_KEY="tu-api-key-aqui"
```

O crea un archivo `.env`:

```
OPENROUTER_API_KEY=tu-api-key-aqui
```

### Opción 2: Docker (Recomendado para Producción)

Ver [docs/DOCKER.md](docs/DOCKER.md) para instrucciones completas.

**Inicio rápido con Docker Compose:**

```bash
# Copiar configuración de ejemplo
cp config/config.yaml.example config/config.yaml

# Configurar API key
export OPENROUTER_API_KEY="tu-api-key-aqui"

# Iniciar servicio
docker-compose up -d
```

### Opción 3: Scripts de Inicio

**Linux/Mac:**
```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

**Windows:**
```cmd
scripts\start.bat
```

## 💻 Uso Básico

### Uso como Módulo Python

```python
import asyncio
from robot_maintenance_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    # Hacer una pregunta de mantenimiento
    response = await tutor.ask_maintenance_question(
        question="¿Cómo cambio el aceite de un robot industrial?",
        robot_type="industrial"
    )
    
    print(response["answer"])
    
    # Diagnóstico con datos de sensores
    diagnosis = await tutor.diagnose_problem(
        symptoms="El robot hace ruidos extraños",
        robot_type="industrial",
        sensor_data={
            "temperature": 85.0,
            "vibration": 6.5,
            "runtime_hours": 1500
        }
    )
    
    print(diagnosis["answer"])
    print(f"Predicción ML: {diagnosis['ml_prediction']}")
    
    await tutor.close()

asyncio.run(main())
```

### Uso con API REST

1. Inicia el servidor:

```bash
python main.py
```

2. Usa los endpoints:

```bash
# Hacer una pregunta
curl -X POST http://localhost:8000/api/robot-maintenance/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cómo cambio el aceite de un robot industrial?",
    "robot_type": "industrial"
  }'

# Diagnóstico
curl -X POST http://localhost:8000/api/robot-maintenance/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "El robot hace ruidos extraños y vibra mucho",
    "robot_type": "industrial",
    "sensor_data": {
      "temperature": 85.0,
      "vibration": 6.5
    }
  }'

# Predicción de mantenimiento
curl -X POST http://localhost:8000/api/robot-maintenance/predict \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial",
    "sensor_data": {
      "temperature": 75.0,
      "vibration": 4.0,
      "runtime_hours": 800
    }
  }'

# Generar checklist
curl -X POST http://localhost:8000/api/robot-maintenance/checklist \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial",
    "maintenance_type": "preventivo"
  }'
```

## 📚 Estructura del Proyecto

```
robot_maintenance_ai/
├── __init__.py
├── README.md
├── requirements.txt
├── main.py
├── config/
│   ├── __init__.py
│   └── maintenance_config.py      # Configuración del sistema
├── core/
│   ├── __init__.py
│   ├── maintenance_tutor.py       # Clase principal del tutor
│   ├── nlp_processor.py          # Procesador NLP
│   ├── ml_predictor.py           # Predictor ML
│   └── conversation_manager.py   # Gestión de conversaciones
├── api/
│   ├── __init__.py
│   └── maintenance_api.py        # Endpoints FastAPI
├── utils/
│   ├── __init__.py
│   └── helpers.py                # Funciones auxiliares
└── examples/
    └── basic_usage.py            # Ejemplos de uso
```

## ⚙️ Configuración

Puedes personalizar la configuración:

```python
from robot_maintenance_ai import MaintenanceConfig, OpenRouterConfig, NLPConfig, MLConfig

openrouter_config = OpenRouterConfig(
    api_key="tu-api-key",
    default_model="openai/gpt-4-turbo",
    temperature=0.7,
    max_tokens=3000
)

nlp_config = NLPConfig(
    language="es",
    use_spacy=True,
    use_transformers=True
)

ml_config = MLConfig(
    enable_predictive_maintenance=True,
    model_path="ml_models"
)

config = MaintenanceConfig(
    openrouter=openrouter_config,
    nlp=nlp_config,
    ml=ml_config
)
```

## 🤖 Tipos de Robots Soportados

- Industrial
- Colaborativo
- Médico
- Agrícola
- Logística
- Doméstico
- Militar
- Espacial

## 🔧 Tipos de Mantenimiento

- **Preventivo**: Mantenimiento programado regular
- **Correctivo**: Reparación después de falla
- **Predictivo**: Basado en predicción ML
- **Emergencia**: Situaciones críticas
- **Calibración**: Ajuste de parámetros
- **Limpieza**: Mantenimiento de limpieza
- **Lubricación**: Mantenimiento de lubricación
- **Inspección**: Revisión y evaluación

## 📊 Características ML

El sistema incluye:

- **Predicción de Mantenimiento**: Predice cuándo se necesita mantenimiento basado en datos de sensores
- **Detección de Anomalías**: Identifica patrones anómalos en datos de sensores
- **Recomendaciones Inteligentes**: Sugiere acciones basadas en análisis ML
- **Estimación de Tiempo de Falla**: Predice tiempo hasta posible falla

## 🚀 Características Avanzadas

### Sistema de Caché
- Caché en memoria con TTL (Time-To-Live) configurable
- Evicción LRU (Least Recently Used) automática
- Estadísticas de hit/miss rate
- Endpoint `/cache/stats` para consultar estadísticas

### Retry con Backoff Exponencial
- Reintentos automáticos para llamadas a API fallidas
- Backoff exponencial configurable
- Manejo robusto de timeouts y errores de conexión

### Validación de Inputs
- Validación completa de todos los parámetros de entrada
- Mensajes de error claros y descriptivos
- Sanitización de inputs para prevenir inyecciones
- Validación con Pydantic v2

### Métricas y Monitoreo
- Tracking de todas las peticiones API
- Estadísticas de rendimiento (tiempo de respuesta, tasa de errores)
- Métricas de caché (hit rate, miss rate)
- Endpoint `/metrics` para consultar estadísticas en tiempo real

### Rate Limiting
- Límite de 100 requests por minuto por IP (configurable)
- Algoritmo token bucket
- Headers `Retry-After` cuando se excede el límite
- Endpoints `/rate-limit/stats` y `/rate-limit/reset`

### Logging Mejorado
- Sistema de logging configurable
- Soporte para logging a archivo
- Formato estructurado con información detallada
- Configuración mediante variables de entorno

### Manejo de Errores Mejorado
- Códigos HTTP apropiados para diferentes tipos de errores
- Logging detallado para debugging
- Manejo específico de timeouts, errores de conexión y validación
- Tracking de errores en métricas

### Tests Unitarios
- Suite completa de tests para validadores
- Tests para sistema de caché
- Tests para rate limiting
- Fixtures compartidos para testing

### Sistema de Autenticación
- Gestión de API keys
- Validación de tokens
- Permisos por usuario
- Revocación de API keys
- Endpoints protegidos opcionales

### Sistema de Notificaciones
- Notificaciones para eventos de mantenimiento
- Suscripción a tipos de notificaciones
- Gestión de notificaciones por usuario
- Marcado de notificaciones como leídas
- Limpieza de notificaciones

## 🔬 Librerías Utilizadas

### NLP
- **spaCy**: Procesamiento de lenguaje natural
- **transformers**: Modelos de lenguaje avanzados
- **torch**: Framework de deep learning

### ML
- **scikit-learn**: Machine learning clásico
- **pandas**: Manipulación de datos
- **numpy**: Cálculos numéricos
- **joblib**: Serialización de modelos

### API
- **FastAPI**: Framework web moderno
- **httpx**: Cliente HTTP asíncrono
- **uvicorn**: Servidor ASGI

## 🧪 Testing

Ejecuta los tests con:

```bash
# Todos los tests
pytest tests/

# Tests específicos
pytest tests/test_validators.py
pytest tests/test_cache_manager.py
pytest tests/test_rate_limiter.py

# Con cobertura
pytest tests/ --cov=. --cov-report=html
```

Ver `tests/README.md` para más información.

## 📖 Ejemplos

Ver `examples/basic_usage.py` para ejemplos completos de uso.

## 📚 Documentación

- **[API Reference](docs/API_REFERENCE.md)** - Referencia completa de la API
- **[WEBSOCKETS.md](docs/WEBSOCKETS.md)** - Guía de WebSockets
- **[AUTHENTICATION.md](docs/AUTHENTICATION.md)** - Guía de autenticación
- **[DOCKER.md](docs/DOCKER.md)** - Guía de Docker
- **[START.md](startup_docs/START.md)** - Guía de inicio rápido
- **[QUICK_REFERENCE.md](startup_docs/QUICK_REFERENCE.md)** - Referencia rápida

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es parte de Blatam Academy.

## 🆘 Soporte

Para soporte, abre un issue en el repositorio o contacta al equipo de Blatam Academy.

## 🎓 Recursos Adicionales

- [Documentación de OpenRouter](https://openrouter.ai/docs)
- [Documentación de spaCy](https://spacy.io/)
- [Documentación de scikit-learn](https://scikit-learn.org/)

Sistema inteligente de enseñanza de mantenimiento de robots y máquinas industriales que utiliza **OpenRouter**, **NLP** (Procesamiento de Lenguaje Natural) y **ML** (Machine Learning) con las mejores librerías disponibles.

## 🎯 Características

- **Enseñanza de Procedimientos**: Guías paso a paso para mantenimiento de robots y máquinas
- **Diagnóstico Inteligente**: Análisis de problemas usando NLP y ML
- **Explicación de Conceptos**: Enseñanza adaptativa según nivel de dificultad
- **Predicción de Mantenimiento**: ML para predecir necesidades de mantenimiento
- **Procesamiento de Lenguaje Natural**: Análisis de texto con spaCy, NLTK y Transformers
- **Machine Learning**: Clasificación, predicción y detección de anomalías con scikit-learn
- **API REST**: Endpoints FastAPI para integración fácil

## 📋 Requisitos

- Python 3.8+
- Clave API de OpenRouter (`OPENROUTER_API_KEY`)

## 🚀 Instalación

1. Instala las dependencias:

```bash
cd agents/backend/onyx/server/features/robot_maintenance_ai
pip install -r requirements.txt
```

2. Descarga el modelo de spaCy (opcional pero recomendado):

```bash
python -m spacy download es_core_news_sm
```

3. Configura la variable de entorno:

```bash
export OPENROUTER_API_KEY="tu-api-key-aqui"
```

O crea un archivo `.env`:

```
OPENROUTER_API_KEY=tu-api-key-aqui
```

## 💻 Uso Básico

### Uso como Módulo Python

```python
import asyncio
from robot_maintenance_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    config = MaintenanceConfig()
    tutor = MaintenanceTutor(config)
    
    # Enseñar un procedimiento de mantenimiento
    result = await tutor.teach_maintenance_procedure(
        robot_type="robots_articulados",
        maintenance_type="lubricacion",
        difficulty="intermedio"
    )
    
    print(result['procedure'])
    
    # Diagnosticar un problema
    diagnosis = await tutor.diagnose_problem(
        problem_description="El robot hace ruidos extraños",
        robot_type="robots_articulados"
    )
    
    print(diagnosis['diagnosis'])
    
    # Explicar un concepto
    explanation = await tutor.explain_concept(
        concept="calibración de encoders",
        difficulty="avanzado"
    )
    
    print(explanation['explanation'])
    
    # Predecir necesidades de mantenimiento
    prediction = await tutor.predict_maintenance_needs(
        robot_type="robots_articulados",
        features={
            "hours_operating": 8500,
            "days_since_maintenance": 120,
            "error_count": 3
        }
    )
    
    print(prediction['ml_prediction'])
    
    await tutor.close()

asyncio.run(main())
```

### Uso con API REST

1. Inicia el servidor:

```bash
python main.py
```

2. Usa los endpoints:

```bash
# Enseñar procedimiento
curl -X POST http://localhost:8001/api/maintenance/procedure \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "robots_articulados",
    "maintenance_type": "lubricacion",
    "difficulty": "intermedio"
  }'

# Diagnosticar problema
curl -X POST http://localhost:8001/api/maintenance/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "problem_description": "El robot hace ruidos extraños",
    "robot_type": "robots_articulados"
  }'

# Explicar concepto
curl -X POST http://localhost:8001/api/maintenance/explain \
  -H "Content-Type: application/json" \
  -d '{
    "concept": "calibración de encoders",
    "difficulty": "avanzado"
  }'

# Predecir mantenimiento
curl -X POST http://localhost:8001/api/maintenance/predict \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "robots_articulados",
    "features": {
      "hours_operating": 8500,
      "days_since_maintenance": 120,
      "error_count": 3
    }
  }'
```

## 📚 Estructura del Proyecto

```
robot_maintenance_ai/
├── __init__.py
├── README.md
├── main.py
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── maintenance_config.py      # Configuración del sistema
├── core/
│   ├── __init__.py
│   ├── maintenance_tutor.py       # Clase principal del tutor
│   ├── nlp_processor.py          # Procesador NLP (spaCy, NLTK, Transformers)
│   └── ml_predictor.py           # Predictor ML (scikit-learn)
├── api/
│   ├── __init__.py
│   └── maintenance_api.py        # Endpoints FastAPI
├── utils/
│   ├── __init__.py
│   └── helpers.py                # Funciones auxiliares
└── examples/
    └── basic_usage.py            # Ejemplos de uso
```

## ⚙️ Configuración

Puedes personalizar la configuración:

```python
from robot_maintenance_ai import MaintenanceConfig, OpenRouterConfig, NLPConfig, MLConfig

openrouter_config = OpenRouterConfig(
    api_key="tu-api-key",
    default_model="openai/gpt-4-turbo",
    temperature=0.7,
    max_tokens=3000
)

nlp_config = NLPConfig(
    language="es",
    use_spacy=True,
    use_nltk=True,
    use_transformers=True
)

ml_config = MLConfig(
    use_scikit_learn=True,
    enable_prediction=True,
    enable_classification=True,
    enable_anomaly_detection=True
)

config = MaintenanceConfig(
    openrouter=openrouter_config,
    nlp=nlp_config,
    ml=ml_config
)
```

## 🤖 Tipos de Robots Soportados

- Robots Articulados
- Robots SCARA
- Robots Cartesianos
- Robots Delta
- Robots Colaborativos
- Robots Móviles
- Máquinas CNC
- Máquinas Herramienta
- Sistemas Automatizados

## 🔧 Categorías de Mantenimiento

- Preventivo
- Correctivo
- Predictivo
- Lubricación
- Calibración
- Diagnóstico
- Reparación
- Inspección

## 📊 Niveles de Dificultad

- **Básico**: Conceptos fundamentales y explicaciones simples
- **Intermedio**: Procedimientos detallados con pasos claros
- **Avanzado**: Técnicas avanzadas y terminología especializada
- **Experto**: Conocimiento profundo y optimizaciones

## 🧠 Tecnologías Utilizadas

### NLP (Procesamiento de Lenguaje Natural)
- **spaCy**: Procesamiento avanzado de texto y reconocimiento de entidades
- **NLTK**: Tokenización, análisis de sentimientos, extracción de keywords
- **Transformers**: Modelos de lenguaje avanzados para análisis semántico

### ML (Machine Learning)
- **scikit-learn**: Clasificación, predicción y detección de anomalías
- **NumPy/Pandas**: Procesamiento de datos
- **SciPy**: Análisis estadístico

### AI (Inteligencia Artificial)
- **OpenRouter**: Acceso a múltiples modelos de lenguaje (GPT-4, Claude, etc.)

## 📖 Ejemplos

Ver el archivo `examples/basic_usage.py` para ejemplos completos de uso.

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es parte de Blatam Academy.

## 🆘 Soporte

Para soporte, abre un issue en el repositorio o contacta al equipo de Blatam Academy.

## 🔗 Enlaces Útiles

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [spaCy Documentation](https://spacy.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
