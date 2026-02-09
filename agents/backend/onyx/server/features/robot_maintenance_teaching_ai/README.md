# Robot Maintenance Teaching AI

Sistema de IA avanzado para enseñar mantenimiento de robots y máquinas usando OpenRouter, NLP y Machine Learning con las mejores librerías disponibles.

## 🎯 Características

- **Enseñanza Personalizada**: Procedimientos de mantenimiento adaptados al tipo de robot y nivel de dificultad
- **Diagnóstico Inteligente**: Análisis de problemas basado en síntomas usando IA
- **NLP Avanzado**: Procesamiento de lenguaje natural con spaCy y Transformers
- **Predicción ML**: Modelos de machine learning para predecir necesidades de mantenimiento
- **Múltiples Tipos de Robots**: Soporte para robots industriales, de servicio, colaborativos, móviles, médicos y agrícolas
- **Programas de Mantenimiento**: Generación automática de calendarios de mantenimiento
- **API REST**: Endpoints completos para integración
- **Análisis de Componentes**: Explicaciones detalladas de componentes y sus procedimientos
- **Sistema de Caché**: Caché inteligente para respuestas de API, mejorando rendimiento
- **Historial de Conversaciones**: Almacenamiento y recuperación del historial de interacciones
- **Manejo de Errores Robusto**: Retry logic con exponential backoff y manejo de errores mejorado
- **Validación de Entrada**: Validación completa de parámetros de entrada
- **Async Context Manager**: Soporte para gestión automática de recursos
- **Endpoint de Entrenamiento ML**: API para entrenar modelos de machine learning
- **Logging Mejorado**: Sistema de logging completo para debugging y monitoreo

## 📋 Requisitos

- Python 3.8+
- Clave API de OpenRouter (`OPENROUTER_API_KEY`)
- 8GB+ RAM recomendado para modelos NLP/ML

## 🚀 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Instalar modelo de spaCy

```bash
python -m spacy download es_core_news_md
```

### 3. Configurar variables de entorno

```bash
export OPENROUTER_API_KEY="tu-api-key-aqui"
```

O crea un archivo `.env`:

```
OPENROUTER_API_KEY=tu-api-key-aqui
```

## 💻 Uso Básico

### Uso como Módulo Python

#### Uso Básico

```python
import asyncio
from robot_maintenance_teaching_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    # Enseñar procedimiento de mantenimiento
    result = await tutor.teach_maintenance_procedure(
        robot_type="industrial_robot",
        maintenance_type="preventive",
        difficulty="intermediate"
    )
    print(result["content"])
    
    # Diagnosticar problema
    diagnosis = await tutor.diagnose_problem(
        symptoms="El robot hace ruidos extraños",
        robot_type="industrial_robot"
    )
    print(diagnosis["content"])
    
    # Explicar componente
    explanation = await tutor.explain_component(
        component_name="reductor de velocidad",
        robot_type="industrial_robot"
    )
    print(explanation["content"])
    
    # Generar programa de mantenimiento
    schedule = await tutor.generate_maintenance_schedule(
        robot_type="industrial_robot",
        usage_hours=8
    )
    print(schedule["content"])
    
    # Obtener historial de conversaciones
    history = tutor.get_conversation_history(limit=10)
    print(f"Historial: {len(history)} conversaciones")
    
    await tutor.close()

asyncio.run(main())
```

#### Uso con Async Context Manager (Recomendado)

```python
import asyncio
from robot_maintenance_teaching_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    config = MaintenanceConfig()
    
    # Usar async context manager para gestión automática de recursos
    async with RobotMaintenanceTutor(config) as tutor:
        result = await tutor.teach_maintenance_procedure(
            robot_type="industrial_robot",
            maintenance_type="preventive",
            difficulty="intermediate"
        )
        print(result["content"])
        # Los recursos se cierran automáticamente al salir del bloque

asyncio.run(main())
```

### Uso con API REST

1. Inicia el servidor:

```bash
python main.py
```

O usando uvicorn directamente:

```bash
uvicorn api.maintenance_api:app --host 0.0.0.0 --port 8000
```

2. Usa los endpoints:

```bash
# Enseñar procedimiento de mantenimiento
curl -X POST http://localhost:8000/api/teach \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial_robot",
    "maintenance_type": "preventive",
    "difficulty": "intermediate"
  }'

# Diagnosticar problema
curl -X POST http://localhost:8000/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "El robot hace ruidos extraños",
    "robot_type": "industrial_robot"
  }'

# Explicar componente
curl -X POST http://localhost:8000/api/explain-component \
  -H "Content-Type: application/json" \
  -d '{
    "component_name": "reductor de velocidad",
    "robot_type": "industrial_robot"
  }'

# Generar programa de mantenimiento
curl -X POST http://localhost:8000/api/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial_robot",
    "usage_hours": 8,
    "environment": "industrial"
  }'

# Responder pregunta
curl -X POST http://localhost:8000/api/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Con qué frecuencia debo lubricar las juntas?",
    "robot_type": "industrial_robot"
  }'

# Análisis NLP
curl -X POST http://localhost:8000/api/nlp/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "El robot necesita mantenimiento preventivo"
  }'

# Predicción ML
curl -X POST http://localhost:8000/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial_robot",
    "operating_hours": 5000.0,
    "error_count": 3,
    "temperature": 45.0,
    "vibration_level": 0.8,
    "last_maintenance_hours": 200.0
  }'
```

## 📚 Estructura del Proyecto

```
robot_maintenance_teaching_ai/
├── __init__.py
├── README.md
├── main.py
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── maintenance_config.py      # Configuración del sistema
├── core/
│   ├── __init__.py
│   ├── maintenance_tutor.py       # Tutor principal con OpenRouter
│   ├── nlp_processor.py          # Procesador NLP (spaCy + Transformers)
│   └── ml_predictor.py            # Predictor ML (scikit-learn)
├── api/
│   ├── __init__.py
│   └── maintenance_api.py         # Endpoints FastAPI
├── examples/
│   ├── basic_usage.py             # Ejemplos básicos
│   └── nlp_ml_example.py          # Ejemplos NLP/ML
├── ml_models/
│   └── saved_models/              # Modelos ML guardados
├── nlp_utils/
│   └── (utilidades NLP adicionales)
└── data/
    └── conversations/              # Historial de conversaciones
```

## ⚙️ Configuración

### Configuración Básica

```python
from robot_maintenance_teaching_ai import MaintenanceConfig, OpenRouterConfig, MLConfig, NLPConfig

# Configurar OpenRouter
openrouter_config = OpenRouterConfig(
    api_key="tu-api-key",
    default_model="openai/gpt-4-turbo",
    temperature=0.7,
    max_tokens=3000
)

# Configurar ML
ml_config = MLConfig(
    model_type="ensemble",
    prediction_threshold=0.7,
    use_pretrained=True
)

# Configurar NLP
nlp_config = NLPConfig(
    language="es",
    model_name="es_core_news_md",
    use_transformer=True,
    transformer_model="dccuchile/bert-base-spanish-wwm-uncased"
)

# Configuración principal
config = MaintenanceConfig(
    openrouter=openrouter_config,
    ml=ml_config,
    nlp=nlp_config,
    robot_types=["industrial_robot", "service_robot"],
    adaptive_learning=True
)
```

## 🤖 Tipos de Robots Soportados

- **industrial_robot**: Robots industriales (brazos robóticos, robots de soldadura, etc.)
- **service_robot**: Robots de servicio (limpieza, atención al cliente, etc.)
- **collaborative_robot**: Robots colaborativos (cobots)
- **mobile_robot**: Robots móviles (AGV, robots de exploración, etc.)
- **medical_robot**: Robots médicos (quirúrgicos, de rehabilitación, etc.)
- **agricultural_robot**: Robots agrícolas (siembra, cosecha, etc.)

## 🔧 Tipos de Mantenimiento

- **preventive**: Mantenimiento preventivo
- **corrective**: Mantenimiento correctivo
- **predictive**: Mantenimiento predictivo
- **emergency**: Mantenimiento de emergencia
- **scheduled**: Mantenimiento programado
- **condition_based**: Mantenimiento basado en condición

## 📊 Niveles de Dificultad

- **beginner**: Para principiantes, explicaciones básicas
- **intermediate**: Nivel intermedio, procedimientos estándar
- **advanced**: Nivel avanzado, procedimientos complejos
- **expert**: Nivel experto, procedimientos especializados

## 🧠 Tecnologías Utilizadas

### NLP (Procesamiento de Lenguaje Natural)
- **spaCy**: Procesamiento de texto y extracción de entidades
- **Transformers (Hugging Face)**: Modelos BERT para análisis avanzado
- **NLTK**: Herramientas adicionales de NLP
- **Gensim**: Análisis semántico y similitud

### Machine Learning
- **scikit-learn**: Modelos de clasificación y regresión
- **Random Forest**: Para predicción de fallos
- **Gradient Boosting**: Modelos ensemble avanzados
- **NumPy/Pandas**: Procesamiento de datos

### IA y APIs
- **OpenRouter**: Acceso a modelos de IA avanzados (GPT-4, Claude, etc.)
- **FastAPI**: Framework web moderno y rápido
- **httpx**: Cliente HTTP asíncrono

## 📖 Ejemplos

### Ejemplo 1: Enseñanza de Mantenimiento

```python
from robot_maintenance_teaching_ai import RobotMaintenanceTutor

tutor = RobotMaintenanceTutor()

result = await tutor.teach_maintenance_procedure(
    robot_type="industrial_robot",
    maintenance_type="preventive",
    difficulty="intermediate"
)

print(result["content"])
```

### Ejemplo 2: Uso de NLP

```python
from robot_maintenance_teaching_ai.core.nlp_processor import MaintenanceNLPProcessor

nlp = MaintenanceNLPProcessor()

text = "El robot necesita revisión de engranajes y lubricación"
analysis = nlp.process_maintenance_query(text)

print("Entidades:", analysis["entities"])
print("Palabras clave:", analysis["keywords"])
print("Sentimiento:", analysis["sentiment"])
```

### Ejemplo 3: Predicción ML

```python
from robot_maintenance_teaching_ai.core.ml_predictor import MaintenancePredictor

predictor = MaintenancePredictor()

prediction = predictor.predict_maintenance_need(
    robot_type="industrial_robot",
    operating_hours=5000.0,
    error_count=3,
    temperature=45.0,
    vibration_level=0.8,
    last_maintenance_hours=200.0
)

print(f"¿Necesita mantenimiento?: {prediction['needs_maintenance']}")
print(f"Confianza: {prediction['confidence']}")
print(f"Recomendación: {prediction['recommendation']}")
```

## 🔍 Endpoints de la API

### GET `/`
Información básica de la API

### POST `/api/teach`
Enseñar procedimiento de mantenimiento

### POST `/api/diagnose`
Diagnosticar problema de robot

### POST `/api/explain-component`
Explicar componente del robot

### POST `/api/schedule`
Generar programa de mantenimiento

### POST `/api/answer`
Responder pregunta sobre mantenimiento

### POST `/api/nlp/analyze`
Analizar texto con NLP

### POST `/api/ml/predict`
Predecir necesidad de mantenimiento

### POST `/api/ml/train`
Entrenar modelo de machine learning con datos sintéticos o reales

### GET `/api/conversation/history`
Obtener historial de conversaciones

### GET `/api/health`
Health check del sistema con información detallada de componentes

## 🛠️ Desarrollo

### Ejecutar ejemplos

```bash
# Ejemplo básico
python examples/basic_usage.py

# Ejemplo NLP/ML
python examples/nlp_ml_example.py
```

### Ejecutar tests

```bash
# (Agregar tests en el futuro)
pytest tests/
```

## 📝 Notas

- El modelo de spaCy se descarga automáticamente la primera vez
- Los modelos de Transformers se descargan automáticamente
- Los modelos ML se pueden entrenar con datos propios usando el endpoint `/api/ml/train`
- Se recomienda usar GPU para mejor rendimiento con Transformers
- El sistema de caché está habilitado por defecto (configurable en `MaintenanceConfig`)
- El historial de conversaciones se guarda automáticamente en `data/conversations/`
- El sistema incluye retry logic automático con exponential backoff para errores transitorios
- Se recomienda usar el async context manager (`async with`) para gestión automática de recursos

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
- [Documentación de spaCy](https://spacy.io/usage)
- [Documentación de Transformers](https://huggingface.co/docs/transformers)
- [Documentación de scikit-learn](https://scikit-learn.org/stable/)






