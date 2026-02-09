# AI Tutor Educacional con Open Router

Sistema de tutoría educacional inteligente que utiliza Open Router para proporcionar asistencia educativa personalizada a estudiantes.

## 🎯 Características

### Funcionalidades Principales
- **Tutoría Personalizada**: Respuestas adaptadas al nivel y estilo de aprendizaje del estudiante
- **Múltiples Materias**: Soporte para matemáticas, ciencias, historia, literatura, programación y más
- **Análisis de Aprendizaje**: Seguimiento del progreso del estudiante y adaptación del contenido
- **Generación de Ejercicios**: Crea ejercicios de práctica personalizados
- **Generación de Quizzes**: Crea quizzes completos con diferentes tipos de preguntas
- **Historial de Conversaciones**: Mantiene contexto de las interacciones previas
- **API REST**: Endpoints fáciles de usar para integración

### Características Avanzadas ⚡
- **Sistema de Cache Inteligente**: Reduce llamadas a la API y mejora tiempos de respuesta
- **Rate Limiting**: Control de velocidad para evitar exceder límites de API
- **Métricas y Analytics**: Seguimiento detallado de uso, rendimiento y costos
- **Múltiples Modelos**: Soporte para diferentes modelos de Open Router
- **Retry Logic**: Reintentos automáticos en caso de errores
- **Análisis de Progreso**: Identificación de fortalezas y debilidades del estudiante
- **Sistema de Reportes**: Generación de reportes completos en JSON, Markdown y HTML
- **Gamificación**: Sistema de badges, puntos, niveles y leaderboards
- **Rachas de Aprendizaje**: Tracking de días consecutivos de estudio
- **Exportación de Datos**: Exporta reportes y estadísticas en múltiples formatos
- **Evaluación Automática**: Sistema inteligente de evaluación de respuestas y quizzes
- **Motor de Recomendaciones**: Recomendaciones personalizadas de aprendizaje
- **Sistema de Notificaciones**: Notificaciones inteligentes para engagement
- **Rutas de Aprendizaje**: Paths estructurados por materia y nivel
- **Feedback Automático**: Feedback instantáneo y personalizado
- **Dashboard de Analytics**: Visualizaciones y estadísticas en tiempo real
- **Sistema de Base de Datos**: Persistencia completa de datos
- **Autenticación y Autorización**: Sistema de usuarios, roles y permisos
- **Backups Automáticos**: Sistema de respaldo de datos
- **Sistema de Webhooks**: Notificaciones de eventos para integraciones
- **Integración con LMS**: Soporte para Moodle, Canvas, Blackboard y más
- **Testing Completo**: Suite de tests con pytest
- **Docker Ready**: Configuración Docker lista para producción
- **Python SDK**: SDK completo para integración fácil
- **Versionado de API**: Soporte para múltiples versiones de API
- **Validación Avanzada**: Sistema robusto de validación de inputs
- **Scripts de Utilidad**: Setup y backup automatizados

## 📋 Requisitos

- Python 3.8+
- Clave API de Open Router (`OPENROUTER_API_KEY`)

## 🚀 Instalación

### Setup Rápido

```bash
# Ejecutar script de setup
python scripts/setup.py

# Actualizar .env con tu API key
# Luego ejecutar
python main.py
```

### Opción 1: Instalación Local

1. Instala las dependencias:

```bash
pip install -r requirements.txt
```

### Opción 2: Docker

1. Construye y ejecuta con Docker Compose:

```bash
docker-compose up -d
```

### Opción 3: Docker Manual

1. Construye la imagen:

```bash
docker build -t ai-tutor-educacional .
```

2. Ejecuta el contenedor:

```bash
docker run -p 8000:8000 -e OPENROUTER_API_KEY=tu-api-key ai-tutor-educacional
```

2. Configura la variable de entorno:

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
from ai_tutor_educacional_openrouter import AITutor, TutorConfig

async def main():
    config = TutorConfig()
    tutor = AITutor(config)
    
    # Hacer una pregunta
    response = await tutor.ask_question(
        question="¿Qué es la fotosíntesis?",
        subject="ciencias",
        difficulty="intermedio"
    )
    
    print(response["answer"])
    
    # Explicar un concepto
    explanation = await tutor.explain_concept(
        concept="derivadas",
        subject="matematicas",
        difficulty="avanzado"
    )
    
    print(explanation["answer"])
    
    # Generar ejercicios
    exercises = await tutor.generate_exercise(
        topic="ecuaciones cuadráticas",
        subject="matematicas",
        difficulty="intermedio",
        num_exercises=5
    )
    
    print(exercises["answer"])
    
    await tutor.close()

asyncio.run(main())
```

### Uso con API REST

1. Inicia el servidor:

```python
from ai_tutor_educacional_openrouter.api.tutor_api import create_tutor_app
import uvicorn

app = create_tutor_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

2. Usa los endpoints:

```bash
# Hacer una pregunta
curl -X POST http://localhost:8000/api/tutor/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué es la fotosíntesis?",
    "subject": "ciencias",
    "difficulty": "intermedio"
  }'

# Explicar un concepto
curl -X POST http://localhost:8000/api/tutor/explain \
  -H "Content-Type: application/json" \
  -d '{
    "concept": "derivadas",
    "subject": "matematicas",
    "difficulty": "avanzado"
  }'

# Generar ejercicios
curl -X POST http://localhost:8000/api/tutor/exercises \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "ecuaciones cuadráticas",
    "subject": "matematicas",
    "difficulty": "intermedio",
    "num_exercises": 5
  }'

# Generar quiz
curl -X POST http://localhost:8000/api/tutor/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "algebra",
    "subject": "matematicas",
    "difficulty": "intermedio",
    "num_questions": 10,
    "question_types": ["multiple_choice", "short_answer"]
  }'

# Obtener métricas
curl http://localhost:8000/api/tutor/metrics

# Limpiar cache
curl -X DELETE http://localhost:8000/api/tutor/cache
```

## 📚 Estructura del Proyecto

```
ai_tutor_educacional_openrouter/
├── __init__.py
├── main.py                       # Punto de entrada del servidor
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── tutor_config.py          # Configuración del sistema
├── core/
│   ├── __init__.py
│   ├── tutor.py                 # Clase principal del tutor (mejorada)
│   ├── conversation_manager.py  # Gestión de conversaciones
│   ├── learning_analyzer.py     # Análisis de aprendizaje
│   ├── cache_manager.py         # Sistema de cache (NUEVO)
│   ├── rate_limiter.py          # Control de velocidad (NUEVO)
│   ├── metrics_collector.py     # Métricas y analytics (NUEVO)
│   └── quiz_generator.py        # Generador de quizzes (NUEVO)
├── api/
│   ├── __init__.py
│   └── tutor_api.py             # Endpoints FastAPI (mejorados)
├── utils/
│   ├── __init__.py
│   └── helpers.py               # Funciones auxiliares
└── examples/
    ├── __init__.py
    ├── basic_usage.py           # Ejemplos de uso básico
    └── api_usage.py             # Ejemplos de API REST
```

## ⚙️ Configuración

Puedes personalizar la configuración creando una instancia de `TutorConfig`:

```python
from ai_tutor_educacional_openrouter import TutorConfig, OpenRouterConfig

openrouter_config = OpenRouterConfig(
    api_key="tu-api-key",
    default_model="openai/gpt-4",
    temperature=0.7,
    max_tokens=2000
)

config = TutorConfig(
    openrouter=openrouter_config,
    subjects=["matematicas", "programacion"],
    adaptive_learning=True,
    provide_exercises=True
)
```

## 🔧 Materias Soportadas

- Matemáticas
- Ciencias
- Historia
- Literatura
- Física
- Química
- Biología
- Programación

## 📊 Niveles de Dificultad

- **Básico**: Conceptos fundamentales y explicaciones simples
- **Intermedio**: Conceptos más complejos con ejemplos
- **Avanzado**: Temas complejos con análisis profundo

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

