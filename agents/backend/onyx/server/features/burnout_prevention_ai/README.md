# Burnout Prevention AI

Sistema de IA para prevención y manejo del burnout laboral, ayudando a identificar signos tempranos y proporcionar estrategias personalizadas de afrontamiento.

## 🎯 Características

- **Evaluación de Burnout**: Análisis completo del riesgo de burnout basado en múltiples factores
- **Chequeo de Bienestar**: Evaluación del estado general de bienestar y recomendaciones
- **Estrategias de Afrontamiento**: Recomendaciones personalizadas para manejar el estrés
- **Chat Conversacional**: Asistente de IA empático y conversacional sobre burnout y bienestar
- **Seguimiento de Progreso**: Análisis del progreso a lo largo del tiempo con insights personalizados
- **Análisis de Tendencias**: Identificación de patrones y predicciones basadas en historial
- **Recursos Educativos**: Biblioteca personalizada de recursos (artículos, videos, podcasts, libros)
- **Planes Personalizados**: Generación de planes estructurados adaptados a cada usuario
- **Integración OpenRouter**: Utiliza modelos avanzados de IA a través de OpenRouter

## 🚀 Instalación

```bash
# Instalación básica (producción)
pip install -r requirements.txt

# Instalación con herramientas de desarrollo
pip install -r requirements-dev.txt

# Instalación mínima (solo core)
pip install -r requirements-minimal.txt

# Configurar variables de entorno
export OPENROUTER_API_KEY="tu-api-key"
export BURNOUT_AI_PORT=8025
```

## ⚙️ Configuración

Crear archivo `.env`:

```env
OPENROUTER_API_KEY=tu-api-key-aqui
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
BURNOUT_AI_HOST=0.0.0.0
BURNOUT_AI_PORT=8025
DEBUG=False
```

## 🏃 Uso

### Iniciar servidor

```bash
python main.py
```

O con uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8025 --reload
```

### Endpoints

#### Evaluar Burnout
```bash
POST /api/v1/assess
```

Body:
```json
{
  "work_hours_per_week": 50,
  "stress_level": 8,
  "sleep_hours_per_night": 5.5,
  "work_satisfaction": 4,
  "physical_symptoms": ["fatiga", "dolores de cabeza"],
  "emotional_symptoms": ["ansiedad", "irritabilidad"],
  "work_environment": "Alta presión, plazos ajustados",
  "additional_context": "Trabajo remoto, múltiples proyectos"
}
```

#### Chequeo de Bienestar
```bash
POST /api/v1/wellness-check
```

Body:
```json
{
  "current_mood": "ansioso y agotado",
  "energy_level": 3,
  "recent_challenges": "Proyecto importante con plazo ajustado",
  "support_system": "Familia y algunos colegas"
}
```

#### Estrategias de Afrontamiento
```bash
POST /api/v1/coping-strategies
```

Body:
```json
{
  "stressor_type": "Sobrecarga de trabajo",
  "current_coping_methods": ["trabajar más horas"],
  "available_time": "30 minutos diarios",
  "preferences": ["ejercicio", "meditación"]
}
```

#### Chat Conversacional
```bash
POST /api/v1/chat
```

Body:
```json
{
  "message": "Me siento muy agotado últimamente, ¿qué puedo hacer?",
  "conversation_history": []
}
```

#### Seguimiento de Progreso
```bash
POST /api/v1/progress
```

Body:
```json
{
  "user_id": "user123",
  "assessment_history": [
    {"date": "2024-01-01", "burnout_score": 75},
    {"date": "2024-01-15", "burnout_score": 65}
  ],
  "goals": ["Reducir horas de trabajo", "Mejorar sueño"],
  "current_status": {"stress_level": 6, "energy_level": 5}
}
```

#### Análisis de Tendencias
```bash
POST /api/v1/trends
```

Body:
```json
{
  "assessments": [
    {"date": "2024-01-01", "burnout_score": 80, "stress_level": 9},
    {"date": "2024-01-08", "burnout_score": 75, "stress_level": 8},
    {"date": "2024-01-15", "burnout_score": 70, "stress_level": 7}
  ],
  "time_period_days": 30
}
```

#### Recursos Educativos
```bash
POST /api/v1/resources
```

Body:
```json
{
  "topic": "Manejo del estrés laboral",
  "level": "intermediate",
  "format_preference": "article"
}
```

#### Plan Personalizado
```bash
POST /api/v1/personalized-plan
```

Body:
```json
{
  "current_situation": {
    "burnout_score": 70,
    "main_stressors": ["Sobrecarga de trabajo", "Falta de límites"]
  },
  "goals": [
    "Reducir burnout score a 50",
    "Establecer límites saludables",
    "Mejorar balance trabajo-vida"
  ],
  "constraints": {
    "available_time": "1 hora diaria",
    "budget": "limitado"
  },
  "preferences": {
    "activities": ["ejercicio", "meditación", "lectura"]
  }
}
```

#### Health Check
```bash
GET /api/v1/health
```

## 📊 Respuestas de Ejemplo

### Evaluación de Burnout
```json
{
  "burnout_risk_level": "high",
  "burnout_score": 75.5,
  "risk_factors": [
    "Horas de trabajo excesivas",
    "Falta de sueño",
    "Alto nivel de estrés"
  ],
  "recommendations": [
    "Establecer límites claros de horario laboral",
    "Priorizar 7-8 horas de sueño",
    "Implementar técnicas de manejo de estrés"
  ],
  "immediate_actions": [
    "Tomar un descanso de 15 minutos ahora",
    "Programar tiempo de recuperación esta semana",
    "Hablar con tu supervisor sobre la carga de trabajo"
  ],
  "long_term_strategies": [
    "Revisar y ajustar expectativas laborales",
    "Desarrollar rutina de autocuidado",
    "Establecer límites saludables"
  ],
  "assessment_date": "2024-01-15T10:30:00"
}
```

### Seguimiento de Progreso
```json
{
  "progress_score": 65.0,
  "trend": "improving",
  "milestones_achieved": [
    "Reducción de 10 puntos en burnout score",
    "Establecimiento de límites de horario",
    "Mejora en horas de sueño"
  ],
  "next_steps": [
    "Mantener límites establecidos",
    "Continuar con rutina de autocuidado",
    "Evaluar carga de trabajo semanal"
  ],
  "insights": "Has mostrado una mejora constante en los últimos 15 días. La reducción en tu burnout score indica que las estrategias implementadas están funcionando. Continúa con el enfoque actual y considera agregar más tiempo para actividades de recuperación.",
  "progress_date": "2024-01-15T10:30:00"
}
```

### Análisis de Tendencias
```json
{
  "overall_trend": "improving",
  "key_metrics": {
    "burnout_score_avg": 72.5,
    "stress_level_trend": "decreasing",
    "improvement_rate": 0.67
  },
  "patterns": [
    "Mejora consistente los fines de semana",
    "Aumento de estrés los lunes",
    "Correlación positiva entre sueño y bienestar"
  ],
  "predictions": {
    "next_week": {
      "expected_score": 68,
      "confidence": "medium"
    },
    "next_month": {
      "expected_score": 55,
      "confidence": "high"
    }
  },
  "recommendations": [
    "Mantener rutina actual de autocuidado",
    "Enfocarse en mejorar calidad de sueño",
    "Planificar mejor los lunes para reducir estrés",
    "Continuar monitoreo semanal"
  ],
  "analysis_date": "2024-01-15T10:30:00"
}
```

### Recursos Educativos
```json
{
  "resources": [
    {
      "title": "Understanding and Preventing Burnout",
      "type": "article",
      "description": "Guía completa sobre burnout y estrategias de prevención",
      "url": "https://example.com/burnout-guide",
      "duration": "15 min lectura"
    },
    {
      "title": "Stress Management Techniques",
      "type": "video",
      "description": "Técnicas prácticas de manejo de estrés",
      "url": "https://example.com/stress-video",
      "duration": "20 min"
    }
  ],
  "learning_path": [
    "1. Entender qué es el burnout",
    "2. Identificar tus factores de riesgo",
    "3. Aprender técnicas de manejo de estrés",
    "4. Implementar estrategias de prevención",
    "5. Monitorear y ajustar continuamente"
  ],
  "key_concepts": [
    "Burnout vs estrés",
    "Síntomas tempranos",
    "Límites saludables",
    "Autocuidado",
    "Balance trabajo-vida"
  ],
  "action_items": [
    "Leer artículo sobre prevención",
    "Practicar una técnica de manejo de estrés",
    "Establecer un límite de horario esta semana"
  ]
}
```

### Plan Personalizado
```json
{
  "plan_name": "Plan de Recuperación y Prevención - 8 Semanas",
  "duration_weeks": 8,
  "weekly_goals": [
    {
      "week": 1,
      "goal": "Establecer límites básicos de horario",
      "actions": [
        "Definir horario de trabajo fijo",
        "Comunicar límites al equipo",
        "Implementar bloqueo de notificaciones después de horas"
      ],
      "focus_area": "Límites laborales"
    },
    {
      "week": 2,
      "goal": "Mejorar calidad de sueño",
      "actions": [
        "Establecer rutina de sueño",
        "Crear ambiente óptimo para dormir",
        "Limitar pantallas antes de dormir"
      ],
      "focus_area": "Recuperación"
    }
  ],
  "daily_actions": [
    "Meditación de 10 minutos",
    "Pausa de 15 minutos cada 2 horas",
    "Ejercicio ligero (caminar 20 min)",
    "Reflexión diaria sobre bienestar",
    "Desconexión completa después de horas laborales"
  ],
  "milestones": [
    {
      "week": 2,
      "milestone": "Límites establecidos y comunicados"
    },
    {
      "week": 4,
      "milestone": "Rutina de sueño mejorada"
    },
    {
      "week": 6,
      "milestone": "Burnout score reducido en 15 puntos"
    },
    {
      "week": 8,
      "milestone": "Sistema de autocuidado establecido"
    }
  ],
  "resources": [
    "App de meditación",
    "Libro sobre límites saludables",
    "Guía de higiene del sueño"
  ],
  "created_date": "2024-01-15T10:30:00"
}
```

## 🏗️ Arquitectura

```
burnout_prevention_ai/
├── main.py                 # Aplicación principal
├── config/                 # Configuración
│   └── app_config.py
├── infrastructure/         # Infraestructura
│   └── openrouter/         # Cliente OpenRouter
│       ├── api_client.py
│       └── openrouter_client.py
├── services/               # Servicios de negocio
│   └── burnout_service.py
├── api/                    # API endpoints
│   └── routes/
│       └── burnout_routes.py
├── schemas.py              # Modelos Pydantic
└── requirements.txt
```

## 🔧 Desarrollo

### Estructura del Código

- **main.py**: Punto de entrada de la aplicación FastAPI
- **config/**: Configuración de la aplicación
- **infrastructure/**: Clientes externos (OpenRouter)
- **services/**: Lógica de negocio
- **api/**: Endpoints de la API
- **schemas.py**: Modelos de datos Pydantic

### Agregar Nuevas Funcionalidades

1. Agregar schema en `schemas.py`
2. Implementar lógica en `services/burnout_service.py`
3. Crear endpoint en `api/routes/burnout_routes.py`

## 📝 Notas

- El servicio utiliza OpenRouter para acceder a modelos de IA avanzados
- Las respuestas son generadas dinámicamente usando Claude 3.5 Sonnet por defecto
- El sistema está diseñado para ser empático, conversacional y no juzgador
- Todas las conversaciones son confidenciales
- El chat adapta su estilo de comunicación al usuario
- Los planes personalizados se generan considerando restricciones y preferencias reales
- El análisis de tendencias identifica patrones para predicciones más precisas
- Los recursos educativos se personalizan según nivel y preferencias del usuario

## 🚀 Mejoras de Rendimiento

El proyecto utiliza librerías optimizadas para mejor rendimiento:

- **orjson**: JSON 2-3x más rápido que la librería estándar
- **uvloop**: Event loop ultra-rápido (Linux/macOS)
- **structlog**: Logging estructurado con JSON
- **httpx**: Cliente HTTP asíncrono moderno
- **tenacity**: Reintentos inteligentes con backoff exponencial

## 📦 Dependencias Principales

- **FastAPI 0.115+**: Framework web moderno y rápido
- **Pydantic 2.9+**: Validación de datos con mejor rendimiento
- **httpx 0.27+**: Cliente HTTP asíncrono
- **structlog**: Logging estructurado para producción
- **prometheus-client**: Métricas para monitoreo
- **slowapi**: Rate limiting opcional

## 🔒 Seguridad

- Las API keys deben almacenarse de forma segura
- Considera implementar autenticación para producción
- Valida todas las entradas del usuario
- Implementa rate limiting para prevenir abuso

## 📚 Recursos

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Burnout Prevention Resources](https://www.who.int/news/item/28-05-2019-burn-out-an-occupational-phenomenon-international-classification-of-diseases)

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte del ecosistema Blatam Academy.

