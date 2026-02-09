# Autonomous Long-Term Agent

Agente autónomo que corre continuamente hasta que se detiene explícitamente. Implementa conceptos de investigación sobre agentes autónomos de largo plazo, continual learning y self-initiated learning.

## 📚 Basado en Papers de Investigación

Este agente implementa conceptos de 20+ papers de investigación sobre agentes autónomos:

### Papers Principales (1-10)

1. **WebResearcher: Unleashing unbounded reasoning capability in Long-Horizon Agents** (2025)
   - Long-horizon reasoning con acumulación continua de conocimiento
   - Razonamiento profundo sobre horizontes temporales largos

2. **AI Autonomy: Self-initiated Open-world Continual Learning and Adaptation** (2023)
   - Paradigma SOLA para aprendizaje auto-iniciado
   - Adaptación continua sin retraining manual

3. **Continual Learning for Robotics** (2019)
   - Estrategias de aprendizaje continuo
   - Aprendizaje desde streams de datos cambiantes

4. **Towards Continual Reinforcement Learning** (2022)
   - Adaptación a tareas/entornos cambiantes
   - Retención de conocimiento previo

5. **Learning to Adapt in Dynamic, Real-World Environments** (2018)
   - Adaptación en tiempo real
   - Meta-reinforcement learning

### Papers Nuevos (11-20)

11. **Building Self-Evolving Agents via Experience-Driven Lifelong Learning** (2025)
    - Framework ELL (Experience-driven Lifelong Learning)
    - Crecimiento continuo mediante interacciones reales
    - Abstracción de habilidades recurrentes
    - Internalización de conocimiento

12. **EvoAgent: Agent Autonomous Evolution with Continual World Model** (2025)
    - Continual world model para tareas de largo horizonte
    - Self-planning, self-control, self-reflection
    - Actualización de experiencias sin intervención humana

13. **Lifelong Learning of Large Language Model based Agents** (2025)
    - Roadmap para lifelong learning en agentes LLM
    - Memoria, percepción multimodal, interacción dinámica

14. **A Survey of Self-Evolving Agents** (2025)
    - Paradigma de agentes self-evolving
    - Evolución de componentes (modelo, memoria, herramientas, arquitectura)

15. **The Landscape of Agentic Reinforcement Learning for LLMs** (2025)
    - Agentic RL para LLMs
    - Planificación, memoria, razonamiento multi-paso

Y más... Ver `papers_references.json` para la lista completa.

## 🚀 Características

### Características Principales
- ✅ **Ejecución Continua**: El agente corre indefinidamente hasta detenerse explícitamente
- ✅ **Múltiples Instancias Paralelas**: Soporte para ejecutar varios agentes en paralelo
- ✅ **OpenRouter Integration**: Acceso a 1000+ modelos de IA
- ✅ **Base de Conocimiento Persistente**: Almacenamiento continuo de conocimiento aprendido
- ✅ **Self-Initiated Learning**: El agente aprende y se adapta automáticamente
- ✅ **Long-Horizon Reasoning**: Razonamiento profundo considerando contexto histórico
- ✅ **Continual Learning**: Aprende continuamente sin olvidar conocimiento previo

### Nuevas Características (Papers 11-20)
- ✅ **Self-Reflection Engine** (EvoAgent): Reflexión automática sobre performance, estrategia y capacidades
- ✅ **Experience-Driven Learning** (ELL): Framework de aprendizaje basado en experiencias reales
- ✅ **Continual World Model** (EvoAgent): Modelo del mundo que se actualiza continuamente
- ✅ **Self-Planning**: Planificación autónoma basada en el estado del mundo
- ✅ **Skill Abstraction**: Abstracción automática de habilidades de experiencias recurrentes
- ✅ **Knowledge Internalization**: Internalización de conocimiento en memoria a largo plazo

## 📦 Instalación

```bash
# Instalar dependencias
pip install fastapi uvicorn httpx pydantic pydantic-settings

# Configurar variables de entorno
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_HTTP_REFERER="https://tu-dominio.com"
```

## ⚙️ Configuración

Crea un archivo `.env` o configura variables de entorno:

```env
# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_HTTP_REFERER=https://blatam-academy.com
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_MAX_TOKENS=4000

# Agent Configuration
AGENT_POLL_INTERVAL=1.0
AGENT_MAX_CONCURRENT_TASKS=10
AGENT_MAX_PARALLEL_INSTANCES=5
AGENT_AUTO_RESTART=true

# Learning Configuration
LEARNING_ENABLED=true
LEARNING_ADAPTATION_RATE=0.1
KNOWLEDGE_BASE_RETENTION_DAYS=30
MAX_KNOWLEDGE_ENTRIES=10000

# Self-Reflection Configuration (EvoAgent paper)
ENABLE_SELF_REFLECTION=true
SELF_REFLECTION_INTERVAL=300.0
SELF_REFLECTION_ON_PERFORMANCE=true
SELF_REFLECTION_ON_STRATEGY=true
SELF_REFLECTION_ON_CAPABILITIES=true

# Experience-Driven Learning Configuration (ELL paper)
ENABLE_EXPERIENCE_LEARNING=true
MAX_EXPERIENCES=5000
SKILL_ABSTRACTION_THRESHOLD=3

# Continual World Model Configuration (EvoAgent paper)
ENABLE_WORLD_MODEL=true
WORLD_MODEL_MAX_OBSERVATIONS=1000
WORLD_MODEL_CHANGE_THRESHOLD=0.3

# Server
HOST=0.0.0.0
PORT=8001
```

## 🎯 Uso

### Iniciar el Servidor

```bash
python -m autonomous_long_term_agent.main
```

O usando uvicorn directamente:

```bash
uvicorn autonomous_long_term_agent.main:app --host 0.0.0.0 --port 8001
```

### API Endpoints

#### 1. Iniciar un Agente

```bash
POST /api/v1/agents/start
Content-Type: application/json

{
  "instruction": "Investigar y aprender sobre inteligencia artificial",
  "agent_id": "optional-custom-id"
}
```

#### 2. Iniciar Múltiples Agentes en Paralelo

```bash
POST /api/v1/agents/parallel
Content-Type: application/json

{
  "count": 3,
  "instruction": "Operar autónomamente y aprender continuamente"
}
```

#### 3. Obtener Estado de un Agente

```bash
GET /api/v1/agents/{agent_id}/status
```

Respuesta incluye:
- Estado del agente
- Métricas (tareas completadas, tokens usados, uptime)
- Tamaño de cola de tareas
- Estadísticas de aprendizaje
- Estadísticas de base de conocimiento

#### 4. Agregar Tarea a un Agente

```bash
POST /api/v1/agents/{agent_id}/tasks
Content-Type: application/json

{
  "instruction": "Analizar las tendencias actuales en machine learning",
  "metadata": {
    "priority": "high",
    "category": "research"
  }
}
```

#### 5. Listar Tareas de un Agente

```bash
GET /api/v1/agents/{agent_id}/tasks?status=completed
```

#### 6. Pausar/Reanudar Agente

```bash
POST /api/v1/agents/{agent_id}/pause
POST /api/v1/agents/{agent_id}/resume
```

#### 7. Detener Agente

```bash
POST /api/v1/agents/{agent_id}/stop
```

**⚠️ IMPORTANTE**: El agente NO se detiene automáticamente. Debes llamar explícitamente al endpoint de stop.

#### 8. Listar Todos los Agentes

```bash
GET /api/v1/agents
```

#### 9. Detener Todos los Agentes

```bash
POST /api/v1/agents/stop-all
```

## 🔄 Flujo de Operación

1. **Inicio**: El agente se inicia y comienza su loop principal
2. **Procesamiento Continuo**: 
   - Procesa tareas de la cola
   - Ejecuta operaciones autónomas cuando no hay tareas
   - Aprende y se adapta continuamente
3. **Almacenamiento de Conocimiento**: Todo el conocimiento se guarda en la base de conocimiento persistente
4. **Self-Initiated Learning**: El agente detecta oportunidades de aprendizaje y se adapta automáticamente
5. **Stop Explícito**: Solo se detiene cuando se llama al endpoint de stop

## 🧠 Conceptos Implementados

### Long-Horizon Reasoning

El agente utiliza razonamiento de largo horizonte al:
- Consultar conocimiento histórico relevante antes de responder
- Considerar implicaciones a largo plazo
- Acumular conocimiento continuamente

### Continual Learning

- **Base de Conocimiento Persistente**: Almacena todo el conocimiento aprendido
- **Retención Selectiva**: Mantiene conocimiento relevante según período de retención
- **Búsqueda Contextual**: Recupera conocimiento relevante para cada tarea

### Self-Initiated Learning (SOLA)

- **Detección Automática**: Detecta oportunidades de aprendizaje
- **Adaptación Automática**: Ajusta parámetros basado en performance
- **Event-Driven Learning**: Aprende de eventos y resultados

### Self-Reflection (EvoAgent Paper)

- **Reflection on Performance**: Analiza métricas y tareas recientes
- **Reflection on Strategy**: Evalúa efectividad de estrategias actuales
- **Reflection on Capabilities**: Identifica brechas y habilidades subutilizadas
- **Periodic Reflection**: Reflexión automática cada 5 minutos (configurable)

### Experience-Driven Learning (ELL Paper)

- **Experience Recording**: Registra todas las interacciones reales
- **Skill Abstraction**: Abstrae habilidades de patrones recurrentes
- **Knowledge Internalization**: Internaliza conocimiento en memoria a largo plazo
- **Lifelong Growth**: Crecimiento continuo a lo largo del ciclo de vida del agente

### Continual World Model (EvoAgent Paper)

- **World State Tracking**: Rastrea estados del mundo continuamente
- **Change Detection**: Detecta cambios significativos en el mundo
- **Self-Planning**: Planificación autónoma basada en el estado del mundo
- **Adaptive Strategy**: Estrategia adaptativa según estabilidad del mundo

## 📊 Monitoreo

### Métricas del Agente

- `tasks_completed`: Tareas completadas exitosamente
- `tasks_failed`: Tareas que fallaron
- `total_tokens_used`: Tokens de OpenRouter consumidos
- `uptime_seconds`: Tiempo de ejecución en segundos
- `last_activity`: Última actividad del agente

### Estadísticas de Aprendizaje

- `total_events`: Total de eventos de aprendizaje registrados
- `recent_events_1h`: Eventos en la última hora
- `adaptation_params`: Parámetros de adaptación actuales
- `performance_metrics`: Métricas de performance

### Estadísticas de Conocimiento

- `total_entries`: Total de entradas en la base de conocimiento
- `oldest_entry`: Entrada más antigua
- `newest_entry`: Entrada más reciente
- `total_access_count`: Total de accesos a conocimiento

### Estadísticas de Self-Reflection (Nuevo)

- `total_reflections`: Total de reflexiones realizadas
- `reflection_types`: Distribución por tipo (performance, strategy, capability, periodic)
- `average_confidence`: Confianza promedio en las reflexiones
- `last_reflection`: Timestamp de la última reflexión

### Estadísticas de Experience Learning (Nuevo)

- `total_experiences`: Total de experiencias registradas
- `recent_experiences_24h`: Experiencias en las últimas 24 horas
- `abstracted_skills_count`: Número de habilidades abstractas
- `experience_types`: Distribución por tipo de experiencia
- `outcome_distribution`: Distribución de resultados (success/failure)

### Estadísticas de World Model (Nuevo)

- `total_states`: Total de estados del mundo rastreados
- `stable_states`: Estados estables
- `changing_states`: Estados en cambio
- `average_confidence`: Confianza promedio en el modelo
- `total_observations`: Total de observaciones registradas

## 🔧 Desarrollo

### Estructura del Proyecto

```
autonomous_long_term_agent/
├── __init__.py
├── config.py                 # Configuración
├── main.py                    # Aplicación FastAPI
├── README.md
├── api/
│   └── v1/
│       ├── routes.py         # Rutas API
│       ├── controllers/     # Controladores
│       └── schemas/          # Schemas Pydantic
├── core/
│   ├── agent.py              # Agente principal
│   ├── task_queue.py         # Cola de tareas
│   └── learning_engine.py    # Motor de aprendizaje
└── infrastructure/
    ├── openrouter/           # Cliente OpenRouter
    └── storage/              # Almacenamiento persistente
        └── knowledge_base.py # Base de conocimiento
```

## 🧪 Ejemplos

### Ejemplo 1: Agente Simple

```python
import requests

# Iniciar agente
response = requests.post("http://localhost:8001/api/v1/agents/start", json={
    "instruction": "Aprender sobre Python y machine learning"
})
agent_id = response.json()["message"].split()[-1]

# Agregar tarea
requests.post(f"http://localhost:8001/api/v1/agents/{agent_id}/tasks", json={
    "instruction": "Investigar las mejores prácticas de Python"
})

# Ver estado
status = requests.get(f"http://localhost:8001/api/v1/agents/{agent_id}/status")
print(status.json())

# Detener cuando termines
requests.post(f"http://localhost:8001/api/v1/agents/{agent_id}/stop")
```

### Ejemplo 2: Múltiples Agentes Paralelos

```python
import requests

# Iniciar 3 agentes en paralelo
response = requests.post("http://localhost:8001/api/v1/agents/parallel", json={
    "count": 3,
    "instruction": "Investigar diferentes aspectos de IA"
})

agent_ids = response.json()["agent_ids"]

# Cada agente puede recibir tareas independientes
for i, agent_id in enumerate(agent_ids):
    requests.post(f"http://localhost:8001/api/v1/agents/{agent_id}/tasks", json={
        "instruction": f"Tarea específica para agente {i+1}"
    })
```

## 🛡️ Seguridad

- Validación de inputs con Pydantic
- Rate limiting configurable
- Manejo seguro de errores
- Limpieza automática de recursos

## 📝 Notas Importantes

1. **El agente NO se detiene automáticamente**: Debes llamar explícitamente al endpoint de stop
2. **Persistencia**: El conocimiento se guarda en disco en `./data/autonomous_agent/`
3. **Recursos**: Múltiples agentes consumen más recursos (tokens, memoria, CPU)
4. **OpenRouter API Key**: Requerida para funcionar

## 🔗 Referencias

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- Papers de investigación mencionados en la sección "Basado en Papers"

## 📄 Licencia

Este proyecto es parte de Blatam Academy.




