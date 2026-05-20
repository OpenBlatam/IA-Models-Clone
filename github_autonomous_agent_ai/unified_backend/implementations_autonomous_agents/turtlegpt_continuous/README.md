# Continuous Generative Agent with OpenRouter/TruthGPT

Agente generativo continuo basado en el paper "Generative Agents: Interactive Simulacra of Human Behavior" (Stanford, 2023).

## 📚 Papers Base

Este agente integra conceptos de múltiples papers:

1. **"Generative Agents: Interactive Simulacra of Human Behavior"** (Stanford, 2023)
   - Memoria episódica y semántica
   - Reflexión periódica sobre experiencias
   - Planificación basada en memoria
   - Funcionamiento continuo 24/7

2. **"ReAct: Synergizing Reasoning and Acting in Language Models"** (2022)
   - Intercalación de razonamiento y acción
   - Ciclo Thought → Action → Observation
   - Integración con herramientas

3. **"Language Agent Tree Search Unifies Reasoning, Acting, and Planning"** (LATS)
   - Búsqueda en árbol con evaluación LLM
   - Unificación de reasoning, acting y planning
   - Búsqueda de caminos óptimos

4. **"From LLM Reasoning to Autonomous AI Agents"**
   - Autonomía progresiva
   - Pipeline de razonamiento a acción
   - Auto-monitoreo y adaptación

5. **"AI Autonomy: Self-Initiated Open-World Continual Learning"**
   - Aprendizaje autónomo iniciado por el agente
   - Identificación de oportunidades de aprendizaje
   - Adaptación continua

6. **"Tree of Thoughts: Deliberate Problem Solving"** (ToT)
   - Razonamiento basado en árbol de pensamientos
   - Búsqueda BFS/DFS de soluciones
   - Evaluación de estados intermedios
   - Se usa automáticamente para tareas de prioridad ≥ 8

7. **"Autonomous Agents Modelling Other Agents: A Comprehensive Survey"** (Theory of Mind)
   - Modelado de estados mentales de otros agentes
   - Predicción de acciones
   - Seguimiento de intenciones
   - Adaptación de comportamiento

8. **"Personality-Driven Decision-Making in LLM-Based Autonomous"**
   - Decisiones influenciadas por personalidad
   - Estados emocionales y sesgos de decisión
   - Expresión de personalidad contextual

9. **"Toolformer: Language Models Can Teach Themselves to Use Tools"**
   - Aprendizaje autónomo de herramientas
   - Auto-supervisión para uso de herramientas
   - Filtrado basado en pérdida

10. **"Sparks of Artificial General Intelligence"**
    - Capacidades emergentes de AGI
    - Razonamiento multi-modal
    - Uso de herramientas y generación de código
    - Evaluación continua de capacidades

## 🚀 Características

- ✅ **Funcionamiento continuo 24/7** hasta detención manual
- ✅ **Memoria episódica y semántica** para recordar experiencias
- ✅ **Reflexión periódica** sobre experiencias para generar insights
- ✅ **Planificación autónoma** basada en memoria y contexto
- ✅ **Integración con OpenRouter/TruthGPT** para razonamiento
- ✅ **Sistema de tareas** con cola y procesamiento concurrente
- ✅ **Métricas y monitoreo** del rendimiento del agente

## 📦 Instalación

```bash
# Asegúrate de tener la API key de OpenRouter
export OPENROUTER_API_KEY="tu-api-key-aqui"
```

## 💻 Uso Básico

```python
import asyncio
from turtlegpt_continuous_agent import (
    TurtleGPTContinuousAgent,
    ContinuousAgentConfig
)

async def main():
    # Configurar agente
    config = ContinuousAgentConfig(
        loop_sleep_seconds=1.0,
        max_concurrent_tasks=3,
        enable_idle_mode=True
    )
    
    # Crear agente
    agent = TurtleGPTContinuousAgent(
        name="MyContinuousAgent",
        api_key="tu-api-key",
        agent_config=config
    )
    
    # Enviar tareas
    task1 = agent.submit_task(
        description="Analiza las tendencias en IA",
        priority=8
    )
    
    task2 = agent.submit_task(
        description="Crea un plan de acción para mejorar productividad",
        priority=6
    )
    
    # Iniciar agente (funciona 24/7 hasta detención)
    try:
        await agent.start()
    except KeyboardInterrupt:
        agent.stop()
        print("Agente detenido")

asyncio.run(main())
```

## 🔧 Configuración

```python
config = ContinuousAgentConfig(
    loop_sleep_seconds=1.0,          # Tiempo entre iteraciones del loop
    task_monitor_sleep_seconds=0.5,  # Tiempo para monitorear tareas
    idle_sleep_seconds=5.0,          # Tiempo en modo idle
    max_concurrent_tasks=3,          # Máximo de tareas concurrentes
    retry_sleep_seconds=2.0,         # Tiempo antes de reintentar
    max_retries=3,                   # Máximo de reintentos
    enable_idle_mode=True,            # Habilitar modo idle
    maintenance_interval_seconds=300.0 # Intervalo de mantenimiento (5 min)
)
```

## 📊 Estado y Métricas

```python
# Obtener estado del agente
status = agent.get_status()

print(f"Tareas en cola: {status['queue_size']}")
print(f"Tareas activas: {status['active_tasks']}")
print(f"Memoria episódica: {status['memory']['episodic_count']}")
print(f"Insights generados: {status['memory']['insights_count']}")
print(f"Oportunidades de aprendizaje: {status['learning']['learning_tasks_count']}")
```

## 🎯 Callbacks

```python
def on_task_completed(task):
    print(f"Tarea completada: {task.task_id}")
    print(f"Resultado: {task.result}")

def on_error(error):
    print(f"Error: {error}")

agent.set_task_callback(on_task_completed)
agent.set_error_callback(on_error)
```

## 🧠 Arquitectura (Paper: Generative Agents)

El paper describe que los agentes funcionan en un ciclo continuo:

1. **Observación**: Observan su entorno (en este caso, procesan tareas)
2. **Recuperación de memoria**: Recuperan memorias relevantes de experiencias pasadas
3. **Reflexión**: Reflexionan periódicamente sobre experiencias para formar insights
4. **Planificación**: Generan planes basados en memoria, situación actual y perfil del agente
5. **Acción**: Ejecutan acciones del plan
6. **Actualización de memoria**: Almacenan nuevas experiencias en memoria episódica

## 📝 Notas

- El agente funciona continuamente hasta que se detenga manualmente (Ctrl+C o `agent.stop()`)
- Usa OpenRouter/TruthGPT para razonamiento y generación de texto
- La memoria episódica almacena experiencias específicas
- La memoria semántica almacena conocimiento general e insights
- Los insights se generan automáticamente durante la reflexión

## 🔗 Referencias

### Papers Integrados:
1. "Generative Agents: Interactive Simulacra of Human Behavior" (Stanford, 2023)
2. "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
3. "Language Agent Tree Search Unifies Reasoning, Acting, and Planning" (LATS)
4. "From LLM Reasoning to Autonomous AI Agents"
5. "AI Autonomy: Self-Initiated Open-World Continual Learning"

### Implementaciones Relacionadas:
- `../generative_agents/` - Generative Agents base
- `../react/` - ReAct framework
- `../lats/` - LATS implementation
- `../tree_of_thoughts/` - Tree of Thoughts
- `../theory_of_mind/` - Theory of Mind
- `../personality_driven/` - Personality-Driven
- `../toolformer/` - Toolformer
- `../sparks_agi/` - Sparks of AGI
- `../llm_to_autonomous/` - LLM to Autonomous
- `../self_initiated_learning/` - Self-Initiated Learning
- `../tree_of_thoughts/` - Tree of Thoughts
- `../theory_of_mind/` - Theory of Mind
- `../personality_driven/` - Personality-Driven Decision-Making
- `../toolformer/` - Toolformer
- `../sparks_agi/` - Sparks of AGI

### Papers en: `../../pdfs/`



