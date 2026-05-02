# Quick Start Guide - Implementaciones de Agentes Autónomos

## 🚀 Inicio Rápido

### Instalación

```bash
# No se requieren dependencias adicionales más allá de las estándar
# Las implementaciones usan solo bibliotecas estándar de Python
```

### Ejemplo Básico: ReAct

```python
from implementations_autonomous_agents.react import ReActAgent
from implementations_autonomous_agents.common.tools import default_tool_registry

# Crear agente ReAct
agent = ReActAgent(
    name="MiAgente",
    tool_registry=default_tool_registry
)

# Ejecutar tarea
result = agent.run("Buscar información sobre Python")
print(result)
```

### Ejemplo: Tree of Thoughts

```python
from implementations_autonomous_agents.tree_of_thoughts import TreeOfThoughts, ToTSearchStrategy

# Crear sistema ToT
tot = TreeOfThoughts(
    search_strategy=ToTSearchStrategy.BFS,
    max_depth=5,
    beam_width=5
)

# Resolver problema
solution = tot.solve("Usar 4, 9, 10, 13 para hacer 24")
print(solution)
```

### Ejemplo: Generative Agents

```python
from implementations_autonomous_agents.generative_agents import GenerativeAgent, AgentProfile

# Crear perfil de agente
profile = AgentProfile(
    name="Alice",
    traits=["curiosa", "ayudante"],
    occupation="Investigadora"
)

# Crear agente generativo
agent = GenerativeAgent(profile=profile)

# Ejecutar
result = agent.run("Planifica tu día")
print(agent.get_status())
```

### Ejemplo: LATS

```python
from implementations_autonomous_agents.lats import LATSAgent
from implementations_autonomous_agents.common.tools import default_tool_registry

agent = LATSAgent(
    tool_registry=default_tool_registry,
    max_depth=5,
    beam_width=3
)

solution = agent.solve("Tarea compleja multi-paso")
print(solution)
```

### Ejemplo: Toolformer

```python
from implementations_autonomous_agents.toolformer import Toolformer, ToolformerTrainer

# Definir herramientas
tools = {
    "calculator": lambda expr: eval(expr),
    "search": lambda query: f"Resultados para: {query}"
}

# Crear Toolformer
toolformer = Toolformer(tools=tools)

# Generar con uso de herramientas
result = toolformer.generate("Calcula 25 * 4 usando la calculadora")
print(result)
```

### Ejemplo: Multi-Agent RL

```python
from implementations_autonomous_agents.multi_agent_rl import (
    MultiAgentRL, MARLEnvironment, MARLAlgorithm
)

# Crear entorno
env = MARLEnvironment(
    num_agents=3,
    state_space=None,
    action_spaces=[list(range(10))] * 3,
    observation_spaces=[None] * 3
)

# Crear sistema MARL
marl = MultiAgentRL(
    environment=env,
    algorithm=MARLAlgorithm.INDEPENDENT_Q_LEARNING
)

# Entrenar
stats = marl.train(num_episodes=100)
print(stats)
```

### Ejemplo: Theory of Mind

```python
from implementations_autonomous_agents.theory_of_mind import TheoryOfMindAgent

# Crear agente con ToM
agent = TheoryOfMindAgent(name="ToMAgent")

# Observar otro agente
agent.observe_agent("agent_2", {"action": "move", "direction": "north"})

# Predecir acción
prediction = agent.predict_agent_action("agent_2")
print(prediction)

# Adaptar comportamiento
adapted = agent.adapt_to_agent("agent_2")
print(adapted)
```

---

## 📚 Frameworks Disponibles

1. **ReAct** - Reasoning + Acting
2. **Tree of Thoughts** - Razonamiento estructurado
3. **Generative Agents** - Comportamiento humano
4. **LATS** - Búsqueda en árbol unificada
5. **Toolformer** - Aprendizaje de herramientas
6. **Multi-Agent RL** - Reinforcement Learning multi-agente
7. **Theory of Mind** - Modelado de otros agentes

---

## 🔧 Componentes Comunes

- `BaseAgent` - Clase base para todos los agentes
- `EpisodicMemory` / `SemanticMemory` - Sistemas de memoria
- `ToolRegistry` - Gestión de herramientas

---

## 📝 Notas

- Los componentes LLM son placeholders - integrar con OpenAI, Anthropic, etc.
- Las implementaciones son modulares y extensibles
- Ver `README.md` para documentación completa



