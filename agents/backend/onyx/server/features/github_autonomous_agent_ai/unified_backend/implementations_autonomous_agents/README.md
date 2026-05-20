# Implementaciones de Agentes Autónomos

Este directorio contiene implementaciones de código basadas en los **papers más influyentes** sobre agentes autónomos, multi-agente, y sistemas de razonamiento con LLMs.

## 📚 Papers Implementados

### 1. **ReAct Framework** (`react/`)
**Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models"  
**arXiv:** 2210.03629

**Características:**
- Interleaves reasoning (thinking) and acting
- Uses tools to interact with environment
- Observes results and adapts
- Loop: Thought → Action → Observation

**Uso:**
```python
from react import ReActAgent
from common.tools import default_tool_registry

agent = ReActAgent(
    name="MyAgent",
    tool_registry=default_tool_registry
)

result = agent.run("Find information about Python")
```

---

### 2. **Tree of Thoughts (ToT)** (`tree_of_thoughts/`)
**Paper:** "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"  
**arXiv:** 2305.10601

**Características:**
- Maintains tree of intermediate thoughts
- Explores multiple reasoning paths
- Evaluates states using heuristics
- Supports BFS and DFS search strategies

**Uso:**
```python
from tree_of_thoughts import TreeOfThoughts, ToTSearchStrategy

tot = TreeOfThoughts(
    search_strategy=ToTSearchStrategy.BFS,
    max_depth=5,
    beam_width=5
)

solution = tot.solve("Solve: Use 4, 9, 10, 13 to make 24")
```

---

### 3. **Generative Agents** (`generative_agents/`)
**Paper:** "Generative Agents: Interactive Simulacra of Human Behavior"  
**Stanford University**

**Características:**
- Episodic and semantic memory systems
- Reflection on experiences
- Planning based on memories
- Human-like behavior simulation

**Uso:**
```python
from generative_agents import GenerativeAgent, AgentProfile

profile = AgentProfile(
    name="Alice",
    traits=["curious", "helpful"],
    occupation="Researcher"
)

agent = GenerativeAgent(profile=profile)
result = agent.run("Plan your day")
```

---

### 4. **Language Agent Tree Search (LATS)** (`lats/`)
**Paper:** "Language Agent Tree Search Unifies Reasoning, Acting, and Planning"

**Características:**
- Unifies reasoning, acting, and planning
- Tree search with LLM evaluation
- Tool execution integration
- Optimal path finding

**Uso:**
```python
from lats import LATSAgent
from common.tools import default_tool_registry

agent = LATSAgent(
    tool_registry=default_tool_registry,
    max_depth=5,
    beam_width=3
)

solution = agent.solve("Complex multi-step task")
```

---

## 📁 Estructura

```
implementations_autonomous_agents/
├── common/                    # Componentes compartidos
│   ├── agent_base.py         # Clase base para agentes
│   ├── memory.py             # Sistemas de memoria
│   └── tools.py              # Sistema de herramientas
├── react/                     # ReAct Framework
│   ├── react.py
│   └── __init__.py
├── tree_of_thoughts/          # Tree of Thoughts
│   ├── tot.py
│   └── __init__.py
├── generative_agents/        # Generative Agents
│   ├── generative_agent.py
│   └── __init__.py
└── lats/                     # Language Agent Tree Search
    ├── lats.py
    └── __init__.py
```

---

### 8. **From LLM Reasoning to Autonomous AI Agents** (`llm_to_autonomous/`)
**Paper:** "From LLM Reasoning to Autonomous AI Agents"

**Características:**
- Progressive autonomy levels
- Reasoning-to-action pipeline
- Goal decomposition and planning
- Self-monitoring and adaptation

**Uso:**
```python
from llm_to_autonomous import LLMToAutonomousAgent, AutonomyLevel
from common.tools import default_tool_registry

agent = LLMToAutonomousAgent(
    name="AutonomousAgent",
    autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS,
    tool_registry=default_tool_registry
)

result = agent.run("Complete complex multi-step task")
```

---

### 9. **Personality-Driven Decision-Making** (`personality_driven/`)
**Paper:** "Personality-Driven Decision-Making in LLM-Based Autonomous"

**Características:**
- Personality traits influence decisions
- Trait-based action selection
- Emotional states and decision bias
- Context-aware personality expression

**Uso:**
```python
from personality_driven import PersonalityDrivenAgent, PersonalityProfile, PersonalityTrait

profile = PersonalityProfile(
    name="CuriousAgent",
    traits={
        PersonalityTrait.OPENNESS: 0.9,
        PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
        PersonalityTrait.EXTRAVERSION: 0.6,
        PersonalityTrait.AGREEABLENESS: 0.8,
        PersonalityTrait.NEUROTICISM: 0.3
    }
)

agent = PersonalityDrivenAgent(name="PersonalityAgent", personality=profile)
result = agent.run("Make a decision about exploring new territory")
```

---

### 10. **MOBILE-AGENT** (`mobile_agent/`)
**Paper:** "MOBILE-AGENT: AUTONOMOUS MULTI-MODAL MOBILE"

**Características:**
- Multi-modal perception (vision, text, audio)
- Mobile device interaction
- Screen understanding and navigation
- Cross-platform compatibility

**Uso:**
```python
from mobile_agent import MobileAgent, ActionType

agent = MobileAgent(name="MobileAgent")

# Capture screen
screen = agent.capture_screen()

# Find element and tap
element = agent.find_element_by_text("Login")
if element:
    result = agent.tap_element(element)

# Execute task
result = agent.run("Open app and search for Python")
```

---

### 11. **Multi-Agent Connected Autonomous Driving** (`autonomous_driving/`)
**Paper:** "Multi-Agent Connected Autonomous Driving using"

**Características:**
- Multiple autonomous vehicles coordination
- Vehicle-to-vehicle (V2V) communication
- Traffic management and optimization
- Collision avoidance

**Uso:**
```python
from autonomous_driving import AutonomousVehicleAgent, Position, VehicleStatus

# Create vehicle agent
vehicle = AutonomousVehicleAgent(
    name="Vehicle1",
    vehicle_id="v1",
    initial_position=Position(x=0.0, y=0.0, heading=0.0, speed=0.0)
)

# Set destination
vehicle.set_destination((100.0, 50.0))

# Drive
result = vehicle.run("Drive to destination")
```

---

### 12. **Creating Multimodal Interactive Agents** (`multimodal_interactive/`)
**Paper:** "Creating Multimodal Interactive Agents with"

**Características:**
- Multi-modal input processing (text, image, audio, video)
- Interactive conversation and task execution
- Cross-modal understanding and generation
- Real-time interaction

**Uso:**
```python
from multimodal_interactive import (
    MultimodalInteractiveAgent,
    MultimodalInput,
    ModalityType
)

agent = MultimodalInteractiveAgent(name="MultimodalAgent")

# Text input
text_input = MultimodalInput(
    input_id="input1",
    modality=ModalityType.TEXT,
    content="What do you see in this image?"
)

# Image input
image_input = MultimodalInput(
    input_id="input2",
    modality=ModalityType.IMAGE,
    content="path/to/image.jpg"
)

# Interact
output = agent.interact(text_input)
print(output.content)
```

---

## 🔧 Componentes Comunes

### BaseAgent
Clase base para todos los agentes con:
- State management
- Action execution
- Observation processing
- History tracking

### Memory Systems
- **EpisodicMemory**: Stores specific events/experiences
- **SemanticMemory**: Stores general knowledge/facts

### Tool System
- Tool registry for managing available tools
- Built-in tools: search, calculator
- Extensible for custom tools

---

## 📊 Estado de Implementaciones

### ✅ Completados (12 frameworks)

1. ✅ **ReAct** - Reasoning + Acting framework
2. ✅ **Tree of Thoughts** - Deliberate problem solving
3. ✅ **Generative Agents** - Human-like behavior simulation
4. ✅ **Language Agent Tree Search** - Unified reasoning/acting/planning
5. ✅ **Toolformer** - Self-supervised tool learning
6. ✅ **Multi-Agent RL** - Multi-agent reinforcement learning
7. ✅ **Theory of Mind** - Agent modeling and prediction
8. ✅ **From LLM Reasoning to Autonomous AI Agents** - Progressive autonomy
9. ✅ **Personality-Driven Decision-Making** - Personality-based decisions
10. ✅ **MOBILE-AGENT** - Multi-modal mobile agents
11. ✅ **Multi-Agent Connected Autonomous Driving** - Vehicle coordination
12. ✅ **Creating Multimodal Interactive Agents** - Multi-modal interaction

---

## 🚀 Próximos Pasos

### Mejoras y Optimizaciones

1. **Integración con LLMs reales** - OpenAI, Anthropic, etc.
2. **Tests unitarios** - Cobertura completa de tests
3. **Ejemplos de uso completos** - Casos de uso reales
4. **Optimización de rendimiento** - Mejoras en eficiencia
5. **Integración con entornos reales** - ADB para mobile, simuladores para driving

---

## 📝 Notas

- Todas las implementaciones están basadas en los papers originales
- Los componentes LLM son placeholders - integrar con OpenAI, Anthropic, etc.
- Las implementaciones son modulares y extensibles
- Ver papers originales en `../../data/papers/` para más detalles

---

## 🔗 Referencias

- Papers disponibles en: `../../data/papers/`
- Índice de papers: `../../data/papers/index.json`
- Resumen: `../../data/AUTONOMOUS_AGENTS_PAPERS_SUMMARY.md`



