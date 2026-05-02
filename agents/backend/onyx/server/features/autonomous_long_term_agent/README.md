# Autonomous Long-Term Agent

> Part of the [Blatam Academy Integrated Platform](../README.md)

Autonomous agent that runs continuously until explicitly stopped. Implements research concepts on long-term autonomous agents, continual learning, and self-initiated learning.

## 📚 Based on Research Papers

This agent implements concepts from 20+ research papers on autonomous agents:

### Key Papers (1-10)

1. **WebResearcher: Unleashing unbounded reasoning capability in Long-Horizon Agents** (2025)
   - Long-horizon reasoning with continuous knowledge accumulation
   - Deep reasoning over long time horizons

2. **AI Autonomy: Self-initiated Open-world Continual Learning and Adaptation** (2023)
   - SOLA paradigm for self-initiated learning
   - Continuous adaptation without manual retraining

3. **Continual Learning for Robotics** (2019)
   - Continual learning strategies
   - Learning from changing data streams

4. **Towards Continual Reinforcement Learning** (2022)
   - Adaptation to changing tasks/environments
   - Retention of prior knowledge

5. **Learning to Adapt in Dynamic, Real-World Environments** (2018)
   - Real-time adaptation
   - Meta-reinforcement learning

### New Papers (11-20)

11. **Building Self-Evolving Agents via Experience-Driven Lifelong Learning** (2025)
    - ELL (Experience-driven Lifelong Learning) Framework
    - Continuous growth through real interactions
    - Recurrent skill abstraction
    - Knowledge internalization

12. **EvoAgent: Agent Autonomous Evolution with Continual World Model** (2025)
    - Continual world model for long-horizon tasks
    - Self-planning, self-control, self-reflection
    - Experience updates without human intervention

13. **Lifelong Learning of Large Language Model based Agents** (2025)
    - Roadmap for lifelong learning in LLM agents
    - Memory, multimodal perception, dynamic interaction

14. **A Survey of Self-Evolving Agents** (2025)
    - Self-evolving agent paradigm
    - Component evolution (model, memory, tools, architecture)

15. **The Landscape of Agentic Reinforcement Learning for LLMs** (2025)
    - Agentic RL for LLMs
    - Planning, memory, multi-step reasoning

And more... See `papers_references.json` for the full list.

## 🚀 Features

### Core Features
- ✅ **Continuous Execution**: The agent runs indefinitely until explicitly stopped
- ✅ **Multiple Parallel Instances**: Support for running multiple agents in parallel
- ✅ **OpenRouter Integration**: Access to 1000+ AI models
- ✅ **Persistent Knowledge Base**: Continuous storage of learned knowledge
- ✅ **Self-Initiated Learning**: The agent learns and adapts automatically
- ✅ **Long-Horizon Reasoning**: Deep reasoning considering historical context
- ✅ **Continual Learning**: Continuously learns without forgetting prior knowledge

### New Features (Papers 11-20)
- ✅ **Self-Reflection Engine** (EvoAgent): Automatic reflection on performance, strategy, and capabilities
- ✅ **Experience-Driven Learning** (ELL): Learning framework based on real experiences
- ✅ **Continual World Model** (EvoAgent): World model that updates continuously
- ✅ **Self-Planning**: Autonomous planning based on world state
- ✅ **Skill Abstraction**: Automatic skill abstraction from recurrent experiences
- ✅ **Knowledge Internalization**: Internalization of knowledge into long-term memory

## 📦 Installation

```bash
# Install dependencies
pip install fastapi uvicorn httpx pydantic pydantic-settings

# Configure environment variables
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_HTTP_REFERER="https://your-domain.com"
```

## ⚙️ Configuration

Create a `.env` file or configure environment variables:

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

## 🎯 Usage

### Start the Server

```bash
python -m autonomous_long_term_agent.main
```

Or using uvicorn directly:

```bash
uvicorn autonomous_long_term_agent.main:app --host 0.0.0.0 --port 8001
```

### API Endpoints

#### 1. Start an Agent

```bash
POST /api/v1/agents/start
Content-Type: application/json

{
  "instruction": "Research and learn about artificial intelligence",
  "agent_id": "optional-custom-id"
}
```

#### 2. Start Multiple Agents in Parallel

```bash
POST /api/v1/agents/parallel
Content-Type: application/json

{
  "count": 3,
  "instruction": "Operate autonomously and learn continuously"
}
```

#### 3. Get Agent Status

```bash
GET /api/v1/agents/{agent_id}/status
```

Response includes:
- Agent state
- Metrics (tasks completed, tokens used, uptime)
- Task queue size
- Learning statistics
- Knowledge base statistics

#### 4. Add Task to an Agent

```bash
POST /api/v1/agents/{agent_id}/tasks
Content-Type: application/json

{
  "instruction": "Analyze current trends in machine learning",
  "metadata": {
    "priority": "high",
    "category": "research"
  }
}
```

#### 5. List Agent Tasks

```bash
GET /api/v1/agents/{agent_id}/tasks?status=completed
```

#### 6. Pause/Resume Agent

```bash
POST /api/v1/agents/{agent_id}/pause
POST /api/v1/agents/{agent_id}/resume
```

#### 7. Stop Agent

```bash
POST /api/v1/agents/{agent_id}/stop
```

**⚠️ IMPORTANT**: The agent does NOT stop automatically. You must explicitly call the stop endpoint.

#### 8. List All Agents

```bash
GET /api/v1/agents
```

#### 9. Stop All Agents

```bash
POST /api/v1/agents/stop-all
```

## 🔄 Operation Flow

1. **Start**: The agent starts and begins its main loop
2. **Continuous Processing**: 
   - Processes tasks from the queue
   - Executes autonomous operations when no tasks are present
   - Learns and adapts continuously
3. **Knowledge Storage**: All knowledge is saved in the persistent knowledge base
4. **Self-Initiated Learning**: The agent detects learning opportunities and adapts automatically
5. **Explicit Stop**: Only stops when the stop endpoint is called

## 🧠 Implemented Concepts

### Long-Horizon Reasoning

The agent uses long-horizon reasoning by:
- Consulting relevant historical knowledge before responding
- Considering long-term implications
- Continuously accumulating knowledge

### Continual Learning

- **Persistent Knowledge Base**: Stores all learned knowledge
- **Selective Retention**: Keeps relevant knowledge based on retention period
- **Contextual Search**: Retrieves relevant knowledge for each task

### Self-Initiated Learning (SOLA)

- **Automatic Detection**: Detects learning opportunities
- **Automatic Adaptation**: Adjusts parameters based on performance
- **Event-Driven Learning**: Learns from events and outcomes

### Self-Reflection (EvoAgent Paper)

- **Reflection on Performance**: Analyzes recent metrics and tasks
- **Reflection on Strategy**: Evaluates effectiveness of current strategies
- **Reflection on Capabilities**: Identifies gaps and underutilized skills
- **Periodic Reflection**: Automatic reflection every 5 minutes (configurable)

### Experience-Driven Learning (ELL Paper)

- **Experience Recording**: Records all real interactions
- **Skill Abstraction**: Abstracts skills from recurrent patterns
- **Knowledge Internalization**: Internalizes knowledge into long-term memory
- **Lifelong Growth**: Continuous growth throughout the agent's lifecycle

### Continual World Model (EvoAgent Paper)

- **World State Tracking**: Tracks world states continuously
- **Change Detection**: Detects significant changes in the world
- **Self-Planning**: Autonomous planning based on world state
- **Adaptive Strategy**: Adaptive strategy based on world stability

## 📊 Monitoring

### Agent Metrics

- `tasks_completed`: Successfully completed tasks
- `tasks_failed`: Failed tasks
- `total_tokens_used`: OpenRouter tokens consumed
- `uptime_seconds`: Uptime in seconds
- `last_activity`: Last agent activity

### Learning Statistics

- `total_events`: Total learning events recorded
- `recent_events_1h`: Events in the last hour
- `adaptation_params`: Current adaptation parameters
- `performance_metrics`: Performance metrics

### Knowledge Statistics

- `total_entries`: Total entries in the knowledge base
- `oldest_entry`: Oldest entry
- `newest_entry`: Newest entry
- `total_access_count`: Total knowledge accesses

### Self-Reflection Statistics (New)

- `total_reflections`: Total reflections performed
- `reflection_types`: Distribution by type (performance, strategy, capability, periodic)
- `average_confidence`: Average confidence in reflections
- `last_reflection`: Timestamp of last reflection

### Experience Learning Statistics (New)

- `total_experiences`: Total experiences recorded
- `recent_experiences_24h`: Experiences in the last 24 hours
- `abstracted_skills_count`: Number of abstracted skills
- `experience_types`: Distribution by experience type
- `outcome_distribution`: Distribution of outcomes (success/failure)

### World Model Statistics (New)

- `total_states`: Total world states tracked
- `stable_states`: Stable states
- `changing_states`: States in change
- `average_confidence`: Average confidence in the model
- `total_observations`: Total observations recorded

## 🔧 Development

### Project Structure

```
autonomous_long_term_agent/
├── __init__.py
├── config.py                 # Configuration
├── main.py                    # FastAPI Application
├── README.md
├── api/
│   └── v1/
│       ├── routes.py         # API Routes
│       ├── controllers/     # Controllers
│       └── schemas/          # Pydantic Schemas
├── core/
│   ├── agent.py              # Main Agent
│   ├── task_queue.py         # Task Queue
│   └── learning_engine.py    # Learning Engine
└── infrastructure/
    ├── openrouter/           # OpenRouter Client
    └── storage/              # Persistent Storage
        └── knowledge_base.py # Knowledge Base
```

## 🧪 Examples

### Example 1: Simple Agent

```python
import requests

# Start agent
response = requests.post("http://localhost:8001/api/v1/agents/start", json={
    "instruction": "Learn about Python and machine learning"
})
agent_id = response.json()["message"].split()[-1]

# Add task
requests.post(f"http://localhost:8001/api/v1/agents/{agent_id}/tasks", json={
    "instruction": "Research Python best practices"
})

# View status
status = requests.get(f"http://localhost:8001/api/v1/agents/{agent_id}/status")
print(status.json())

# Stop when done
requests.post(f"http://localhost:8001/api/v1/agents/{agent_id}/stop")
```

### Example 2: Multiple Parallel Agents

```python
import requests

# Start 3 agents in parallel
response = requests.post("http://localhost:8001/api/v1/agents/parallel", json={
    "count": 3,
    "instruction": "Research different aspects of AI"
})

agent_ids = response.json()["agent_ids"]

# Each agent can receive independent tasks
for i, agent_id in enumerate(agent_ids):
    requests.post(f"http://localhost:8001/api/v1/agents/{agent_id}/tasks", json={
        "instruction": f"Specific task for agent {i+1}"
    })
```

## 🛡️ Security

- Input validation with Pydantic
- Configurable rate limiting
- Secure error handling
- Automatic resource cleanup

## 📝 Important Notes

1. **The agent does NOT stop automatically**: You must explicitly call the stop endpoint
2. **Persistence**: Knowledge is saved to disk in `./data/autonomous_agent/`
3. **Resources**: Multiple agents consume more resources (tokens, memory, CPU)
4. **OpenRouter API Key**: Required to function

## 🔗 References

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- Research papers mentioned in the "Based on Research Papers" section

## 📄 License

This project is part of Blatam Academy.
