"""
Constants Module
================

Constantes y configuraciones por defecto para el agente continuo.
"""

# Estrategias disponibles
STRATEGY_LATS = "lats"
STRATEGY_TOT = "tot"
STRATEGY_REACT = "react"
STRATEGY_STANDARD = "standard"

# Prioridades de estrategias
PRIORITY_LATS_THRESHOLD = 9
PRIORITY_TOT_THRESHOLD = 8
PRIORITY_REACT_THRESHOLD = 7
PRIORITY_MIN = 1
PRIORITY_MAX = 10

# Configuración por defecto de estrategias
DEFAULT_REACT_ENABLED = True
DEFAULT_LATS_ENABLED = True
DEFAULT_TOT_ENABLED = True
DEFAULT_TOM_ENABLED = True
DEFAULT_PERSONALITY_ENABLED = False
DEFAULT_TOOLFORMER_ENABLED = True
DEFAULT_LEARNING_ENABLED = True

# Umbrales de aprendizaje
DEFAULT_PERFORMANCE_THRESHOLD = 0.6
DEFAULT_ERROR_THRESHOLD = 3

# Reflection y Planning
DEFAULT_REFLECTION_THRESHOLD = 5
DEFAULT_PLANNING_HORIZON = 3

# Tree of Thoughts
DEFAULT_TOT_STRATEGY = "bfs"
VALID_TOT_STRATEGIES = ["bfs", "dfs", "beam"]

# Autonomía
DEFAULT_AUTONOMY_LEVEL = "fully_autonomous"

# Health Monitor
HEALTH_STATUS_HEALTHY = "healthy"
HEALTH_STATUS_DEGRADED = "degraded"
HEALTH_STATUS_UNHEALTHY = "unhealthy"
HEALTH_STATUS_UNKNOWN = "unknown"

# Event Types (para referencia rápida)
EVENT_TASK_SUBMITTED = "task_submitted"
EVENT_TASK_STARTED = "task_started"
EVENT_TASK_COMPLETED = "task_completed"
EVENT_TASK_FAILED = "task_failed"
EVENT_REFLECTION_TRIGGERED = "reflection_triggered"
EVENT_LEARNING_OPPORTUNITY = "learning_opportunity"
EVENT_STRATEGY_SELECTED = "strategy_selected"
EVENT_MEMORY_UPDATED = "memory_updated"
EVENT_METRICS_UPDATED = "metrics_updated"
EVENT_ERROR_OCCURRED = "error_occurred"
EVENT_AGENT_STARTED = "agent_started"
EVENT_AGENT_STOPPED = "agent_stopped"

# Límites
MAX_EVENT_HISTORY_SIZE = 1000
MAX_HEALTH_HISTORY_SIZE = 100
MAX_LEARNING_OPPORTUNITIES = 100

# Timeouts y delays
DEFAULT_LOOP_SLEEP_SECONDS = 1.0
DEFAULT_IDLE_SLEEP_SECONDS = 5.0
DEFAULT_RETRY_SLEEP_SECONDS = 2.0
DEFAULT_MAINTENANCE_INTERVAL_SECONDS = 300.0  # 5 minutos

# LLM Configuration
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RETRY_COUNT = 3

# Papers integrados (para referencia)
PAPERS = [
    "Generative Agents: Interactive Simulacra of Human Behavior",
    "ReAct: Synergizing Reasoning and Acting in Language Models",
    "Language Agent Tree Search Unifies Reasoning, Acting, and Planning",
    "From LLM Reasoning to Autonomous AI Agents",
    "AI Autonomy: Self-Initiated Open-World Continual Learning",
    "Tree of Thoughts: Deliberate Problem Solving",
    "Autonomous Agents Modelling Other Agents: A Comprehensive Survey",
    "Personality-Driven Decision-Making in LLM-Based Autonomous",
    "Toolformer: Language Models Can Teach Themselves to Use Tools",
    "Sparks of Artificial General Intelligence"
]
