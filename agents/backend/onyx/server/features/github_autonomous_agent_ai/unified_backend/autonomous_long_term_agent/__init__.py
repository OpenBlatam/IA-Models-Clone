"""
Autonomous Long-Term Agent
==========================

Agente autónomo que corre continuamente hasta que se detiene explícitamente.
Implementa conceptos de continual learning, long-horizon reasoning y self-initiated learning
basado en papers de investigación sobre agentes autónomos de largo plazo.

Características:
- Ejecución continua hasta stop explícito
- Soporte para múltiples instancias en paralelo
- Integración con OpenRouter para modelos de IA
- Base de conocimiento persistente para continual learning
- Self-initiated learning y adaptación
- Versión mejorada con optimizaciones basadas en papers (EnhancedAutonomousAgent)

Uso:
    # Versión estándar
    from autonomous_long_term_agent.core.agent import AutonomousLongTermAgent
    agent = AutonomousLongTermAgent()
    
    # Versión mejorada (recomendada)
    from autonomous_long_term_agent.core.agent_factory import create_enhanced_agent
    agent = create_enhanced_agent()
    
    # O usar factory
    from autonomous_long_term_agent.core.agent_factory import create_agent
    agent = create_agent(enhanced=True)
"""

__version__ = "1.1.0"

# Exports
from .core.agent import AutonomousLongTermAgent
from .core.agent_factory import (
    create_agent,
    create_standard_agent,
    create_enhanced_agent
)

try:
    from .core.agent_enhanced import EnhancedAutonomousAgent
    __all__ = [
        "AutonomousLongTermAgent",
        "EnhancedAutonomousAgent",
        "create_agent",
        "create_standard_agent",
        "create_enhanced_agent",
    ]
except ImportError:
    __all__ = [
        "AutonomousLongTermAgent",
        "create_agent",
        "create_standard_agent",
        "create_enhanced_agent",
    ]

