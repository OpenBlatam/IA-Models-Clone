"""
Cursor Agent 24/7
=================

Agente persistente que escucha comandos desde Cursor y los ejecuta
continuamente, incluso cuando la computadora está apagada (como servicio).

Características:
- Escucha comandos desde la ventana de Cursor
- Ejecuta tareas de forma continua sin parar
- Control simple con botón de start/stop
- Servicio persistente que puede correr en background
- Procesamiento con IA para comandos inteligentes
- Métricas y monitoreo en tiempo real
- Sistema de plugins extensible
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Persistent agent that listens to commands from Cursor and executes them continuously, even when the computer is off"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.agent import CursorAgent
    from .core.task_executor import TaskExecutor
    from .api.agent_api import AgentAPI
else:
    # Imports reales con manejo de errores
    try:
        from .core.agent import CursorAgent
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import CursorAgent: {e}", exc_info=True)
        CursorAgent = None  # type: ignore
    
    try:
        from .core.task_executor import TaskExecutor
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import TaskExecutor: {e}", exc_info=True)
        TaskExecutor = None  # type: ignore
    
    try:
        from .api.agent_api import AgentAPI
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import AgentAPI: {e}", exc_info=True)
        AgentAPI = None  # type: ignore

__all__ = [
    "CursorAgent",
    "TaskExecutor",
    "AgentAPI",
]
