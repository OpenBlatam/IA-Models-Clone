"""
Example: TruthGPT Continuous Agent
==================================

Ejemplo de uso del agente continuo TruthGPT.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Agregar el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent))

from turtlegpt_continuous_agent import (
    TurtleGPTContinuousAgent,
    ContinuousAgentConfig
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Ejemplo principal."""
    
    # Obtener API key de OpenRouter
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        return
    
    # Configurar agente
    config = ContinuousAgentConfig(
        loop_sleep_seconds=1.0,
        task_monitor_sleep_seconds=0.5,
        idle_sleep_seconds=5.0,
        max_concurrent_tasks=3,
        enable_idle_mode=True
    )
    
    # Crear agente
    agent = TurtleGPTContinuousAgent(
        name="ExampleTurtleGPTAgent",
        api_key=api_key,
        agent_config=config
    )
    
    # Configurar callbacks
    def on_task_completed(task):
        logger.info(f"Task completed: {task.task_id} - {task.description}")
        if task.result:
            logger.info(f"Result: {task.result.get('final_state', {}).get('status', 'unknown')}")
    
    def on_error(error):
        logger.error(f"Error occurred: {error}")
    
    agent.set_task_callback(on_task_completed)
    agent.set_error_callback(on_error)
    
    # Enviar algunas tareas de ejemplo
    logger.info("Submitting example tasks...")
    
    task1 = agent.submit_task(
        description="Analiza las tendencias actuales en inteligencia artificial y genera un resumen",
        priority=8
    )
    
    task2 = agent.submit_task(
        description="Crea un plan de acción para mejorar la productividad en desarrollo de software",
        priority=6
    )
    
    task3 = agent.submit_task(
        description="Explica los conceptos clave de machine learning para principiantes",
        priority=5
    )
    
    logger.info(f"Submitted tasks: {task1}, {task2}, {task3}")
    
    # Iniciar agente en modo continuo
    # El agente seguirá funcionando hasta que se detenga manualmente
    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Stopping agent...")
        agent.stop()
        await asyncio.sleep(1)  # Dar tiempo para limpiar


if __name__ == "__main__":
    asyncio.run(main())



