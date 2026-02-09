"""
Example Usage of Autonomous SAM3 Agent
======================================

Ejemplos de cómo usar el agente autónomo.
"""

import asyncio
import logging
from pathlib import Path
from core.agent_core import AutonomousSAM3Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_single_task():
    """Ejemplo: Procesar una sola tarea."""
    logger.info("=== Ejemplo: Tarea Única ===")
    
    # Inicializar agente
    agent = AutonomousSAM3Agent(
        openrouter_api_key=None,  # Usará OPENROUTER_API_KEY env var
        max_parallel_tasks=5,
        output_dir="example_output",
    )
    
    # Enviar tarea
    task_id = await agent.submit_task(
        image_path="path/to/your/image.jpg",  # Cambiar por ruta real
        text_prompt="a person holding a cup",
        priority=5,
    )
    
    logger.info(f"Tarea enviada: {task_id}")
    
    # Esperar y verificar estado
    for i in range(30):  # Esperar hasta 30 segundos
        await asyncio.sleep(1)
        status = await agent.get_task_status(task_id)
        logger.info(f"Estado: {status['status']}")
        
        if status['status'] == 'completed':
            result = await agent.get_task_result(task_id)
            logger.info(f"Resultado: {len(result.get('pred_masks', []))} máscaras encontradas")
            break
        elif status['status'] == 'failed':
            logger.error(f"Error: {status.get('error')}")
            break
    
    await agent.stop()


async def example_multiple_tasks():
    """Ejemplo: Procesar múltiples tareas en paralelo."""
    logger.info("=== Ejemplo: Múltiples Tareas ===")
    
    agent = AutonomousSAM3Agent(
        max_parallel_tasks=10,
        output_dir="example_output",
    )
    
    # Iniciar agente en background
    agent_task = asyncio.create_task(agent.start())
    
    # Enviar múltiples tareas
    tasks = [
        ("image1.jpg", "a dog playing in the park", 5),
        ("image2.jpg", "a person reading a book", 3),
        ("image3.jpg", "a car on the street", 4),
    ]
    
    task_ids = []
    for image_path, prompt, priority in tasks:
        task_id = await agent.submit_task(
            image_path=image_path,
            text_prompt=prompt,
            priority=priority,
        )
        task_ids.append(task_id)
        logger.info(f"Tarea {task_id} enviada: {prompt}")
    
    # Esperar resultados
    await asyncio.sleep(60)  # Esperar 60 segundos
    
    # Verificar resultados
    for task_id in task_ids:
        status = await agent.get_task_status(task_id)
        logger.info(f"Tarea {task_id}: {status['status']}")
        
        if status['status'] == 'completed':
            result = await agent.get_task_result(task_id)
            logger.info(f"  Máscaras: {len(result.get('pred_masks', []))}")
    
    # Detener agente
    await agent.stop()
    agent_task.cancel()


async def example_continuous_mode():
    """Ejemplo: Modo continuo 24/7."""
    logger.info("=== Ejemplo: Modo Continuo ===")
    
    agent = AutonomousSAM3Agent(
        max_parallel_tasks=10,
        output_dir="continuous_output",
    )
    
    # Iniciar agente en modo continuo
    # El agente procesará tareas automáticamente
    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Deteniendo agente...")
    finally:
        await agent.stop()


async def example_with_custom_config():
    """Ejemplo: Usar configuración personalizada."""
    logger.info("=== Ejemplo: Configuración Personalizada ===")
    
    agent = AutonomousSAM3Agent(
        openrouter_api_key="custom-api-key",
        sam3_model_path="/path/to/custom/model.pt",
        max_parallel_tasks=20,  # Más workers
        output_dir="custom_output",
        model="anthropic/claude-3-opus",  # Modelo diferente
        debug=True,  # Modo debug
    )
    
    # Usar el agente normalmente
    task_id = await agent.submit_task(
        image_path="test.jpg",
        text_prompt="test prompt",
    )
    
    logger.info(f"Tarea {task_id} enviada")
    
    await agent.stop()


if __name__ == "__main__":
    # Ejecutar ejemplo
    print("Selecciona un ejemplo:")
    print("1. Tarea única")
    print("2. Múltiples tareas")
    print("3. Modo continuo")
    print("4. Configuración personalizada")
    
    choice = input("Opción (1-4): ")
    
    if choice == "1":
        asyncio.run(example_single_task())
    elif choice == "2":
        asyncio.run(example_multiple_tasks())
    elif choice == "3":
        asyncio.run(example_continuous_mode())
    elif choice == "4":
        asyncio.run(example_with_custom_config())
    else:
        print("Opción inválida")
