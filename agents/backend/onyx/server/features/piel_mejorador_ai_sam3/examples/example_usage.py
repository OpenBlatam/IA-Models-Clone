"""
Example usage of Piel Mejorador AI SAM3
========================================
"""

import asyncio
import logging
from pathlib import Path

from piel_mejorador_ai_sam3 import PielMejoradorAgent, PielMejoradorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_mejorar_imagen():
    """Example: Enhance an image."""
    logger.info("=== Example: Enhance Image ===")
    
    # Create configuration
    config = PielMejoradorConfig()
    config.validate()
    
    # Create agent
    agent = PielMejoradorAgent(config=config)
    
    # Enhance image
    task_id = await agent.mejorar_imagen(
        file_path="path/to/image.jpg",
        enhancement_level="medium",
        realism_level=0.8,
        custom_instructions="Enfocarse en suavizar textura y reducir poros",
        priority=0
    )
    
    logger.info(f"Task submitted: {task_id}")
    
    # Wait for completion
    while True:
        status = await agent.get_task_status(task_id)
        logger.info(f"Task status: {status['status']}")
        
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            logger.info(f"Result: {result}")
            break
        elif status["status"] == "failed":
            logger.error(f"Task failed: {status.get('error')}")
            break
        
        await asyncio.sleep(2)
    
    await agent.close()


async def example_mejorar_video():
    """Example: Enhance a video."""
    logger.info("=== Example: Enhance Video ===")
    
    config = PielMejoradorConfig()
    config.validate()
    
    agent = PielMejoradorAgent(config=config)
    
    # Enhance video
    task_id = await agent.mejorar_video(
        file_path="path/to/video.mp4",
        enhancement_level="high",
        realism_level=0.9,
        custom_instructions="Mantener movimiento natural y consistencia entre frames",
        priority=0
    )
    
    logger.info(f"Task submitted: {task_id}")
    
    # Wait for completion
    while True:
        status = await agent.get_task_status(task_id)
        logger.info(f"Task status: {status['status']}")
        
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            logger.info(f"Result: {result}")
            break
        elif status["status"] == "failed":
            logger.error(f"Task failed: {status.get('error')}")
            break
        
        await asyncio.sleep(2)
    
    await agent.close()


async def example_analizar_piel():
    """Example: Analyze skin condition."""
    logger.info("=== Example: Analyze Skin ===")
    
    config = PielMejoradorConfig()
    config.validate()
    
    agent = PielMejoradorAgent(config=config)
    
    # Analyze skin
    task_id = await agent.analizar_piel(
        file_path="path/to/image.jpg",
        file_type="image",
        priority=0
    )
    
    logger.info(f"Task submitted: {task_id}")
    
    # Wait for completion
    while True:
        status = await agent.get_task_status(task_id)
        logger.info(f"Task status: {status['status']}")
        
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            logger.info(f"Analysis result: {result}")
            break
        elif status["status"] == "failed":
            logger.error(f"Task failed: {status.get('error')}")
            break
        
        await asyncio.sleep(2)
    
    await agent.close()


async def example_different_levels():
    """Example: Try different enhancement levels."""
    logger.info("=== Example: Different Enhancement Levels ===")
    
    config = PielMejoradorConfig()
    config.validate()
    
    agent = PielMejoradorAgent(config=config)
    
    levels = ["low", "medium", "high", "ultra"]
    
    for level in levels:
        logger.info(f"Processing with level: {level}")
        
        task_id = await agent.mejorar_imagen(
            file_path="path/to/image.jpg",
            enhancement_level=level,
            priority=0
        )
        
        logger.info(f"Task {level} submitted: {task_id}")
        
        # Wait briefly
        await asyncio.sleep(1)
    
    await agent.close()


async def main():
    """Run all examples."""
    try:
        # Uncomment the example you want to run:
        # await example_mejorar_imagen()
        # await example_mejorar_video()
        # await example_analizar_piel()
        # await example_different_levels()
        
        logger.info("Examples completed")
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())




