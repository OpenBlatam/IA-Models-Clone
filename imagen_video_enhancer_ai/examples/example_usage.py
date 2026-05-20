"""
Example usage of Imagen Video Enhancer AI
=========================================
"""

import asyncio
import logging
from pathlib import Path

from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_enhance_image():
    """Example: Enhance an image."""
    logger.info("=== Example: Enhance Image ===")
    
    # Create configuration
    config = EnhancerConfig()
    
    # Create agent
    agent = EnhancerAgent(config=config)
    
    # Submit enhancement task
    task_id = await agent.enhance_image(
        file_path="path/to/image.jpg",
        enhancement_type="general",
        options={
            "quality": "high",
            "preserve_details": True
        },
        priority=5
    )
    
    logger.info(f"Task created: {task_id}")
    
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


async def example_enhance_video():
    """Example: Enhance a video."""
    logger.info("=== Example: Enhance Video ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    task_id = await agent.enhance_video(
        file_path="path/to/video.mp4",
        enhancement_type="general",
        options={
            "fps": 60,
            "resolution": "4k"
        }
    )
    
    logger.info(f"Task created: {task_id}")
    
    # Check status
    status = await agent.get_task_status(task_id)
    logger.info(f"Status: {status}")
    
    await agent.close()


async def example_upscale():
    """Example: Upscale an image."""
    logger.info("=== Example: Upscale ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    task_id = await agent.upscale(
        file_path="path/to/image.jpg",
        scale_factor=4,  # 4x upscale
        options={
            "method": "ai",
            "preserve_details": True
        }
    )
    
    logger.info(f"Task created: {task_id}")
    
    await agent.close()


async def example_multiple_tasks():
    """Example: Submit multiple tasks."""
    logger.info("=== Example: Multiple Tasks ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    # Submit multiple tasks with different priorities
    tasks = [
        await agent.enhance_image("image1.jpg", priority=10),
        await agent.denoise("image2.jpg", noise_level="high", priority=5),
        await agent.color_correction("image3.jpg", priority=0),
    ]
    
    logger.info(f"Submitted {len(tasks)} tasks")
    
    # Monitor all tasks
    for task_id in tasks:
        status = await agent.get_task_status(task_id)
        logger.info(f"Task {task_id}: {status['status']}")
    
    await agent.close()


async def example_batch_processing():
    """Example: Batch processing."""
    logger.info("=== Example: Batch Processing ===")
    
    from imagen_video_enhancer_ai.core.batch_processor import BatchItem
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    # Create batch items
    batch_items = [
        BatchItem(
            file_path="image1.jpg",
            service_type="enhance_image",
            enhancement_type="general",
            priority=5
        ),
        BatchItem(
            file_path="image2.jpg",
            service_type="upscale",
            options={"scale_factor": 2},
            priority=3
        ),
        BatchItem(
            file_path="image3.jpg",
            service_type="denoise",
            options={"noise_level": "medium"},
            priority=1
        ),
    ]
    
    # Progress callback
    async def progress_callback(completed: int, failed: int, total: int):
        logger.info(f"Progress: {completed}/{total} completed, {failed} failed")
    
    # Process batch
    result = await agent.process_batch(batch_items, progress_callback=progress_callback)
    
    logger.info(f"Batch completed: {result.completed} succeeded, {result.failed} failed")
    logger.info(f"Success rate: {result.success_rate:.2%}")
    logger.info(f"Duration: {result.duration:.2f}s")
    
    await agent.close()


async def example_stats():
    """Example: Get statistics."""
    logger.info("=== Example: Statistics ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    # Get stats
    stats = agent.get_stats()
    logger.info(f"Stats: {stats}")
    
    await agent.close()


async def main():
    """Run all examples."""
    try:
        await example_enhance_image()
        await asyncio.sleep(1)
        
        await example_enhance_video()
        await asyncio.sleep(1)
        
        await example_upscale()
        await asyncio.sleep(1)
        
        await example_multiple_tasks()
        await asyncio.sleep(1)
        
        await example_batch_processing()
        await asyncio.sleep(1)
        
        await example_stats()
        
    except Exception as e:
        logger.error(f"Error in examples: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

