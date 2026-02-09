"""
Advanced Usage Examples for Imagen Video Enhancer AI
====================================================

Examples of advanced features and integrations.
"""

import asyncio
import logging
from pathlib import Path

from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig
from imagen_video_enhancer_ai.core.batch_processor import BatchItem
from imagen_video_enhancer_ai.core.webhook_manager import Webhook, WebhookEvent
from imagen_video_enhancer_ai.core.event_bus import Event, EventType
from imagen_video_enhancer_ai.core.metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_metrics_collection():
    """Example: Collecting and analyzing metrics."""
    logger.info("=== Example: Metrics Collection ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    # Record custom metrics
    agent.metrics_collector.record("custom.metric", 42.5, tags={"source": "example"})
    agent.metrics_collector.increment("custom.counter")
    agent.metrics_collector.set_gauge("custom.gauge", 100.0)
    agent.metrics_collector.record_histogram("custom.histogram", 50.0)
    
    # Get statistics
    stats = agent.metrics_collector.get_statistics("custom.metric")
    logger.info(f"Metric statistics: {stats}")
    
    # Get percentiles
    percentiles = agent.metrics_collector.get_percentiles("custom.histogram")
    logger.info(f"Percentiles: {percentiles}")
    
    # Get rate
    rate = agent.metrics_collector.get_rate("custom.metric", window_seconds=60.0)
    logger.info(f"Rate: {rate} events/second")
    
    await agent.close()


async def example_event_subscription():
    """Example: Subscribing to events."""
    logger.info("=== Example: Event Subscription ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    # Define event handler
    async def on_task_completed(event: Event):
        logger.info(f"Task completed: {event.data.get('task_id')}")
    
    async def on_task_failed(event: Event):
        logger.error(f"Task failed: {event.data.get('task_id')} - {event.data.get('error')}")
    
    # Subscribe to events
    agent.event_bus.subscribe(EventType.TASK_COMPLETED, on_task_completed)
    agent.event_bus.subscribe(EventType.TASK_FAILED, on_task_failed)
    
    # Process a task (events will be published automatically)
    # task_id = await agent.enhance_image("image.jpg")
    
    # Get event history
    history = agent.event_bus.get_history(EventType.TASK_COMPLETED, limit=10)
    logger.info(f"Event history: {len(history)} events")
    
    await agent.close()


async def example_webhook_integration():
    """Example: Webhook integration."""
    logger.info("=== Example: Webhook Integration ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    # Register webhook
    webhook = Webhook(
        url="https://example.com/webhook",
        events=[WebhookEvent.TASK_COMPLETED, WebhookEvent.TASK_FAILED],
        secret="your-secret-key"
    )
    
    agent.webhook_manager.register(webhook)
    
    # Process tasks (webhooks will be sent automatically)
    # task_id = await agent.enhance_image("image.jpg")
    
    # Get webhook stats
    stats = agent.webhook_manager.get_stats()
    logger.info(f"Webhook stats: {stats}")
    
    await agent.close()


async def example_batch_with_metrics():
    """Example: Batch processing with metrics."""
    logger.info("=== Example: Batch Processing with Metrics ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    # Create batch items
    batch_items = [
        BatchItem(
            file_path=f"image_{i}.jpg",
            service_type="enhance_image",
            enhancement_type="general"
        )
        for i in range(5)
    ]
    
    # Progress callback with metrics
    async def progress_callback(completed: int, failed: int, total: int):
        agent.metrics_collector.set_gauge("batch.progress", completed / total * 100)
        logger.info(f"Progress: {completed}/{total} completed, {failed} failed")
    
    # Process batch
    result = await agent.process_batch(batch_items, progress_callback=progress_callback)
    
    # Record batch metrics
    agent.metrics_collector.record("batch.duration", result.duration or 0)
    agent.metrics_collector.record("batch.success_rate", result.success_rate)
    
    logger.info(f"Batch completed: {result.completed}/{result.total_items}")
    
    await agent.close()


async def example_comprehensive_monitoring():
    """Example: Comprehensive monitoring setup."""
    logger.info("=== Example: Comprehensive Monitoring ===")
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config)
    
    # Setup event handlers for monitoring
    async def monitor_task_events(event: Event):
        # Record metrics based on events
        if event.event_type == EventType.TASK_COMPLETED:
            agent.metrics_collector.increment("monitoring.tasks.completed")
        elif event.event_type == EventType.TASK_FAILED:
            agent.metrics_collector.increment("monitoring.tasks.failed")
    
    # Subscribe to all events
    agent.event_bus.subscribe(None, monitor_task_events)
    
    # Get comprehensive stats
    stats = agent.get_stats()
    logger.info(f"Comprehensive stats: {stats}")
    
    # Get metrics
    counters = agent.metrics_collector.get_counters()
    gauges = agent.metrics_collector.get_gauges()
    logger.info(f"Counters: {counters}")
    logger.info(f"Gauges: {gauges}")
    
    await agent.close()


if __name__ == "__main__":
    asyncio.run(example_metrics_collection())
    asyncio.sleep(1)
    
    asyncio.run(example_event_subscription())
    asyncio.sleep(1)
    
    asyncio.run(example_webhook_integration())
    asyncio.sleep(1)
    
    asyncio.run(example_batch_with_metrics())
    asyncio.sleep(1)
    
    asyncio.run(example_comprehensive_monitoring())




