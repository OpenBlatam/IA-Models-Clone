from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
from pathlib import Path
import sys
from optimized_video_pipeline import OptimizedVideoPipeline
from optimized_nlp_service import OptimizedNLPService, ProcessingConfig
from optimized_cache_manager import OptimizedCacheManager, CacheConfig
from optimized_async_processor import OptimizedAsyncProcessor, ProcessorConfig, TaskPriority, TaskType
from optimized_performance_monitor import OptimizedPerformanceMonitor, PerformanceConfig
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Run optimized components directly for testing and development
"""


# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_video_pipeline():
    """Test optimized video pipeline"""
    logger.info("Testing Video Pipeline...")
    
    pipeline = OptimizedVideoPipeline(
        device="cuda" if torch.cuda.is_available() else "cpu",
        processing_mode="gpu" if torch.cuda.is_available() else "cpu",
        max_workers=4
    )
    
    try:
        result = await pipeline.create_video(
            prompt="Beautiful sunset over mountains",
            duration=5,
            output_path="test_output.mp4"
        )
        
        logger.info(f"Video created: {result}")
        
        stats = pipeline.get_performance_stats()
        logger.info(f"Video pipeline stats: {stats}")
        
    finally:
        await pipeline.close()

async def test_nlp_service():
    """Test optimized NLP service"""
    logger.info("Testing NLP Service...")
    
    nlp_service = OptimizedNLPService(
        device="cuda" if torch.cuda.is_available() else "cpu",
        config=ProcessingConfig(
            max_length=512,
            batch_size=8,
            use_gpu=True,
            cache_embeddings=True,
            parallel_processing=True
        )
    )
    
    try:
        # Test single text analysis
        result = await nlp_service.analyze_text("I love this amazing product!")
        logger.info(f"NLP analysis result: {result}")
        
        # Test batch analysis
        texts = [
            "This is great!",
            "I hate this product.",
            "The weather is nice today."
        ]
        
        batch_results = await nlp_service.batch_analyze(texts)
        logger.info(f"Batch analysis completed: {len(batch_results)} texts")
        
        # Test question answering
        qa_result = await nlp_service.answer_question(
            "Where is the Eiffel Tower?",
            "The Eiffel Tower is located in Paris, France."
        )
        logger.info(f"Q&A result: {qa_result}")
        
        stats = nlp_service.get_performance_stats()
        logger.info(f"NLP service stats: {stats}")
        
    finally:
        await nlp_service.close()

async def test_cache_manager():
    """Test optimized cache manager"""
    logger.info("Testing Cache Manager...")
    
    cache_manager = OptimizedCacheManager(
        redis_url="redis://localhost:6379",
        config=CacheConfig(
            max_memory_size=50 * 1024 * 1024,  # 50MB
            max_disk_size=500 * 1024 * 1024,   # 500MB
            ttl=1800,  # 30 minutes
            compression="zstd",
            compression_level=3,
            enable_stats=True,
            enable_eviction=True,
            eviction_policy="lru"
        )
    )
    
    try:
        # Test basic operations
        await cache_manager.set("test_key", "test_value", ttl=3600)
        value = await cache_manager.get("test_key")
        logger.info(f"Cache get result: {value}")
        
        # Test batch operations
        items = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        await cache_manager.batch_set(items)
        results = await cache_manager.batch_get(["key1", "key2", "key3"])
        logger.info(f"Batch cache results: {results}")
        
        stats = cache_manager.get_stats()
        logger.info(f"Cache stats: {stats}")
        
    finally:
        await cache_manager.close()

async def test_async_processor():
    """Test optimized async processor"""
    logger.info("Testing Async Processor...")
    
    processor = OptimizedAsyncProcessor(
        config=ProcessorConfig(
            max_workers=4,
            max_thread_workers=10,
            max_process_workers=2,
            enable_priority_queue=True,
            enable_auto_scaling=True,
            enable_monitoring=True
        )
    )
    
    try:
        await processor.start()
        
        # Test CPU-intensive task
        def cpu_task(n) -> Any:
            result = 0
            for i in range(n):
                result += i ** 2
            return result
        
        task_id = await processor.submit_task(
            cpu_task, 1000000,
            priority=TaskPriority.HIGH,
            task_type=TaskType.CPU_INTENSIVE
        )
        
        result = await processor.get_task_result(task_id)
        logger.info(f"CPU task result: {result}")
        
        # Test I/O task
        async def io_task(delay) -> Any:
            await asyncio.sleep(delay)
            return f"IO task completed after {delay}s"
        
        io_task_id = await processor.submit_task(
            lambda: io_task(2),
            priority=TaskPriority.NORMAL,
            task_type=TaskType.IO_INTENSIVE
        )
        
        io_result = await processor.get_task_result(io_task_id)
        logger.info(f"IO task result: {io_result}")
        
        stats = processor.get_stats()
        logger.info(f"Processor stats: {stats}")
        
    finally:
        await processor.stop()

async def test_performance_monitor():
    """Test optimized performance monitor"""
    logger.info("Testing Performance Monitor...")
    
    monitor = OptimizedPerformanceMonitor(
        config=PerformanceConfig(
            collection_interval=2.0,
            retention_period=3600,  # 1 hour
            enable_prometheus=True,
            enable_alerting=True,
            enable_storage=True,
            alert_thresholds={
                'system.cpu.usage': {'warning': 80.0, 'error': 90.0},
                'system.memory.usage': {'warning': 85.0, 'error': 95.0}
            }
        )
    )
    
    try:
        await monitor.start()
        
        # Monitor for some time
        await asyncio.sleep(10)
        
        # Get metrics
        metrics = monitor.get_metric("system.cpu.usage")
        logger.info(f"CPU usage metrics: {len(metrics)} data points")
        
        # Get statistics
        stats = monitor.get_metric_statistics("system.cpu.usage")
        logger.info(f"CPU usage statistics: {stats}")
        
        # Get alerts
        alerts = monitor.get_alerts()
        logger.info(f"Active alerts: {len(alerts)}")
        
        # Generate report
        report = monitor.generate_report()
        logger.info("Performance report generated")
        
    finally:
        await monitor.stop()

async def run_all_tests():
    """Run all component tests"""
    logger.info("Starting all component tests...")
    
    start_time = time.time()
    
    try:
        # Test cache manager first (others may depend on it)
        await test_cache_manager()
        
        # Test other components
        await test_nlp_service()
        await test_async_processor()
        await test_performance_monitor()
        await test_video_pipeline()
        
        total_time = time.time() - start_time
        logger.info(f"All tests completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

match __name__:
    case "__main__":
    asyncio.run(run_all_tests()) 