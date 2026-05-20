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
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from optimized_video_pipeline import OptimizedVideoPipeline
from optimized_nlp_service import OptimizedNLPService, ProcessingConfig
from optimized_cache_manager import OptimizedCacheManager, CacheConfig
from optimized_async_processor import OptimizedAsyncProcessor, ProcessorConfig, TaskPriority, TaskType
from optimized_performance_monitor import OptimizedPerformanceMonitor, PerformanceConfig
from integrated_app import app
from refactored_architecture import RefactoredOSContentApplication
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Comprehensive test script for the OS Content system
Tests all optimized components and integrations
"""


# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import all components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTester:
    """Comprehensive system tester"""
    
    def __init__(self) -> Any:
        self.results = {}
        self.start_time = time.time()
        
    async def test_video_pipeline(self) -> Dict[str, Any]:
        """Test video pipeline functionality"""
        logger.info("Testing Video Pipeline...")
        start_time = time.time()
        
        try:
            pipeline = OptimizedVideoPipeline(
                device="cuda" if torch.cuda.is_available() else "cpu",
                processing_mode="gpu" if torch.cuda.is_available() else "cpu",
                max_workers=2
            )
            
            # Test video generation
            result = await pipeline.create_video(
                prompt="Beautiful sunset over mountains",
                duration=3,
                output_path="test_video.mp4"
            )
            
            # Get performance stats
            stats = pipeline.get_performance_stats()
            
            await pipeline.close()
            
            return {
                "success": True,
                "result": result,
                "stats": stats,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Video pipeline test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def test_nlp_service(self) -> Dict[str, Any]:
        """Test NLP service functionality"""
        logger.info("Testing NLP Service...")
        start_time = time.time()
        
        try:
            nlp_service = OptimizedNLPService(
                device="cuda" if torch.cuda.is_available() else "cpu",
                config=ProcessingConfig(
                    max_length=256,
                    batch_size=4,
                    use_gpu=True,
                    cache_embeddings=True
                )
            )
            
            # Test single text analysis
            result = await nlp_service.analyze_text("I love this amazing product!")
            
            # Test batch analysis
            texts = ["This is great!", "I hate this product.", "The weather is nice."]
            batch_results = await nlp_service.batch_analyze(texts)
            
            # Test question answering
            qa_result = await nlp_service.answer_question(
                "Where is the Eiffel Tower?",
                "The Eiffel Tower is located in Paris, France."
            )
            
            # Get performance stats
            stats = nlp_service.get_performance_stats()
            
            await nlp_service.close()
            
            return {
                "success": True,
                "single_analysis": result,
                "batch_analysis": batch_results,
                "qa_result": qa_result,
                "stats": stats,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"NLP service test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def test_cache_manager(self) -> Dict[str, Any]:
        """Test cache manager functionality"""
        logger.info("Testing Cache Manager...")
        start_time = time.time()
        
        try:
            cache_manager = OptimizedCacheManager(
                redis_url="redis://localhost:6379",
                config=CacheConfig(
                    max_memory_size=10 * 1024 * 1024,  # 10MB
                    max_disk_size=100 * 1024 * 1024,   # 100MB
                    ttl=300,  # 5 minutes
                    compression="zstd",
                    compression_level=1
                )
            )
            
            # Test basic operations
            await cache_manager.set("test_key", "test_value", ttl=60)
            value = await cache_manager.get("test_key")
            
            # Test complex data
            complex_data = {
                "user": {"id": 123, "name": "John Doe"},
                "items": [1, 2, 3, 4, 5],
                "metadata": {"created": time.time()}
            }
            await cache_manager.set("complex_key", complex_data, ttl=60)
            retrieved_data = await cache_manager.get("complex_key")
            
            # Test batch operations
            items = {
                "batch_key1": "value1",
                "batch_key2": "value2",
                "batch_key3": "value3"
            }
            await cache_manager.batch_set(items)
            batch_results = await cache_manager.batch_get(["batch_key1", "batch_key2", "batch_key3"])
            
            # Get stats
            stats = cache_manager.get_stats()
            
            await cache_manager.close()
            
            return {
                "success": True,
                "basic_operation": value == "test_value",
                "complex_data": retrieved_data == complex_data,
                "batch_operations": len(batch_results) == 3,
                "stats": stats,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Cache manager test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def test_async_processor(self) -> Dict[str, Any]:
        """Test async processor functionality"""
        logger.info("Testing Async Processor...")
        start_time = time.time()
        
        try:
            processor = OptimizedAsyncProcessor(
                config=ProcessorConfig(
                    max_workers=2,
                    max_thread_workers=4,
                    max_process_workers=1,
                    enable_priority_queue=True,
                    enable_auto_scaling=False
                )
            )
            
            await processor.start()
            
            # Test CPU-intensive task
            def cpu_task(n) -> Any:
                result = 0
                for i in range(n):
                    result += i ** 2
                return result
            
            cpu_task_id = await processor.submit_task(
                cpu_task, 100000,
                priority=TaskPriority.HIGH,
                task_type=TaskType.CPU_INTENSIVE
            )
            
            cpu_result = await processor.get_task_result(cpu_task_id)
            
            # Test I/O task
            async def io_task(delay) -> Any:
                await asyncio.sleep(delay)
                return f"IO task completed after {delay}s"
            
            io_task_id = await processor.submit_task(
                lambda: io_task(1),
                priority=TaskPriority.NORMAL,
                task_type=TaskType.IO_INTENSIVE
            )
            
            io_result = await processor.get_task_result(io_task_id)
            
            # Get stats
            stats = processor.get_stats()
            
            await processor.stop()
            
            return {
                "success": True,
                "cpu_task_result": cpu_result,
                "io_task_result": io_result,
                "stats": stats,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Async processor test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def test_performance_monitor(self) -> Dict[str, Any]:
        """Test performance monitor functionality"""
        logger.info("Testing Performance Monitor...")
        start_time = time.time()
        
        try:
            monitor = OptimizedPerformanceMonitor(
                config=PerformanceConfig(
                    collection_interval=1.0,
                    retention_period=300,  # 5 minutes
                    enable_prometheus=True,
                    enable_alerting=True,
                    enable_storage=False  # Disable storage for testing
                )
            )
            
            await monitor.start()
            
            # Monitor for some time
            await asyncio.sleep(5)
            
            # Get metrics
            cpu_metrics = monitor.get_metric("system.cpu.usage")
            memory_metrics = monitor.get_metric("system.memory.usage")
            
            # Get statistics
            cpu_stats = monitor.get_metric_statistics("system.cpu.usage")
            memory_stats = monitor.get_metric_statistics("system.memory.usage")
            
            # Get alerts
            alerts = monitor.get_alerts()
            
            await monitor.stop()
            
            return {
                "success": True,
                "cpu_metrics_count": len(cpu_metrics),
                "memory_metrics_count": len(memory_metrics),
                "cpu_stats": cpu_stats,
                "memory_stats": memory_stats,
                "alerts_count": len(alerts),
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Performance monitor test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def test_integrated_app(self) -> Dict[str, Any]:
        """Test integrated application"""
        logger.info("Testing Integrated Application...")
        start_time = time.time()
        
        try:
            # Test application initialization
            app_instance = RefactoredOSContentApplication()
            await app_instance.initialize()
            
            # Get use cases
            video_use_case = app_instance.get_video_use_case()
            nlp_use_case = app_instance.get_nlp_use_case()
            cache_use_case = app_instance.get_cache_use_case()
            perf_use_case = app_instance.get_performance_use_case()
            
            # Test use case availability
            use_cases_available = all([
                video_use_case is not None,
                nlp_use_case is not None,
                cache_use_case is not None,
                perf_use_case is not None
            ])
            
            await app_instance.shutdown()
            
            return {
                "success": True,
                "use_cases_available": use_cases_available,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Integrated app test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        logger.info("Starting comprehensive system tests...")
        
        tests = [
            ("video_pipeline", self.test_video_pipeline),
            ("nlp_service", self.test_nlp_service),
            ("cache_manager", self.test_cache_manager),
            ("async_processor", self.test_async_processor),
            ("performance_monitor", self.test_performance_monitor),
            ("integrated_app", self.test_integrated_app)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name} test...")
            self.results[test_name] = await test_func()
            await asyncio.sleep(1)  # Brief pause between tests
        
        total_duration = time.time() - self.start_time
        
        # Calculate summary
        successful_tests = sum(1 for result in self.results.values() if result["success"])
        total_tests = len(self.results)
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests) * 100,
                "total_duration": total_duration
            },
            "results": self.results
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results"""
        print("\n" + "="*60)
        print("OS CONTENT SYSTEM TEST RESULTS")
        print("="*60)
        
        summary = results["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Successful: {summary['successful_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Total Duration: {summary['total_duration']:.2f}s")
        
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in results["results"].items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            duration = f"{result['duration']:.2f}s"
            print(f"   {test_name:20} {status} ({duration})")
            
            if not result["success"]:
                print(f"      Error: {result['error']}")
        
        print("\n" + "="*60)
        
        # Save results to file
        with open("test_results.json", "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        print("Results saved to test_results.json")

async def main():
    """Main test function"""
    tester = SystemTester()
    
    try:
        results = await tester.run_all_tests()
        tester.print_results(results)
        
        # Exit with error code if any tests failed
        if results["summary"]["failed_tests"] > 0:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        sys.exit(1)

match __name__:
    case "__main__":
    asyncio.run(main()) 