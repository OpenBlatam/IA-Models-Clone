"""
Comprehensive Test Suite for Ultimate Opus Clip Improvements

This script tests all the improvements and new features implemented
in the Ultimate Opus Clip system.
"""

import asyncio
import time
import sys
import json
from pathlib import Path
import structlog
import requests
from typing import Dict, List, Any

# Import our new modules
from intelligent_cache import IntelligentCache, CacheConfig, CacheType
from batch_processor import BatchProcessor, BatchConfig, ProcessingPriority
from monitoring_dashboard import MonitoringDashboard
from system_improvements import SystemImprovements

logger = structlog.get_logger("test_all_improvements")

class ComprehensiveTester:
    """Comprehensive test suite for all improvements."""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.test_results = {}
        self.cache = None
        self.batch_processor = None
        self.dashboard = None
        self.system_improvements = None
    
    async def setup_test_environment(self):
        """Setup the test environment."""
        try:
            logger.info("Setting up test environment...")
            
            # Initialize cache
            cache_config = CacheConfig(
                max_size_mb=512,
                max_entries=1000,
                default_ttl=3600
            )
            self.cache = IntelligentCache(cache_config)
            
            # Initialize batch processor
            batch_config = BatchConfig(
                max_concurrent_jobs=2,
                max_queue_size=100,
                timeout_seconds=300
            )
            self.batch_processor = BatchProcessor(batch_config)
            
            # Initialize monitoring dashboard
            self.dashboard = MonitoringDashboard()
            
            # Initialize system improvements
            self.system_improvements = SystemImprovements()
            
            logger.info("Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False
    
    async def test_intelligent_cache(self):
        """Test the intelligent cache system."""
        try:
            logger.info("Testing intelligent cache system...")
            
            # Test basic cache operations
            test_key = "test_key_1"
            test_value = {"data": "test_value", "timestamp": time.time()}
            
            # Test set
            success = self.cache.set(test_key, test_value, CacheType.PROCESSING_RESULTS)
            if not success:
                self.test_results["cache_set"] = "FAIL"
                return
            
            # Test get
            retrieved_value = self.cache.get(test_key)
            if retrieved_value != test_value:
                self.test_results["cache_get"] = "FAIL"
                return
            
            # Test cache stats
            stats = self.cache.get_stats()
            if not isinstance(stats, dict) or "total_entries" not in stats:
                self.test_results["cache_stats"] = "FAIL"
                return
            
            # Test cache decorator
            @self.cache.cache_function(CacheType.MODEL_PREDICTIONS, ttl=60)
            def test_function(x, y):
                return x + y
            
            result1 = test_function(1, 2)
            result2 = test_function(1, 2)  # Should come from cache
            
            if result1 != result2 or result1 != 3:
                self.test_results["cache_decorator"] = "FAIL"
                return
            
            self.test_results["intelligent_cache"] = "PASS"
            logger.info("Intelligent cache test passed")
            
        except Exception as e:
            self.test_results["intelligent_cache"] = "ERROR"
            logger.error(f"Intelligent cache test error: {e}")
    
    async def test_batch_processor(self):
        """Test the batch processor system."""
        try:
            logger.info("Testing batch processor system...")
            
            # Start batch processor
            self.batch_processor.start()
            
            # Add test jobs
            test_videos = [
                "test_video_1.mp4",
                "test_video_2.mp4",
                "test_video_3.mp4"
            ]
            
            job_ids = self.batch_processor.add_batch(
                test_videos,
                {"quality": "high", "platform": "tiktok"},
                ProcessingPriority.NORMAL
            )
            
            if len(job_ids) != len(test_videos):
                self.test_results["batch_add_jobs"] = "FAIL"
                return
            
            # Wait for jobs to complete
            max_wait_time = 60  # 1 minute
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status = self.batch_processor.get_batch_status()
                
                if status["completed_jobs"] == len(test_videos):
                    break
                
                await asyncio.sleep(1)
            
            # Check final status
            final_status = self.batch_processor.get_batch_status()
            if final_status["completed_jobs"] != len(test_videos):
                self.test_results["batch_processing"] = "FAIL"
                return
            
            # Test job status retrieval
            job_status = self.batch_processor.get_job_status(job_ids[0])
            if not job_status or job_status["status"] != "completed":
                self.test_results["batch_job_status"] = "FAIL"
                return
            
            # Test results retrieval
            results = self.batch_processor.get_completed_results()
            if len(results) != len(test_videos):
                self.test_results["batch_results"] = "FAIL"
                return
            
            self.batch_processor.stop()
            self.test_results["batch_processor"] = "PASS"
            logger.info("Batch processor test passed")
            
        except Exception as e:
            self.test_results["batch_processor"] = "ERROR"
            logger.error(f"Batch processor test error: {e}")
    
    async def test_monitoring_dashboard(self):
        """Test the monitoring dashboard system."""
        try:
            logger.info("Testing monitoring dashboard system...")
            
            # Start monitoring
            self.dashboard.start_monitoring()
            
            # Wait for some metrics to be collected
            await asyncio.sleep(5)
            
            # Test system status
            status = self.dashboard.get_system_status()
            if not isinstance(status, dict) or "cpu_usage" not in status:
                self.test_results["dashboard_status"] = "FAIL"
                return
            
            # Test metrics data
            metrics = self.dashboard.get_metrics_data(limit=10)
            if not isinstance(metrics, list):
                self.test_results["dashboard_metrics"] = "FAIL"
                return
            
            # Test specific metric type
            cpu_metrics = self.dashboard.get_metrics_data("cpu_usage", limit=5)
            if not isinstance(cpu_metrics, list):
                self.test_results["dashboard_cpu_metrics"] = "FAIL"
                return
            
            self.dashboard.stop_monitoring()
            self.test_results["monitoring_dashboard"] = "PASS"
            logger.info("Monitoring dashboard test passed")
            
        except Exception as e:
            self.test_results["monitoring_dashboard"] = "ERROR"
            logger.error(f"Monitoring dashboard test error: {e}")
    
    async def test_system_improvements(self):
        """Test the system improvements."""
        try:
            logger.info("Testing system improvements...")
            
            # Test improvement application
            await self.system_improvements.apply_performance_improvements()
            
            # Test improvement status
            status = self.system_improvements.get_improvement_status()
            if not isinstance(status, dict) or "total_improvements" not in status:
                self.test_results["system_improvements"] = "FAIL"
                return
            
            self.test_results["system_improvements"] = "PASS"
            logger.info("System improvements test passed")
            
        except Exception as e:
            self.test_results["system_improvements"] = "ERROR"
            logger.error(f"System improvements test error: {e}")
    
    async def test_api_integration(self):
        """Test API integration with new features."""
        try:
            logger.info("Testing API integration...")
            
            # Test health endpoint
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=10)
                if response.status_code == 200:
                    self.test_results["api_health"] = "PASS"
                else:
                    self.test_results["api_health"] = "FAIL"
            except Exception:
                self.test_results["api_health"] = "SKIP"  # API might not be running
            
            # Test cache integration (if API supports it)
            try:
                response = requests.get(f"{self.api_base_url}/cache/stats", timeout=10)
                if response.status_code == 200:
                    self.test_results["api_cache_integration"] = "PASS"
                else:
                    self.test_results["api_cache_integration"] = "SKIP"
            except Exception:
                self.test_results["api_cache_integration"] = "SKIP"
            
            # Test batch processing integration (if API supports it)
            try:
                response = requests.get(f"{self.api_base_url}/batch/status", timeout=10)
                if response.status_code == 200:
                    self.test_results["api_batch_integration"] = "PASS"
                else:
                    self.test_results["api_batch_integration"] = "SKIP"
            except Exception:
                self.test_results["api_batch_integration"] = "SKIP"
            
            logger.info("API integration test completed")
            
        except Exception as e:
            logger.error(f"API integration test error: {e}")
    
    async def test_performance_improvements(self):
        """Test performance improvements."""
        try:
            logger.info("Testing performance improvements...")
            
            # Test cache performance
            start_time = time.time()
            
            # Add many cache entries
            for i in range(100):
                self.cache.set(f"perf_test_{i}", {"data": f"value_{i}"}, CacheType.TEMPORARY)
            
            # Retrieve many cache entries
            for i in range(100):
                self.cache.get(f"perf_test_{i}")
            
            cache_time = time.time() - start_time
            
            if cache_time < 1.0:  # Should be very fast
                self.test_results["cache_performance"] = "PASS"
            else:
                self.test_results["cache_performance"] = "FAIL"
            
            # Test batch processing performance
            start_time = time.time()
            
            # Add batch of jobs
            test_videos = [f"perf_video_{i}.mp4" for i in range(10)]
            self.batch_processor.add_batch(
                test_videos,
                {"quality": "medium"},
                ProcessingPriority.LOW
            )
            
            batch_setup_time = time.time() - start_time
            
            if batch_setup_time < 0.5:  # Should be very fast
                self.test_results["batch_performance"] = "PASS"
            else:
                self.test_results["batch_performance"] = "FAIL"
            
            logger.info("Performance improvements test completed")
            
        except Exception as e:
            logger.error(f"Performance improvements test error: {e}")
    
    async def run_all_tests(self):
        """Run all comprehensive tests."""
        logger.info("Starting comprehensive Ultimate Opus Clip improvement tests...")
        
        # Setup
        if not await self.setup_test_environment():
            logger.error("Failed to setup test environment")
            return
        
        # Run tests
        tests = [
            self.test_intelligent_cache,
            self.test_batch_processor,
            self.test_monitoring_dashboard,
            self.test_system_improvements,
            self.test_api_integration,
            self.test_performance_improvements
        ]
        
        for test in tests:
            try:
                await test()
                await asyncio.sleep(1)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
        
        # Cleanup
        await self.cleanup_test_environment()
        
        # Print results
        self.print_comprehensive_results()
    
    async def cleanup_test_environment(self):
        """Cleanup test environment."""
        try:
            if self.cache:
                self.cache.cleanup()
            
            if self.batch_processor:
                self.batch_processor.cleanup()
            
            if self.dashboard:
                self.dashboard.stop_monitoring()
            
            logger.info("Test environment cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def print_comprehensive_results(self):
        """Print comprehensive test results."""
        logger.info("=" * 60)
        logger.info("ULTIMATE OPUS CLIP - COMPREHENSIVE IMPROVEMENT TEST RESULTS")
        logger.info("=" * 60)
        
        # Categorize results
        categories = {
            "Core Systems": [
                "intelligent_cache",
                "batch_processor",
                "monitoring_dashboard",
                "system_improvements"
            ],
            "API Integration": [
                "api_health",
                "api_cache_integration",
                "api_batch_integration"
            ],
            "Performance": [
                "cache_performance",
                "batch_performance"
            ]
        }
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASS")
        failed_tests = sum(1 for result in self.test_results.values() if result == "FAIL")
        error_tests = sum(1 for result in self.test_results.values() if result == "ERROR")
        skipped_tests = sum(1 for result in self.test_results.values() if result == "SKIP")
        
        # Print results by category
        for category, tests in categories.items():
            logger.info(f"\n{category}:")
            logger.info("-" * 40)
            
            for test in tests:
                if test in self.test_results:
                    result = self.test_results[test]
                    status_icon = "✅" if result == "PASS" else "❌" if result == "FAIL" else "⚠️" if result == "ERROR" else "⏭️"
                    logger.info(f"{status_icon} {test}: {result}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Errors: {error_tests}")
        logger.info(f"Skipped: {skipped_tests}")
        
        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
            logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Print recommendations
        logger.info("\nRECOMMENDATIONS:")
        logger.info("-" * 40)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! All improvements are working correctly.")
            logger.info("✅ The system is ready for production use.")
        elif passed_tests > total_tests * 0.8:
            logger.info("⚠️ Most tests passed. Some improvements may need minor attention.")
            logger.info("✅ The system is mostly ready for production use.")
        elif passed_tests > total_tests * 0.5:
            logger.info("⚠️ Some tests failed. Several improvements need attention.")
            logger.info("🔧 Review failed tests and fix issues before production.")
        else:
            logger.error("❌ Many tests failed. Significant work needed.")
            logger.error("🔧 Fix major issues before considering production use.")
        
        logger.info("=" * 60)

async def main():
    """Main test function."""
    tester = ComprehensiveTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())


