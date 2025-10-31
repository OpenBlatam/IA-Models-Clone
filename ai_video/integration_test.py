from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
import time
from typing import Dict, Any, List
from pathlib import Path
    from refactored_optimization_system import (
    from refactored_workflow_engine import (
                import numpy as np
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Integration Test for Refactored Optimization System and Workflow Engine

This test validates that all components work together correctly in a real-world scenario.
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
        OptimizationManager, create_optimization_manager,
        monitor_performance, retry_on_failure,
        OptimizationError, LibraryNotAvailableError
    )
        RefactoredWorkflowEngine, create_workflow_engine,
        WorkflowState, WorkflowStatus
    )
    REFACTORED_AVAILABLE = True
except ImportError as e:
    logger.error(f"Refactored systems not available: {e}")
    REFACTORED_AVAILABLE = False


class IntegrationTestSuite:
    """Comprehensive integration test suite for the refactored system."""
    
    def __init__(self) -> Any:
        self.test_results = {}
        self.optimization_manager = None
        self.workflow_engine = None
        
    async def setup_systems(self) -> Any:
        """Initialize all systems for testing."""
        logger.info("Setting up integration test systems...")
        
        # Create optimization manager
        config = {
            "numba": {"enabled": True},
            "dask": {"n_workers": 4, "threads_per_worker": 2},
            "redis": {"host": "localhost", "port": 6379, "db": 0},
            "prometheus": {"port": 8001},  # Different port to avoid conflicts
            "ray": {"enabled": False},  # Disable for testing
            "optuna": {"enabled": False}  # Disable for testing
        }
        
        self.optimization_manager = create_optimization_manager(config)
        init_results = self.optimization_manager.initialize_all()
        logger.info(f"Optimization manager initialization: {init_results}")
        
        # Create workflow engine
        workflow_config = {
            "optimization_manager": self.optimization_manager,
            "cache_ttl": 3600,
            "max_retries": 3
        }
        
        self.workflow_engine = create_workflow_engine(workflow_config)
        await self.workflow_engine.initialize()
        
        return init_results
    
    @monitor_performance
    async def test_optimization_system_integration(self) -> Any:
        """Test that all optimization components work together."""
        logger.info("Testing optimization system integration...")
        
        results = {}
        
        # Test Numba with numerical data
        numba_optimizer = self.optimization_manager.get_optimizer("numba")
        if numba_optimizer and numba_optimizer.is_available():
            try:
                
                def test_function(x, y) -> Any:
                    return np.sqrt(x**2 + y**2)
                
                compiled_func = numba_optimizer.compile_function(test_function)
                result = compiled_func(3.0, 4.0)
                
                results["numba"] = {
                    "status": "success",
                    "result": result,
                    "expected": 5.0,
                    "correct": abs(result - 5.0) < 1e-6
                }
            except Exception as e:
                results["numba"] = {"status": "failed", "error": str(e)}
        else:
            results["numba"] = {"status": "not_available"}
        
        # Test Dask parallel processing
        dask_optimizer = self.optimization_manager.get_optimizer("dask")
        if dask_optimizer and dask_optimizer.is_available():
            try:
                def process_item(item) -> Any:
                    return item * 2
                
                test_data = [1, 2, 3, 4, 5]
                results_dask = dask_optimizer.parallel_processing(process_item, test_data)
                
                results["dask"] = {
                    "status": "success",
                    "input": test_data,
                    "output": results_dask,
                    "expected": [2, 4, 6, 8, 10],
                    "correct": results_dask == [2, 4, 6, 8, 10]
                }
            except Exception as e:
                results["dask"] = {"status": "failed", "error": str(e)}
        else:
            results["dask"] = {"status": "not_available"}
        
        # Test Redis caching
        redis_optimizer = self.optimization_manager.get_optimizer("redis")
        if redis_optimizer and redis_optimizer.is_available():
            try:
                test_data = {"test": "data", "timestamp": time.time()}
                cache_key = "integration_test_key"
                
                # Set cache
                set_success = redis_optimizer.set(cache_key, test_data, ttl=60)
                
                # Get cache
                cached_data = redis_optimizer.get(cache_key)
                
                results["redis"] = {
                    "status": "success",
                    "set_success": set_success,
                    "cached_data": cached_data,
                    "cache_hit": cached_data is not None,
                    "data_match": cached_data == test_data if cached_data else False
                }
            except Exception as e:
                results["redis"] = {"status": "failed", "error": str(e)}
        else:
            results["redis"] = {"status": "not_available"}
        
        # Test Prometheus metrics
        prometheus_optimizer = self.optimization_manager.get_optimizer("prometheus")
        if prometheus_optimizer and prometheus_optimizer.is_available():
            try:
                # Record metrics with correct labels
                prometheus_optimizer.record_metric("duration_seconds", 1.5, {"optimizer": "integration_test"})
                prometheus_optimizer.record_metric("requests_total", 1, {"optimizer": "integration_test", "status": "success"})
                
                results["prometheus"] = {
                    "status": "success",
                    "metrics_recorded": True,
                    "port": prometheus_optimizer.port
                }
            except Exception as e:
                results["prometheus"] = {"status": "failed", "error": str(e)}
        else:
            results["prometheus"] = {"status": "not_available"}
        
        return results
    
    @monitor_performance
    async def test_workflow_engine_integration(self) -> Any:
        """Test that the workflow engine integrates with all optimizers."""
        logger.info("Testing workflow engine integration...")
        
        results = {}
        
        # Test single workflow execution
        try:
            workflow_result = await self.workflow_engine.execute_workflow(
                url="https://integration-test.com",
                workflow_id="integration_test_001",
                avatar="test_avatar",
                user_edits={"quality": "high", "format": "mp4"}
            )
            
            results["single_workflow"] = {
                "status": "success",
                "workflow_status": workflow_result.status.value,
                "video_url": workflow_result.video_url,
                "optimizations_used": workflow_result.optimizations_used,
                "cache_hits": workflow_result.metrics.cache_hits,
                "cache_misses": workflow_result.metrics.cache_misses,
                "duration": workflow_result.metrics.duration,
                "has_content": workflow_result.content is not None,
                "has_suggestions": workflow_result.suggestions is not None
            }
        except Exception as e:
            results["single_workflow"] = {"status": "failed", "error": str(e)}
        
        # Test batch workflow execution
        try:
            batch_configs = [
                {
                    "url": f"https://batch-test-{i}.com",
                    "workflow_id": f"batch_integration_{i:03d}",
                    "avatar": f"avatar_{i}",
                    "user_edits": {"quality": "medium", "format": "webm"}
                }
                for i in range(3)
            ]
            
            batch_results = await self.workflow_engine.execute_batch_workflows(batch_configs)
            
            results["batch_workflow"] = {
                "status": "success",
                "batch_size": len(batch_results),
                "successful_workflows": sum(1 for r in batch_results if r.status == WorkflowStatus.COMPLETED),
                "failed_workflows": sum(1 for r in batch_results if r.status == WorkflowStatus.FAILED),
                "total_duration": sum(r.metrics.duration or 0 for r in batch_results),
                "total_cache_hits": sum(r.metrics.cache_hits for r in batch_results),
                "total_cache_misses": sum(r.metrics.cache_misses for r in batch_results)
            }
        except Exception as e:
            results["batch_workflow"] = {"status": "failed", "error": str(e)}
        
        return results
    
    @monitor_performance
    async def test_error_handling_integration(self) -> Any:
        """Test error handling across the entire system."""
        logger.info("Testing error handling integration...")
        
        results = {}
        
        # Test optimization error handling
        try:
            # Try to use an unavailable optimizer
            unavailable_optimizer = self.optimization_manager.get_optimizer("nonexistent")
            if unavailable_optimizer is None:
                results["optimization_errors"] = {
                    "status": "success",
                    "handled": True,
                    "message": "Gracefully handled missing optimizer"
                }
        except Exception as e:
            results["optimization_errors"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test workflow error handling
        try:
            # Try to execute workflow with invalid URL
            await self.workflow_engine.execute_workflow(
                url="invalid-url",
                workflow_id="error_test_001"
            )
            results["workflow_errors"] = {
                "status": "failed",
                "message": "Should have failed with invalid URL"
            }
        except Exception as e:
            results["workflow_errors"] = {
                "status": "success",
                "handled": True,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        
        return results
    
    @monitor_performance
    async def test_performance_monitoring_integration(self) -> Any:
        """Test performance monitoring across the system."""
        logger.info("Testing performance monitoring integration...")
        
        results = {}
        
        # Test decorated function performance
        @monitor_performance
        def test_performance_function(data) -> Any:
            time.sleep(0.1)  # Simulate work
            return [x * 2 for x in data]
        
        try:
            test_data = list(range(10))
            result = test_performance_function(test_data)
            
            results["performance_monitoring"] = {
                "status": "success",
                "input": test_data,
                "output": result,
                "expected": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                "correct": result == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
            }
        except Exception as e:
            results["performance_monitoring"] = {"status": "failed", "error": str(e)}
        
        # Test retry mechanism
        @retry_on_failure(max_retries=2, delay=0.1)
        def test_retry_function(should_fail=False) -> Any:
            if should_fail:
                raise ValueError("Simulated failure")
            return "success"
        
        try:
            # Test successful execution
            success_result = test_retry_function(should_fail=False)
            
            # Test retry mechanism
            try:
                test_retry_function(should_fail=True)
                results["retry_mechanism"] = {
                    "status": "failed",
                    "message": "Should have failed after retries"
                }
            except Exception as e:
                results["retry_mechanism"] = {
                    "status": "success",
                    "success_result": success_result,
                    "failure_handled": True,
                    "final_error": str(e)
                }
        except Exception as e:
            results["retry_mechanism"] = {"status": "failed", "error": str(e)}
        
        return results
    
    async def run_comprehensive_integration_test(self) -> Any:
        """Run all integration tests."""
        logger.info("Starting comprehensive integration test...")
        
        if not REFACTORED_AVAILABLE:
            logger.error("Refactored systems not available")
            return {"error": "Refactored systems not available"}
        
        # Setup systems
        setup_results = await self.setup_systems()
        self.test_results["setup"] = setup_results
        
        # Run integration tests
        self.test_results["optimization_integration"] = await self.test_optimization_system_integration()
        self.test_results["workflow_integration"] = await self.test_workflow_engine_integration()
        self.test_results["error_handling_integration"] = await self.test_error_handling_integration()
        self.test_results["performance_monitoring_integration"] = await self.test_performance_monitoring_integration()
        
        # Get system status
        if self.optimization_manager:
            self.test_results["optimization_status"] = self.optimization_manager.get_status()
        
        if self.workflow_engine:
            self.test_results["workflow_status"] = self.workflow_engine.get_status()
        
        # Calculate test summary
        self.test_results["summary"] = self._calculate_test_summary()
        
        return self.test_results
    
    def _calculate_test_summary(self) -> Any:
        """Calculate overall test summary."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category, results in self.test_results.items():
            if category in ["setup", "optimization_status", "workflow_status", "summary"]:
                continue
            
            if isinstance(results, dict):
                for test_name, result in results.items():
                    total_tests += 1
                    if isinstance(result, dict) and result.get("status") == "success":
                        passed_tests += 1
                    else:
                        failed_tests += 1
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "overall_status": "PASS" if failed_tests == 0 else "FAIL"
        }
    
    def cleanup(self) -> Any:
        """Cleanup test resources."""
        if self.workflow_engine:
            self.workflow_engine.cleanup()
        
        if self.optimization_manager:
            self.optimization_manager.cleanup_all()
        
        logger.info("Integration test cleanup completed")


async def main():
    """Main function to run integration tests."""
    test_suite = IntegrationTestSuite()
    
    try:
        results = await test_suite.run_comprehensive_integration_test()
        
        # Save results
        with open("integration_test_results.json", "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        summary = results.get("summary", {})
        logger.info("=== INTEGRATION TEST SUMMARY ===")
        logger.info(f"Total Tests: {summary.get('total_tests', 0)}")
        logger.info(f"Passed: {summary.get('passed_tests', 0)}")
        logger.info(f"Failed: {summary.get('failed_tests', 0)}")
        logger.info(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        logger.info(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        
        # Print detailed results
        for category, category_results in results.items():
            if category in ["setup", "optimization_status", "workflow_status", "summary"]:
                continue
            
            logger.info(f"\n--- {category.upper()} ---")
            for test_name, result in category_results.items():
                status = result.get("status", "unknown")
                logger.info(f"{test_name}: {status}")
        
        logger.info("\nIntegration test completed. Results saved to integration_test_results.json")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
    finally:
        test_suite.cleanup()


match __name__:
    case "__main__":
    asyncio.run(main()) 