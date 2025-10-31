"""
Blaze AI Advanced Modular Example v7.2.0

This example demonstrates the complete modular system working together,
showcasing all modules: Cache, Monitoring, Optimization, Storage, Execution, and Engines.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List

# Import all modules
from .base import BaseModule, ModuleStatus
from .cache import CacheModule, create_cache_module_with_defaults
from .monitoring import MonitoringModule, create_monitoring_module_with_defaults
from .optimization import OptimizationModule, create_optimization_module_with_defaults
from .storage import StorageModule, create_storage_module_with_defaults
from .execution import ExecutionModule, create_execution_module_with_defaults, TaskPriority
from .engines import EnginesModule, create_engines_module_with_defaults
from .registry import ModuleRegistry, create_module_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# EXAMPLE FUNCTIONS AND TASKS
# ============================================================================

def example_optimization_function(x: float) -> float:
    """Example optimization function: f(x) = x^2 + 2x + 1"""
    return x**2 + 2*x + 1

def example_neural_task(data: Dict[str, Any]) -> Dict[str, Any]:
    """Example neural network task."""
    # Simulate neural network processing
    time.sleep(0.1)
    return {
        "input": data,
        "output": [random.random() for _ in range(10)],
        "confidence": random.random(),
        "processing_time": random.random() * 0.1
    }

def example_storage_task(key: str, data: Any) -> bool:
    """Example storage task."""
    # Simulate storage operation
    time.sleep(0.05)
    return True

def example_computation_task(n: int) -> int:
    """Example computation task."""
    # Simulate heavy computation
    result = 0
    for i in range(n):
        result += i * i
    return result

async def example_async_task(delay: float, task_id: str) -> str:
    """Example asynchronous task."""
    await asyncio.sleep(delay)
    return f"Task {task_id} completed after {delay}s"

# ============================================================================
# ADVANCED INTEGRATION EXAMPLE
# ============================================================================

async def advanced_integration_example():
    """
    Advanced example showing all modules working together.
    
    This demonstrates:
    1. Module initialization and registration
    2. Cross-module communication
    3. Complex task orchestration
    4. Performance monitoring and optimization
    5. Data storage and retrieval
    6. Engine-based optimization
    """
    
    logger.info("🚀 Starting Advanced Blaze AI Modular Integration Example")
    
    try:
        # ====================================================================
        # STEP 1: Create and initialize all modules
        # ====================================================================
        
        logger.info("📦 Creating and initializing all modules...")
        
        # Create modules with default configurations
        cache_module = create_cache_module_with_defaults()
        monitoring_module = create_monitoring_module_with_defaults()
        optimization_module = create_optimization_module_with_defaults()
        storage_module = create_storage_module_with_defaults()
        execution_module = create_execution_module_with_defaults()
        engines_module = create_engines_module_with_defaults()
        
        # Initialize all modules
        modules = [
            cache_module,
            monitoring_module,
            optimization_module,
            storage_module,
            execution_module,
            engines_module
        ]
        
        for module in modules:
            success = await module.initialize()
            if not success:
                logger.error(f"Failed to initialize {module.config.name}")
                return
        
        logger.info("✅ All modules initialized successfully")
        
        # ====================================================================
        # STEP 2: Create module registry and register all modules
        # ====================================================================
        
        logger.info("🔧 Creating module registry...")
        
        registry = create_module_registry()
        
        # Register all modules
        for module in modules:
            registry.register_module(module)
        
        logger.info(f"✅ Registry created with {len(modules)} modules")
        
        # ====================================================================
        # STEP 3: Demonstrate cross-module functionality
        # ====================================================================
        
        logger.info("🔄 Demonstrating cross-module functionality...")
        
        # Cache some data
        await cache_module.set("example_data", {"numbers": [1, 2, 3, 4, 5]}, tags=["example"])
        await cache_module.set("optimization_params", {"x_range": [-10, 10], "iterations": 100}, tags=["optimization"])
        
        # Store data in storage module
        await storage_module.store("large_dataset", {"data": [random.random() for _ in range(1000)]})
        await storage_module.store("configuration", {"cache_size": 1000, "workers": 8, "timeout": 60})
        
        # Submit tasks to execution module
        task_ids = []
        
        # Submit optimization task
        opt_task_id = await execution_module.submit_task(
            example_optimization_function,
            5.0,
            priority=TaskPriority.HIGH,
            tags=["optimization", "mathematical"],
            metadata={"function": "quadratic", "domain": "real_numbers"}
        )
        task_ids.append(opt_task_id)
        
        # Submit neural task
        neural_task_id = await execution_module.submit_task(
            example_neural_task,
            {"input_size": 100, "layers": [64, 32, 16]},
            priority=TaskPriority.NORMAL,
            tags=["neural", "ml"],
            metadata={"model_type": "feedforward", "activation": "relu"}
        )
        task_ids.append(neural_task_id)
        
        # Submit storage task
        storage_task_id = await execution_module.submit_task(
            example_storage_task,
            "processed_result",
            {"result": "success", "timestamp": time.time()},
            priority=TaskPriority.LOW,
            tags=["storage", "io"]
        )
        task_ids.append(storage_task_id)
        
        # Submit computation task
        comp_task_id = await execution_module.submit_task(
            example_computation_task,
            10000,
            priority=TaskPriority.BACKGROUND,
            tags=["computation", "cpu_intensive"]
        )
        task_ids.append(comp_task_id)
        
        # Submit async task
        async_task_id = await execution_module.submit_task(
            example_async_task,
            2.0,
            "async_example",
            priority=TaskPriority.NORMAL,
            tags=["async", "io_bound"]
        )
        task_ids.append(async_task_id)
        
        logger.info(f"✅ Submitted {len(task_ids)} tasks for execution")
        
        # ====================================================================
        # STEP 4: Use engines for optimization
        # ====================================================================
        
        logger.info("⚡ Using engines for optimization...")
        
        # Create optimization problem
        optimization_problem = {
            "type": "optimization",
            "variables": [5.0, -3.0, 1.0],
            "constraints": ["x >= -10", "x <= 10"],
            "objective": "minimize",
            "function": "quadratic"
        }
        
        # Execute with quantum engine
        try:
            quantum_result = await engines_module.execute_with_engine("quantum", optimization_problem)
            logger.info(f"🔮 Quantum optimization result: {quantum_result}")
        except Exception as e:
            logger.warning(f"Quantum engine not available: {e}")
        
        # Execute with neural turbo engine
        try:
            neural_data = {
                "type": "transformer",
                "input": {"sequence": [1, 2, 3, 4, 5], "attention_heads": 8}
            }
            neural_result = await engines_module.execute_with_engine("neural_turbo", neural_data)
            logger.info(f"🧠 Neural turbo result: {neural_result}")
        except Exception as e:
            logger.warning(f"Neural turbo engine not available: {e}")
        
        # Execute with hybrid engine
        try:
            hybrid_result = await engines_module.execute_with_engine("hybrid", {
                "type": "optimization",
                "priority": 1,
                "data": optimization_problem
            })
            logger.info(f"🔄 Hybrid engine result: {hybrid_result}")
        except Exception as e:
            logger.warning(f"Hybrid engine not available: {e}")
        
        # ====================================================================
        # STEP 5: Monitor and collect metrics
        # ====================================================================
        
        logger.info("📊 Collecting metrics and monitoring...")
        
        # Wait for some tasks to complete
        await asyncio.sleep(3)
        
        # Get metrics from all modules
        all_metrics = {}
        for module in modules:
            try:
                metrics = await module.get_metrics()
                all_metrics[module.config.name] = metrics
            except Exception as e:
                logger.warning(f"Failed to get metrics from {module.config.name}: {e}")
        
        # Get health status
        all_health = {}
        for module in modules:
            try:
                health = await module.health_check()
                all_health[module.config.name] = health
            except Exception as e:
                logger.warning(f"Failed to get health from {module.config.name}: {e}")
        
        # Get engine status
        try:
            engine_status = await engines_module.get_engine_status()
            all_metrics["engines_detailed"] = engine_status
        except Exception as e:
            logger.warning(f"Failed to get engine status: {e}")
        
        # ====================================================================
        # STEP 6: Demonstrate advanced features
        # ====================================================================
        
        logger.info("🚀 Demonstrating advanced features...")
        
        # Cache performance test
        cache_start = time.time()
        for i in range(100):
            await cache_module.set(f"perf_test_{i}", f"value_{i}", tags=["performance"])
        cache_time = time.time() - cache_start
        logger.info(f"⏱️ Cache performance: 100 operations in {cache_time:.3f}s")
        
        # Storage performance test
        storage_start = time.time()
        for i in range(50):
            await storage_module.store(f"storage_test_{i}", {"data": f"large_data_{i}" * 100})
        storage_time = time.time() - storage_start
        logger.info(f"💾 Storage performance: 50 operations in {storage_time:.3f}s")
        
        # Execution queue status
        queue_size = execution_module.task_queue.size()
        active_workers = len([w for w in execution_module.workers if w.is_active])
        logger.info(f"⚙️ Execution status: {queue_size} tasks in queue, {active_workers} active workers")
        
        # ====================================================================
        # STEP 7: Wait for remaining tasks and collect results
        # ====================================================================
        
        logger.info("⏳ Waiting for remaining tasks to complete...")
        
        # Wait for all tasks to complete
        for task_id in task_ids:
            try:
                result = await execution_module.wait_for_task(task_id, timeout=30)
                logger.info(f"✅ Task {task_id} completed with result: {result}")
            except Exception as e:
                logger.warning(f"Task {task_id} failed or timed out: {e}")
        
        # ====================================================================
        # STEP 8: Final metrics and summary
        # ====================================================================
        
        logger.info("📈 Final metrics and summary...")
        
        # Get final metrics
        final_metrics = {}
        for module in modules:
            try:
                metrics = await module.get_metrics()
                final_metrics[module.config.name] = metrics
            except Exception as e:
                logger.warning(f"Failed to get final metrics from {module.config.name}: {e}")
        
        # Print summary
        logger.info("🎯 MODULAR SYSTEM SUMMARY:")
        logger.info(f"   • Cache hits: {final_metrics.get('cache', {}).get('cache_metrics', {}).get('hits', 0)}")
        logger.info(f"   • Storage entries: {final_metrics.get('storage', {}).get('storage_metrics', {}).get('total_stored', 0)}")
        logger.info(f"   • Tasks completed: {final_metrics.get('execution', {}).get('execution_metrics', {}).get('completed_tasks', 0)}")
        logger.info(f"   • Engines active: {final_metrics.get('engines', {}).get('engine_metrics', {}).get('engines_active', 0)}")
        
        # ====================================================================
        # STEP 9: Cleanup and shutdown
        # ====================================================================
        
        logger.info("🧹 Cleaning up and shutting down...")
        
        # Shutdown all modules
        for module in modules:
            try:
                await module.shutdown()
                logger.info(f"✅ {module.config.name} shutdown successfully")
            except Exception as e:
                logger.error(f"❌ Error shutting down {module.config.name}: {e}")
        
        logger.info("🎉 Advanced integration example completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Advanced integration example failed: {e}")
        raise

# ============================================================================
# PERFORMANCE BENCHMARK EXAMPLE
# ============================================================================

async def performance_benchmark():
    """Run performance benchmarks on the modular system."""
    
    logger.info("🏃 Starting performance benchmark...")
    
    try:
        # Create minimal modules for benchmarking
        cache_module = create_cache_module_with_defaults()
        storage_module = create_storage_module_with_defaults()
        execution_module = create_execution_module_with_defaults()
        
        # Initialize modules
        await cache_module.initialize()
        await storage_module.initialize()
        await execution_module.initialize()
        
        # Benchmark cache operations
        logger.info("📊 Benchmarking cache operations...")
        cache_start = time.time()
        
        for i in range(1000):
            await cache_module.set(f"bench_{i}", f"value_{i}")
        
        for i in range(1000):
            await cache_module.get(f"bench_{i}")
        
        cache_time = time.time() - cache_start
        cache_ops_per_second = 2000 / cache_time
        
        logger.info(f"   Cache: {cache_ops_per_second:.0f} ops/sec")
        
        # Benchmark storage operations
        logger.info("📊 Benchmarking storage operations...")
        storage_start = time.time()
        
        for i in range(100):
            await storage_module.store(f"bench_{i}", {"data": f"large_data_{i}" * 100})
        
        for i in range(100):
            await storage_module.retrieve(f"bench_{i}")
        
        storage_time = time.time() - storage_start
        storage_ops_per_second = 200 / storage_time
        
        logger.info(f"   Storage: {storage_ops_per_second:.0f} ops/sec")
        
        # Benchmark execution module
        logger.info("📊 Benchmarking execution module...")
        execution_start = time.time()
        
        task_ids = []
        for i in range(100):
            task_id = await execution_module.submit_task(
                example_computation_task,
                1000,
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)
        
        # Wait for completion
        for task_id in task_ids:
            await execution_module.wait_for_task(task_id, timeout=60)
        
        execution_time = time.time() - execution_start
        execution_ops_per_second = 100 / execution_time
        
        logger.info(f"   Execution: {execution_ops_per_second:.0f} ops/sec")
        
        # Summary
        logger.info("🏆 PERFORMANCE BENCHMARK SUMMARY:")
        logger.info(f"   • Cache: {cache_ops_per_second:.0f} ops/sec")
        logger.info(f"   • Storage: {storage_ops_per_second:.0f} ops/sec")
        logger.info(f"   • Execution: {execution_ops_per_second:.0f} ops/sec")
        
        # Cleanup
        await cache_module.shutdown()
        await storage_module.shutdown()
        await execution_module.shutdown()
        
        logger.info("✅ Performance benchmark completed!")
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        raise

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main function to run the advanced examples."""
    
    logger.info("🚀 Blaze AI Advanced Modular Example v7.2.0")
    logger.info("=" * 60)
    
    try:
        # Run advanced integration example
        await advanced_integration_example()
        
        logger.info("=" * 60)
        
        # Run performance benchmark
        await performance_benchmark()
        
        logger.info("=" * 60)
        logger.info("🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example execution failed: {e}")
        raise

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
