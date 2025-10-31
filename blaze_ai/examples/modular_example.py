"""
Modular Blaze AI System Example

This example demonstrates how to use the modular Blaze AI system
with different modules working together.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import the modular system
from ..modules import (
    ModuleRegistry,
    CacheModule,
    MonitoringModule,
    OptimizationModule,
    StorageModule,
    ExecutionModule,
    EnginesModule,
    MLModule,
    DataAnalysisModule,
    AIIntelligenceModule,
    create_cache_module,
    create_monitoring_module,
    create_optimization_module,
    create_storage_module,
    create_execution_module,
    create_engines_module,
    create_ml_module,
    create_data_analysis_module,
    create_ai_intelligence_module,
    create_module_registry
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# EXAMPLE FUNCTIONS
# ============================================================================

def objective_function(params: Dict[str, float]) -> float:
    """Example objective function for optimization."""
    x = params.get('x', 0.0)
    y = params.get('y', 0.0)
    
    # Simple quadratic function: f(x,y) = x^2 + y^2
    return x**2 + y**2

def constraint_function(params: Dict[str, float]) -> float:
    """Example constraint function."""
    x = params.get('x', 0.0)
    y = params.get('y', 0.0)
    
    # Constraint: x + y <= 2
    return 2 - (x + y)

async def custom_metric_collector() -> float:
    """Custom metric collector for monitoring."""
    return time.time() % 100  # Simple time-based metric

async def alert_handler(alert):
    """Handle alerts from monitoring module."""
    logger.info(f"Alert received: {alert.level.name} - {alert.message}")

# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    """Main example function."""
    logger.info("🚀 Starting Modular Blaze AI System Example")
    
    # Create module registry
    registry = create_module_registry()
    
    try:
        # Initialize registry
        await registry.initialize()
        
        # ========================================================================
        # CREATE AND REGISTER MODULES
        # ========================================================================
        
        logger.info("📦 Creating and registering modules...")
        
        # 1. Create monitoring module
        monitoring = create_monitoring_module(
            name="system_monitoring",
            collection_interval=5.0,  # Collect metrics every 5 seconds
            priority=1  # High priority
        )
        
        # 2. Create cache module
        cache = create_cache_module(
            name="main_cache",
            max_size=1000,
            strategy="HYBRID",
            compression="LZ4",
            priority=2
        )
        
        # 3. Create optimization module
        optimization = create_optimization_module(
            name="main_optimizer",
            optimization_type="GENETIC",
            priority=3
        )
        
        # 4. Create AI Intelligence module
        ai_intelligence = create_ai_intelligence_module(
            name="ai_intelligence",
            enable_nlp=True,
            enable_vision=True,
            enable_reasoning=True,
            enable_multimodal=True,
            priority=4
        )
        
        # Register modules
        await registry.register_module(monitoring)
        await registry.register_module(cache)
        await registry.register_module(optimization)
        await registry.register_module(ai_intelligence)
        
        logger.info("✅ All modules registered successfully")
        
        # ========================================================================
        # CONFIGURE MODULES
        # ========================================================================
        
        logger.info("⚙️ Configuring modules...")
        
        # Add custom metric collector to monitoring
        monitoring.register_custom_collector("time_metric", custom_metric_collector)
        
        # Add alert handler
        monitoring.add_alert_handler(alert_handler)
        
        # Wait for modules to be ready
        await asyncio.sleep(2)
        
        # ========================================================================
        # DEMONSTRATE CACHE FUNCTIONALITY
        # ========================================================================
        
        logger.info("💾 Demonstrating cache functionality...")
        
        # Store some data in cache
        await cache.set("user:123", {"name": "John Doe", "age": 30}, ttl=3600)
        await cache.set("config:app", {"version": "1.0.0", "debug": False}, ttl=7200)
        
        # Retrieve data
        user_data = await cache.get("user:123")
        config_data = await cache.get("config:app")
        
        logger.info(f"Retrieved user data: {user_data}")
        logger.info(f"Retrieved config data: {config_data}")
        
        # Get cache statistics
        cache_stats = cache.get_cache_stats()
        logger.info(f"Cache hit rate: {cache_stats.hit_rate:.2%}")
        
        # ========================================================================
        # DEMONSTRATE MONITORING FUNCTIONALITY
        # ========================================================================
        
        logger.info("📊 Demonstrating monitoring functionality...")
        
        # Collect metrics manually
        metrics = await monitoring.collect_metrics_now()
        logger.info(f"Collected {sum(len(m) for m in metrics.values())} metrics")
        
        # Get metric summary
        summary = monitoring.get_metric_summary()
        logger.info(f"Metric summary: {len(summary)} sources")
        
        # Check for alerts
        alerts = monitoring.get_alerts()
        logger.info(f"Active alerts: {len(alerts)}")
        
        # ========================================================================
        # DEMONSTRATE OPTIMIZATION FUNCTIONALITY
        # ========================================================================
        
        logger.info("🔬 Demonstrating optimization functionality...")
        
        # Submit optimization task
        task_id = await optimization.submit_task(
            name="minimize_quadratic",
            objective_function=objective_function,
            constraints=[constraint_function],
            bounds={
                'x': (-5.0, 5.0),
                'y': (-5.0, 5.0)
            }
        )
        
        logger.info(f"Submitted optimization task: {task_id}")
        
        # Wait for task to complete
        max_wait = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            task_status = await optimization.get_task_status(task_id)
            
            if task_status and task_status.get('status') == 'completed':
                logger.info(f"Optimization task completed: {task_status}")
                break
            
            await asyncio.sleep(1)
        else:
            logger.warning("Optimization task did not complete within time limit")
        
        # Get available algorithms
        algorithms = optimization.get_available_algorithms()
        logger.info(f"Available algorithms: {len(algorithms)}")
        
        # ========================================================================
        # DEMONSTRATE AI INTELLIGENCE FUNCTIONALITY
        # ========================================================================
        
        logger.info("🧠 Demonstrating AI Intelligence functionality...")
        
        # Test NLP processing
        nlp_result = await ai_intelligence.process_nlp_task(
            "This is an amazing product that I absolutely love!",
            task="sentiment"
        )
        logger.info(f"NLP sentiment analysis: {nlp_result}")
        
        # Test vision processing (simulated image data)
        simulated_image = b"simulated_image_data_for_testing"
        vision_result = await ai_intelligence.process_vision_task(
            simulated_image,
            task="object_detection"
        )
        logger.info(f"Vision object detection: {vision_result}")
        
        # Test reasoning
        reasoning_result = await ai_intelligence.process_reasoning_task(
            "If all humans are mortal and Socrates is human, what can we conclude?",
            reasoning_type="logical"
        )
        logger.info(f"Logical reasoning: {reasoning_result}")
        
        # Test multimodal processing
        multimodal_result = await ai_intelligence.process_multimodal_task(
            "A beautiful landscape with mountains and trees",
            simulated_image,
            task="analysis"
        )
        logger.info(f"Multimodal analysis: {vision_result}")
        
        # Get AI Intelligence metrics
        ai_metrics = await ai_intelligence.get_metrics()
        logger.info(f"AI Intelligence metrics: {ai_metrics}")
        
        # ========================================================================
        # DEMONSTRATE REGISTRY FUNCTIONALITY
        # ========================================================================
        
        logger.info("📋 Demonstrating registry functionality...")
        
        # Get registry status
        registry_status = registry.get_registry_status()
        logger.info(f"Registry stats: {registry_status['stats']}")
        
        # Get dependency tree for a module
        dependency_tree = registry.get_dependency_tree("main_optimizer")
        logger.info(f"Dependency tree for optimizer: {dependency_tree}")
        
        # List all modules
        all_modules = registry.list_modules()
        logger.info(f"All registered modules: {all_modules}")
        
        # ========================================================================
        # PERFORMANCE TESTING
        # ========================================================================
        
        logger.info("⚡ Running performance tests...")
        
        # Cache performance test
        start_time = time.perf_counter()
        for i in range(100):
            await cache.set(f"test_key_{i}", f"test_value_{i}")
            await cache.get(f"test_key_{i}")
        
        cache_time = time.perf_counter() - start_time
        logger.info(f"Cache performance: 100 operations in {cache_time:.3f}s")
        
        # Monitoring performance test
        start_time = time.perf_counter()
        for i in range(10):
            await monitoring.collect_metrics_now()
        
        monitoring_time = time.perf_counter() - start_time
        logger.info(f"Monitoring performance: 10 collections in {monitoring_time:.3f}s")
        
        # ========================================================================
        # SYSTEM STATUS
        # ========================================================================
        
        logger.info("📈 Final system status...")
        
        # Get final status of all modules
        for module_name in registry.list_modules():
            module = registry.get_module(module_name)
            if module:
                status = module.get_status()
                logger.info(f"Module {module_name}: {status['status']}")
        
        # Get final registry stats
        final_stats = registry.get_registry_stats()
        logger.info(f"Final registry stats: {final_stats.to_dict()}")
        
        logger.info("🎉 Example completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise
    
    finally:
        # Shutdown registry
        logger.info("🔄 Shutting down system...")
        await registry.shutdown()
        logger.info("✅ System shutdown completed")

# ============================================================================
# ADVANCED EXAMPLE WITH CUSTOM MODULE
# ============================================================================

class CustomModule(MonitoringModule):
    """Custom module that extends monitoring functionality."""
    
    def __init__(self, config):
        super().__init__(config)
        self.custom_data = {}
    
    async def _initialize_impl(self) -> bool:
        """Custom initialization."""
        await super()._initialize_impl()
        self.custom_data["initialized"] = True
        return True
    
    async def custom_operation(self, data: str) -> str:
        """Custom operation specific to this module."""
        result = f"Processed: {data.upper()}"
        self.custom_data[data] = result
        return result
    
    def get_custom_data(self) -> Dict[str, Any]:
        """Get custom data."""
        return self.custom_data.copy()

async def advanced_example():
    """Advanced example with custom module."""
    logger.info("🚀 Starting Advanced Modular Example")
    
    registry = create_module_registry()
    
    try:
        await registry.initialize()
        
        # Create custom module
        custom_config = MonitoringConfig(
            name="custom_module",
            collection_interval=10.0,
            priority=1
        )
        custom_module = CustomModule(custom_config)
        
        # Register custom module
        await registry.register_module(custom_module)
        
        # Use custom functionality
        result = await custom_module.custom_operation("hello world")
        logger.info(f"Custom operation result: {result}")
        
        custom_data = custom_module.get_custom_data()
        logger.info(f"Custom module data: {custom_data}")
        
        # Get custom module status
        status = custom_module.get_status()
        logger.info(f"Custom module status: {status}")
        
    except Exception as e:
        logger.error(f"Advanced example failed: {e}")
        raise
    
    finally:
        await registry.shutdown()

# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Run basic example
    asyncio.run(main())
    
    # Run advanced example
    # asyncio.run(advanced_example())
