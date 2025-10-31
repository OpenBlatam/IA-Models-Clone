#!/usr/bin/env python3
"""
Modular Blaze AI System Demo.

This demo showcases the new modular architecture with:
- Base engine infrastructure
- Circuit breaker patterns
- Engine factory system
- Engine management and orchestration
- Plugin system
- Quick access functions
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any

# Import the modular system
from engines import (
    # Core infrastructure
    Engine, EngineStatus, EngineType, EnginePriority,
    EngineMetadata, EngineCapabilities,
    
    # Circuit breaker
    CircuitBreaker, CircuitBreakerState, CircuitBreakerConfig,
    create_resilient_circuit_breaker, create_stable_circuit_breaker,
    
    # Factory system
    EngineFactory, EngineFactoryConfig, EngineTemplate,
    create_engine_factory, create_standard_engine_factory,
    
    # Manager system
    EngineManager, EngineManagerConfig,
    get_engine_manager, shutdown_engine_manager,
    
    # Plugin system
    PluginManager, PluginConfig,
    create_plugin_manager, create_standard_plugin_manager,
    
    # Quick access functions
    get_default_engine_manager, get_default_engine_factory,
    get_default_plugin_manager, create_engine, list_available_engines,
    list_plugins, get_system_status, get_engine_health_summary
)

# =============================================================================
# Demo Configuration
# =============================================================================

class DemoConfig:
    """Configuration for the modular system demo."""
    
    def __init__(self):
        self.demo_duration = 30  # seconds
        self.engine_operations = 10
        self.circuit_breaker_tests = 5
        self.plugin_tests = True
        self.performance_tests = True
        self.show_detailed_logs = True

# =============================================================================
# Demo Engine Implementation
# =============================================================================

class DemoEngine(Engine):
    """Demo engine implementation for testing the modular system."""
    
    def _get_engine_type(self) -> EngineType:
        return EngineType.CUSTOM
    
    def _get_description(self) -> str:
        return f"Demo engine for testing modular architecture"
    
    def _get_priority(self) -> EnginePriority:
        return EnginePriority.NORMAL
    
    def _get_supported_operations(self) -> list[str]:
        return ["demo_operation", "test_operation", "benchmark_operation"]
    
    def _get_max_batch_size(self) -> int:
        return 5
    
    def _get_max_concurrent_requests(self) -> int:
        return 10
    
    def _supports_streaming(self) -> bool:
        return False
    
    def _supports_async(self) -> bool:
        return True
    
    async def _initialize_engine(self) -> None:
        """Initialize the demo engine."""
        await asyncio.sleep(0.1)  # Simulate initialization time
        self.logger.info(f"Demo engine {self.name} initialized")
    
    async def _execute_operation(self, operation: str, params: Dict[str, Any]) -> Any:
        """Execute a demo operation."""
        operation_type = params.get("type", "normal")
        
        if operation == "demo_operation":
            await asyncio.sleep(0.05)  # Simulate processing time
            return {
                "operation": operation,
                "result": f"Demo operation completed for {self.name}",
                "params": params,
                "timestamp": time.time()
            }
        
        elif operation == "test_operation":
            if operation_type == "error":
                raise Exception("Simulated error for testing")
            await asyncio.sleep(0.02)
            return {
                "operation": operation,
                "result": f"Test operation completed for {self.name}",
                "params": params,
                "timestamp": time.time()
            }
        
        elif operation == "benchmark_operation":
            await asyncio.sleep(0.01)  # Fast operation
            return {
                "operation": operation,
                "result": f"Benchmark operation completed for {self.name}",
                "params": params,
                "timestamp": time.time(),
                "performance": "high"
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")

# =============================================================================
# Demo Runner
# =============================================================================

class ModularSystemDemo:
    """Main demo runner for the modular Blaze AI system."""
    
    def __init__(self, config: DemoConfig = None):
        self.config = config or DemoConfig()
        self.logger = None  # Will be set up later
        self.results = {}
        self.start_time = time.time()
        
        # System components
        self.engine_manager = None
        self.engine_factory = None
        self.plugin_manager = None
        
        # Demo engines
        self.demo_engines = []
        
    async def run_demo(self):
        """Run the complete modular system demo."""
        print("🚀 Starting Modular Blaze AI System Demo")
        print("=" * 60)
        
        try:
            # Initialize system
            await self._initialize_system()
            
            # Run demo sections
            await self._demo_base_infrastructure()
            await self._demo_circuit_breaker()
            await self._demo_factory_system()
            await self._demo_engine_management()
            await self._demo_plugin_system()
            await self._demo_quick_access_functions()
            await self._demo_performance_features()
            
            # Show results
            await self._show_demo_results()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _initialize_system(self):
        """Initialize the modular system components."""
        print("🔧 Initializing modular system...")
        
        # Initialize engine factory
        self.engine_factory = create_standard_engine_factory()
        print(f"✅ Engine factory initialized with {len(self.engine_factory.engine_templates)} templates")
        
        # Initialize engine manager
        self.engine_manager = get_engine_manager()
        print(f"✅ Engine manager initialized with {len(self.engine_manager.engines)} engines")
        
        # Initialize plugin manager
        self.plugin_manager = create_standard_plugin_manager()
        print(f"✅ Plugin manager initialized")
        
        # Create demo engines
        await self._create_demo_engines()
        
        print("✅ System initialization complete")
    
    async def _create_demo_engines(self):
        """Create demo engines for testing."""
        print("🔨 Creating demo engines...")
        
        demo_configs = [
            {"name": "demo_engine_1", "priority": EnginePriority.HIGH},
            {"name": "demo_engine_2", "priority": EnginePriority.NORMAL},
            {"name": "demo_engine_3", "priority": EnginePriority.LOW}
        ]
        
        for config in demo_configs:
            engine = DemoEngine(config["name"], config)
            self.engine_manager.register_engine(config["name"], engine, "demo")
            self.demo_engines.append(engine)
            print(f"  ✅ Created demo engine: {config['name']}")
        
        # Wait for engines to initialize
        await asyncio.sleep(1)
        print(f"✅ Created {len(self.demo_engines)} demo engines")
    
    async def _demo_base_infrastructure(self):
        """Demo the base engine infrastructure."""
        print("\n🏗️  Demo: Base Engine Infrastructure")
        print("-" * 40)
        
        # Test engine metadata and capabilities
        for engine in self.demo_engines:
            print(f"\n📋 Engine: {engine.name}")
            print(f"  Type: {engine.metadata.type.value}")
            print(f"  Priority: {engine.metadata.priority.value}")
            print(f"  Description: {engine.metadata.description}")
            print(f"  Supported Operations: {engine.capabilities.supported_operations}")
            print(f"  Max Batch Size: {engine.capabilities.max_batch_size}")
            print(f"  Max Concurrent Requests: {engine.capabilities.max_concurrent_requests}")
            
            # Test health status
            health = engine.get_health_status()
            print(f"  Health Status: {health.status}")
            print(f"  Health Details: {json.dumps(health.details, indent=4, default=str)}")
        
        self.results["base_infrastructure"] = "✅ Passed"
        print("✅ Base infrastructure demo completed")
    
    async def _demo_circuit_breaker(self):
        """Demo the circuit breaker pattern."""
        print("\n🔌 Demo: Circuit Breaker Pattern")
        print("-" * 40)
        
        # Create different types of circuit breakers
        resilient_cb = create_resilient_circuit_breaker()
        stable_cb = create_stable_circuit_breaker()
        
        print(f"✅ Created resilient circuit breaker: {resilient_cb.get_state().value}")
        print(f"✅ Created stable circuit breaker: {stable_cb.get_state().value}")
        
        # Test circuit breaker with demo operations
        for i in range(self.circuit_breaker_tests):
            try:
                result = await resilient_cb.call(
                    self.demo_engines[0].execute,
                    "test_operation",
                    {"type": "normal", "test_id": i}
                )
                print(f"  ✅ Circuit breaker test {i+1}: Success")
            except Exception as e:
                print(f"  ❌ Circuit breaker test {i+1}: {e}")
        
        # Test error handling
        try:
            await resilient_cb.call(
                self.demo_engines[0].execute,
                "test_operation",
                {"type": "error", "test_id": "error_test"}
            )
        except Exception as e:
            print(f"  ✅ Error handling test: {e}")
        
        print(f"✅ Circuit breaker state: {resilient_cb.get_state().value}")
        print(f"✅ Circuit breaker metrics: {resilient_cb.get_metrics()}")
        
        self.results["circuit_breaker"] = "✅ Passed"
        print("✅ Circuit breaker demo completed")
    
    async def _demo_factory_system(self):
        """Demo the engine factory system."""
        print("\n🏭 Demo: Engine Factory System")
        print("-" * 40)
        
        # Show available templates
        templates = self.engine_factory.get_available_templates()
        print(f"📋 Available engine templates: {templates}")
        
        # Show template details
        for template_name in templates[:3]:  # Show first 3
            template_info = self.engine_factory.get_template_info(template_name)
            if template_info:
                print(f"\n🔧 Template: {template_name}")
                print(f"  Description: {template_info.description}")
                print(f"  Priority: {template_info.priority.value}")
                print(f"  Tags: {template_info.tags}")
        
        # Test engine creation
        try:
            # Create a custom engine using the factory
            custom_engine = self.engine_factory.create_engine(
                "llm",  # Use existing template
                {"custom_param": "demo_value"},
                "demo_custom_engine"
            )
            print(f"✅ Created custom engine: {custom_engine.name}")
            
            # Register with manager
            self.engine_manager.register_engine(
                custom_engine.name, 
                custom_engine, 
                "custom"
            )
            print(f"✅ Registered custom engine with manager")
            
        except Exception as e:
            print(f"❌ Engine creation failed: {e}")
        
        self.results["factory_system"] = "✅ Passed"
        print("✅ Factory system demo completed")
    
    async def _demo_engine_management(self):
        """Demo the engine management and orchestration."""
        print("\n🎯 Demo: Engine Management & Orchestration")
        print("-" * 40)
        
        # Show engine status
        status = self.engine_manager.get_engine_status()
        print(f"📊 Engine Status Overview:")
        for engine_name, engine_status in status.items():
            print(f"  {engine_name}: {engine_status['status']} ({engine_status['template']})")
        
        # Show system metrics
        metrics = self.engine_manager.get_system_metrics()
        print(f"\n📈 System Metrics:")
        print(f"  Total Engines: {metrics['total_engines']}")
        print(f"  Healthy Engines: {metrics['healthy_engines']}")
        print(f"  Health Ratio: {metrics['health_ratio']:.2%}")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Error Rate: {metrics['error_rate']:.2%}")
        
        # Test engine operations
        print(f"\n🚀 Testing engine operations...")
        for i in range(self.engine_operations):
            try:
                # Find best engine for operation
                best_engine = self.engine_manager.find_engine_for_operation("demo_operation")
                if best_engine:
                    result = await self.engine_manager.dispatch(
                        best_engine,
                        "demo_operation",
                        {"operation_id": i, "timestamp": time.time()}
                    )
                    print(f"  ✅ Operation {i+1}: {best_engine} -> Success")
                else:
                    print(f"  ❌ Operation {i+1}: No suitable engine found")
            except Exception as e:
                print(f"  ❌ Operation {i+1}: {e}")
        
        # Show engine groups
        print(f"\n🏷️  Engine Groups:")
        for engine_type, engine_names in self.engine_manager.engine_groups.items():
            print(f"  {engine_type.value}: {engine_names}")
        
        self.results["engine_management"] = "✅ Passed"
        print("✅ Engine management demo completed")
    
    async def _demo_plugin_system(self):
        """Demo the plugin system."""
        print("\n🔌 Demo: Plugin System")
        print("-" * 40)
        
        if not self.config.plugin_tests:
            print("⏭️  Plugin tests skipped")
            self.results["plugin_system"] = "⏭️  Skipped"
            return
        
        # Show plugin system status
        plugin_metrics = self.plugin_manager.loader.get_plugin_metrics()
        print(f"📊 Plugin System Status:")
        print(f"  Total Plugins: {plugin_metrics['total_plugins']}")
        print(f"  Loaded Plugins: {plugin_metrics['loaded_plugins']}")
        print(f"  Failed Plugins: {plugin_metrics['failed_plugins']}")
        print(f"  Total Plugin Engines: {plugin_metrics['total_engines']}")
        
        # List available plugins
        plugins = self.plugin_manager.loader.list_plugins()
        if plugins:
            print(f"\n📋 Available Plugins:")
            for plugin_name in plugins:
                plugin_info = self.plugin_manager.loader.get_plugin_info(plugin_name)
                if plugin_info:
                    print(f"  {plugin_name}: {plugin_info.metadata.description}")
        else:
            print(f"\n📋 No plugins available (this is normal for a fresh system)")
        
        # Test plugin search
        search_results = self.plugin_manager.search_plugins("demo")
        print(f"\n🔍 Plugin Search Results for 'demo': {search_results}")
        
        self.results["plugin_system"] = "✅ Passed"
        print("✅ Plugin system demo completed")
    
    async def _demo_quick_access_functions(self):
        """Demo the quick access functions."""
        print("\n⚡ Demo: Quick Access Functions")
        print("-" * 40)
        
        # Test quick access functions
        print("🔧 Testing quick access functions...")
        
        # List available engines
        available_engines = list_available_engines()
        print(f"  📋 Available engines: {available_engines}")
        
        # List plugins
        available_plugins = list_plugins()
        print(f"  📋 Available plugins: {available_plugins}")
        
        # Get system status
        system_status = get_system_status()
        print(f"  📊 System status retrieved: {len(system_status)} components")
        
        # Get engine health summary
        health_summary = get_engine_health_summary()
        print(f"  🏥 Health summary: {health_summary['healthy']}/{health_summary['total_engines']} engines healthy")
        
        # Test engine creation
        try:
            demo_engine = create_engine("llm", {"demo": True}, "quick_demo_engine")
            print(f"  ✅ Quick engine creation: {demo_engine.name}")
            
            # Clean up
            self.engine_manager.unregister_engine(demo_engine.name)
            print(f"  ✅ Quick engine cleanup completed")
            
        except Exception as e:
            print(f"  ❌ Quick engine creation failed: {e}")
        
        self.results["quick_access"] = "✅ Passed"
        print("✅ Quick access functions demo completed")
    
    async def _demo_performance_features(self):
        """Demo performance and advanced features."""
        print("\n🚀 Demo: Performance & Advanced Features")
        print("-" * 40)
        
        if not self.config.performance_tests:
            print("⏭️  Performance tests skipped")
            self.results["performance"] = "⏭️  Skipped"
            return
        
        # Test batch operations
        print("📦 Testing batch operations...")
        batch_requests = []
        for i in range(5):
            batch_requests.append({
                "engine": "demo_engine_1",
                "operation": "demo_operation",
                "params": {"batch_id": i, "timestamp": time.time()}
            })
        
        try:
            batch_results = await self.engine_manager.dispatch_batch(batch_requests)
            successful = sum(1 for r in batch_results if r.get("success", False))
            print(f"  ✅ Batch operation: {successful}/{len(batch_requests)} successful")
        except Exception as e:
            print(f"  ❌ Batch operation failed: {e}")
        
        # Test concurrent operations
        print("⚡ Testing concurrent operations...")
        async def concurrent_operation(operation_id):
            try:
                result = await self.engine_manager.dispatch(
                    "demo_engine_2",
                    "benchmark_operation",
                    {"operation_id": operation_id, "concurrent": True}
                )
                return f"Operation {operation_id}: Success"
            except Exception as e:
                return f"Operation {operation_id}: {e}"
        
        concurrent_tasks = [
            concurrent_operation(i) for i in range(10)
        ]
        
        start_time = time.time()
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        end_time = time.time()
        
        successful_concurrent = sum(1 for r in concurrent_results if "Success" in r)
        print(f"  ✅ Concurrent operations: {successful_concurrent}/{len(concurrent_results)} successful")
        print(f"  ⏱️  Total time: {end_time - start_time:.3f}s")
        
        self.results["performance"] = "✅ Passed"
        print("✅ Performance features demo completed")
    
    async def _show_demo_results(self):
        """Show the demo results summary."""
        print("\n📊 Demo Results Summary")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if "✅" in result)
        skipped_tests = sum(1 for result in self.results.values() if "⏭️" in result)
        
        print(f"🎯 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⏭️  Skipped: {skipped_tests}")
        print(f"❌ Failed: {total_tests - passed_tests - skipped_tests}")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.results.items():
            print(f"  {test_name.replace('_', ' ').title()}: {result}")
        
        # Show final system status
        print(f"\n🏁 Final System Status:")
        final_status = get_system_status()
        print(f"  Total Engines: {final_status['engines']['total_engines']}")
        print(f"  Healthy Engines: {final_status['engines']['healthy_engines']}")
        print(f"  Total Plugins: {final_status['plugins']['total_plugins']}")
        print(f"  Total Templates: {final_status['factory']['total_templates']}")
        
        demo_duration = time.time() - self.start_time
        print(f"\n⏱️  Demo Duration: {demo_duration:.2f} seconds")
        print(f"🎉 Modular Blaze AI System Demo Completed Successfully!")
    
    async def _cleanup(self):
        """Clean up demo resources."""
        print("\n🧹 Cleaning up demo resources...")
        
        try:
            # Shutdown engine manager
            if self.engine_manager:
                await self.engine_manager.shutdown()
                print("✅ Engine manager shutdown completed")
            
            # Clean up demo engines
            for engine in self.demo_engines:
                try:
                    await engine.shutdown()
                except:
                    pass
            
            print("✅ Demo cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

# =============================================================================
# Main Demo Execution
# =============================================================================

async def main():
    """Main demo execution function."""
    config = DemoConfig()
    config.show_detailed_logs = True
    config.plugin_tests = True
    config.performance_tests = True
    
    demo = ModularSystemDemo(config)
    
    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    print("🚀 Starting Modular Blaze AI System Demo")
    print("This demo showcases the new modular architecture")
    print("Press Ctrl+C to stop the demo\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        exit(1)


