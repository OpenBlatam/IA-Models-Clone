from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
    from plugins import (
    from plugins.examples import WebExtractorPlugin
                import psutil
                import os
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Plugin System Test Suite

This script provides comprehensive testing of the plugin system including:
- Plugin loading and validation
- Lifecycle management
- Error handling and recovery
- Performance testing
- Integration testing

Usage:
    python test_system.py [options]
    
Options:
    --unit          Run unit tests
    --integration   Run integration tests
    --performance   Run performance tests
    --all           Run all tests
    --verbose       Verbose output
"""


# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
        PluginManager, 
        ManagerConfig, 
        ValidationLevel,
        BasePlugin,
        PluginMetadata
    )
except ImportError as e:
    print(f"❌ Failed to import plugin system: {e}")
    sys.exit(1)


class PluginSystemTester:
    """Comprehensive test suite for the plugin system."""
    
    def __init__(self, verbose: bool = False):
        
    """__init__ function."""
self.verbose = verbose
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0
        }
        
        # Configure logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run_tests(self, test_types: List[str]):
        """Run the specified test types."""
        print("🧪 Plugin System Test Suite")
        print("=" * 40)
        
        if "all" in test_types or "unit" in test_types:
            self.run_unit_tests()
        
        if "all" in test_types or "integration" in test_types:
            self.run_integration_tests()
        
        if "all" in test_types or "performance" in test_types:
            self.run_performance_tests()
        
        self.print_summary()
    
    def run_unit_tests(self) -> Any:
        """Run unit tests for individual components."""
        print("\n🔬 Unit Tests")
        print("-" * 15)
        
        # Test plugin metadata
        self.test_plugin_metadata()
        
        # Test configuration validation
        self.test_configuration_validation()
        
        # Test plugin loading
        self.test_plugin_loading()
        
        # Test plugin lifecycle
        self.test_plugin_lifecycle()
        
        # Test error handling
        self.test_error_handling()
    
    def test_plugin_metadata(self) -> Any:
        """Test plugin metadata functionality."""
        self.log_test("Plugin Metadata")
        
        try:
            # Create a test plugin
            plugin = WebExtractorPlugin()
            metadata = plugin.get_metadata()
            
            # Validate metadata
            assert metadata.name == "web_extractor"
            assert metadata.version == "1.0.0"
            assert metadata.description is not None
            assert metadata.author is not None
            assert metadata.category == "extractor"
            
            self.log_success("Plugin metadata is valid")
            
        except Exception as e:
            self.log_failure(f"Plugin metadata test failed: {e}")
    
    def test_configuration_validation(self) -> Any:
        """Test configuration validation."""
        self.log_test("Configuration Validation")
        
        try:
            plugin = WebExtractorPlugin()
            
            # Test valid configuration
            valid_config = {
                "timeout": 30,
                "max_retries": 3,
                "extraction_methods": ["newspaper3k", "trafilatura"]
            }
            assert plugin.validate_config(valid_config)
            
            # Test invalid configuration
            invalid_config = {
                "timeout": -1,  # Invalid timeout
                "max_retries": 999  # Invalid retries
            }
            assert not plugin.validate_config(invalid_config)
            
            self.log_success("Configuration validation works correctly")
            
        except Exception as e:
            self.log_failure(f"Configuration validation test failed: {e}")
    
    def test_plugin_loading(self) -> Any:
        """Test plugin loading functionality."""
        self.log_test("Plugin Loading")
        
        async def _test_loading():
            
    """_test_loading function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Load plugin
                plugin = await manager.load_plugin("web_extractor", {
                    "timeout": 30,
                    "max_retries": 3
                })
                
                assert plugin is not None
                assert plugin.name == "web_extractor"
                
                # Get plugin info
                info = manager.get_plugin_info("web_extractor")
                assert info is not None
                assert info.name == "web_extractor"
                
                await manager.shutdown()
                self.log_success("Plugin loading works correctly")
                
            except Exception as e:
                self.log_failure(f"Plugin loading test failed: {e}")
        
        asyncio.run(_test_loading())
    
    def test_plugin_lifecycle(self) -> Any:
        """Test plugin lifecycle management."""
        self.log_test("Plugin Lifecycle")
        
        async def _test_lifecycle():
            
    """_test_lifecycle function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Load plugin
                plugin = await manager.load_plugin("web_extractor")
                
                # Initialize plugin
                await manager.initialize_plugin("web_extractor")
                state = manager.get_plugin_state("web_extractor")
                assert state.value == "initialized"
                
                # Start plugin
                await manager.start_plugin("web_extractor")
                state = manager.get_plugin_state("web_extractor")
                assert state.value == "running"
                
                # Stop plugin
                await manager.stop_plugin("web_extractor")
                state = manager.get_plugin_state("web_extractor")
                assert state.value == "stopped"
                
                # Unload plugin
                success = await manager.unload_plugin("web_extractor")
                assert success
                
                await manager.shutdown()
                self.log_success("Plugin lifecycle management works correctly")
                
            except Exception as e:
                self.log_failure(f"Plugin lifecycle test failed: {e}")
        
        asyncio.run(_test_lifecycle())
    
    def test_error_handling(self) -> Any:
        """Test error handling and recovery."""
        self.log_test("Error Handling")
        
        async def _test_error_handling():
            
    """_test_error_handling function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Test loading non-existent plugin
                try:
                    await manager.load_plugin("non_existent_plugin")
                    self.log_failure("Should have failed to load non-existent plugin")
                except Exception:
                    # Expected error
                    pass
                
                # Test loading plugin with invalid config
                try:
                    await manager.load_plugin("web_extractor", {
                        "timeout": -1,
                        "max_retries": 999
                    })
                    self.log_failure("Should have failed to load plugin with invalid config")
                except Exception:
                    # Expected error
                    pass
                
                # Test plugin recovery
                plugin = await manager.load_plugin("web_extractor")
                success = plugin.update_config({
                    "timeout": 30,
                    "max_retries": 3
                })
                assert success
                
                await manager.shutdown()
                self.log_success("Error handling works correctly")
                
            except Exception as e:
                self.log_failure(f"Error handling test failed: {e}")
        
        asyncio.run(_test_error_handling())
    
    def run_integration_tests(self) -> Any:
        """Run integration tests."""
        print("\n🔗 Integration Tests")
        print("-" * 20)
        
        # Test plugin manager with multiple plugins
        self.test_multiple_plugins()
        
        # Test event handling
        self.test_event_handling()
        
        # Test configuration management
        self.test_configuration_management()
        
        # Test health monitoring
        self.test_health_monitoring()
    
    def test_multiple_plugins(self) -> Any:
        """Test managing multiple plugins."""
        self.log_test("Multiple Plugins")
        
        async def _test_multiple_plugins():
            
    """_test_multiple_plugins function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Load multiple plugins
                plugins = []
                for i in range(3):
                    plugin_name = f"test_plugin_{i}"
                    plugin = await manager.load_plugin("web_extractor", {
                        "timeout": 30 + i,
                        "max_retries": 3
                    })
                    plugins.append(plugin_name)
                
                # Initialize all plugins
                initialized = await manager.initialize_all_plugins()
                assert len(initialized) == 3
                
                # Start all plugins
                started = await manager.start_all_plugins()
                assert len(started) == 3
                
                # Get plugin list
                all_plugins = manager.list_plugins()
                assert len(all_plugins) == 3
                
                # Stop all plugins
                stopped = await manager.stop_all_plugins()
                assert len(stopped) == 3
                
                # Unload all plugins
                unloaded = await manager.unload_all_plugins()
                assert len(unloaded) == 3
                
                await manager.shutdown()
                self.log_success("Multiple plugin management works correctly")
                
            except Exception as e:
                self.log_failure(f"Multiple plugins test failed: {e}")
        
        asyncio.run(_test_multiple_plugins())
    
    def test_event_handling(self) -> Any:
        """Test event handling system."""
        self.log_test("Event Handling")
        
        async def _test_event_handling():
            
    """_test_event_handling function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Track events
                events = []
                
                def event_handler(plugin_name, plugin=None) -> Any:
                    events.append(plugin_name)
                
                # Add event handlers
                manager.add_event_handler("plugin_loaded", event_handler)
                manager.add_event_handler("plugin_initialized", event_handler)
                manager.add_event_handler("plugin_started", event_handler)
                
                # Load and initialize plugin to trigger events
                await manager.load_plugin("web_extractor")
                await manager.initialize_plugin("web_extractor")
                await manager.start_plugin("web_extractor")
                
                # Check that events were triggered
                assert len(events) >= 3
                assert "web_extractor" in events
                
                await manager.shutdown()
                self.log_success("Event handling works correctly")
                
            except Exception as e:
                self.log_failure(f"Event handling test failed: {e}")
        
        asyncio.run(_test_event_handling())
    
    def test_configuration_management(self) -> Any:
        """Test configuration management."""
        self.log_test("Configuration Management")
        
        async def _test_config_management():
            
    """_test_config_management function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Load plugin
                await manager.load_plugin("web_extractor")
                
                # Get initial config
                initial_config = manager.get_plugin_config("web_extractor")
                assert initial_config is not None
                
                # Update config
                new_config = {"timeout": 45, "max_retries": 5}
                success = manager.update_plugin_config("web_extractor", new_config)
                assert success
                
                # Verify config update
                updated_config = manager.get_plugin_config("web_extractor")
                assert updated_config["timeout"] == 45
                assert updated_config["max_retries"] == 5
                
                await manager.shutdown()
                self.log_success("Configuration management works correctly")
                
            except Exception as e:
                self.log_failure(f"Configuration management test failed: {e}")
        
        asyncio.run(_test_config_management())
    
    def test_health_monitoring(self) -> Any:
        """Test health monitoring system."""
        self.log_test("Health Monitoring")
        
        async def _test_health_monitoring():
            
    """_test_health_monitoring function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Load plugin
                await manager.load_plugin("web_extractor")
                
                # Get health report
                health = manager.get_health_report()
                assert health["overall_status"] in ["healthy", "unhealthy"]
                assert health["total_plugins"] >= 1
                
                # Get plugin stats
                stats = manager.get_stats()
                assert "total_plugins" in stats
                assert "loaded_plugins" in stats
                
                await manager.shutdown()
                self.log_success("Health monitoring works correctly")
                
            except Exception as e:
                self.log_failure(f"Health monitoring test failed: {e}")
        
        asyncio.run(_test_health_monitoring())
    
    def run_performance_tests(self) -> Any:
        """Run performance tests."""
        print("\n⚡ Performance Tests")
        print("-" * 20)
        
        # Test plugin loading performance
        self.test_loading_performance()
        
        # Test concurrent plugin operations
        self.test_concurrent_operations()
        
        # Test memory usage
        self.test_memory_usage()
    
    def test_loading_performance(self) -> Any:
        """Test plugin loading performance."""
        self.log_test("Loading Performance")
        
        async def _test_loading_performance():
            
    """_test_loading_performance function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Measure loading time
                start_time = time.time()
                
                for i in range(5):
                    plugin_name = f"test_plugin_{i}"
                    await manager.load_plugin("web_extractor", {
                        "timeout": 30,
                        "max_retries": 3
                    })
                
                loading_time = time.time() - start_time
                avg_loading_time = loading_time / 5
                
                # Performance assertions
                assert avg_loading_time < 2.0  # Should load in under 2 seconds
                
                await manager.shutdown()
                self.log_success(f"Loading performance: {avg_loading_time:.2f}s average")
                
            except Exception as e:
                self.log_failure(f"Loading performance test failed: {e}")
        
        asyncio.run(_test_loading_performance())
    
    def test_concurrent_operations(self) -> Any:
        """Test concurrent plugin operations."""
        self.log_test("Concurrent Operations")
        
        async def _test_concurrent_operations():
            
    """_test_concurrent_operations function."""
try:
                # Create plugin manager
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                # Load multiple plugins concurrently
                async def load_plugin(i) -> Any:
                    plugin_name = f"concurrent_plugin_{i}"
                    return await manager.load_plugin("web_extractor", {
                        "timeout": 30,
                        "max_retries": 3
                    })
                
                start_time = time.time()
                
                # Load 5 plugins concurrently
                tasks = [load_plugin(i) for i in range(5)]
                plugins = await asyncio.gather(*tasks)
                
                concurrent_time = time.time() - start_time
                
                assert len(plugins) == 5
                assert concurrent_time < 5.0  # Should complete in under 5 seconds
                
                await manager.shutdown()
                self.log_success(f"Concurrent operations: {concurrent_time:.2f}s for 5 plugins")
                
            except Exception as e:
                self.log_failure(f"Concurrent operations test failed: {e}")
        
        asyncio.run(_test_concurrent_operations())
    
    def test_memory_usage(self) -> Any:
        """Test memory usage patterns."""
        self.log_test("Memory Usage")
        
        async def _test_memory_usage():
            
    """_test_memory_usage function."""
try:
                
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Create plugin manager and load plugins
                config = ManagerConfig(auto_discover=False, auto_load=False)
                manager = PluginManager(config)
                await manager.start()
                
                for i in range(10):
                    await manager.load_plugin("web_extractor")
                
                # Get memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory usage should be reasonable (less than 100MB increase for 10 plugins)
                assert memory_increase < 100
                
                await manager.shutdown()
                self.log_success(f"Memory usage: {memory_increase:.1f}MB increase for 10 plugins")
                
            except ImportError:
                self.log_skip("psutil not available, skipping memory test")
            except Exception as e:
                self.log_failure(f"Memory usage test failed: {e}")
        
        asyncio.run(_test_memory_usage())
    
    def log_test(self, test_name: str):
        """Log test start."""
        print(f"  Testing: {test_name}")
        self.test_results['total'] += 1
    
    def log_success(self, message: str):
        """Log test success."""
        print(f"    ✅ {message}")
        self.test_results['passed'] += 1
    
    def log_failure(self, message: str):
        """Log test failure."""
        print(f"    ❌ {message}")
        self.test_results['failed'] += 1
    
    def log_skip(self, message: str):
        """Log test skip."""
        print(f"    ⏭️ {message}")
        self.test_results['skipped'] += 1
    
    def print_summary(self) -> Any:
        """Print test summary."""
        print("\n📊 Test Summary")
        print("-" * 15)
        print(f"Total tests: {self.test_results['total']}")
        print(f"Passed: {self.test_results['passed']}")
        print(f"Failed: {self.test_results['failed']}")
        print(f"Skipped: {self.test_results['skipped']}")
        
        success_rate = (self.test_results['passed'] / self.test_results['total'] * 100) if self.test_results['total'] > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        if self.test_results['failed'] == 0:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️ {self.test_results['failed']} tests failed")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    # Parse command line arguments
    test_types = []
    verbose = False
    
    for arg in sys.argv[1:]:
        if arg == "--verbose":
            verbose = True
        elif arg.startswith("--"):
            test_types.append(arg[2:])
        else:
            test_types.append(arg)
    
    # Run tests
    tester = PluginSystemTester(verbose=verbose)
    tester.run_tests(test_types)


match __name__:
    case "__main__":
    main() 