from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import sys
import time
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import logging
import traceback
    from config import load_config, AIVideoConfig, ConfigManager
    from integrated_workflow import IntegratedVideoWorkflow, create_integrated_workflow
    from plugins import PluginManager, ManagerConfig, ValidationLevel
    from plugins.integration import create_plugin_integration
    from video_workflow import run_full_workflow
    from models import AIVideo
                from plugins.base import PluginMetadata
                from main import AIVideoSystem
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
AI Video System - Comprehensive Test Suite

This script provides comprehensive testing for the complete AI video system,
including plugins, workflow, configuration, and integration components.
"""


# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import system components
try:
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_SUCCESS = False

logger = logging.getLogger(__name__)


class AIVideoTester:
    """
    Comprehensive tester for the AI Video System.
    
    This class provides:
    - Unit tests for individual components
    - Integration tests for system components
    - Performance tests
    - Plugin system tests
    - Configuration tests
    - End-to-end workflow tests
    """
    
    def __init__(self, config_file: Optional[str] = None):
        
    """__init__ function."""
self.config_file = config_file
        self.test_results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'plugin_tests': {},
            'configuration_tests': {},
            'workflow_tests': {}
        }
        self.start_time = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    async def run_all_tests(self, options: Dict[str, bool]) -> bool:
        """
        Run all tests based on options.
        
        Args:
            options: Test options
            
        Returns:
            bool: True if all tests passed
        """
        logger.info("🧪 Starting AI Video System tests...")
        
        all_passed = True
        
        try:
            # Configuration tests
            if options.get('configuration', True):
                logger.info("📋 Running configuration tests...")
                config_passed = await self._test_configuration()
                all_passed = all_passed and config_passed
            
            # Plugin system tests
            if options.get('plugins', True):
                logger.info("🔌 Running plugin system tests...")
                plugin_passed = await self._test_plugin_system()
                all_passed = all_passed and plugin_passed
            
            # Unit tests
            if options.get('unit', True):
                logger.info("🔬 Running unit tests...")
                unit_passed = await self._test_units()
                all_passed = all_passed and unit_passed
            
            # Integration tests
            if options.get('integration', True):
                logger.info("🔗 Running integration tests...")
                integration_passed = await self._test_integration()
                all_passed = all_passed and integration_passed
            
            # Performance tests
            if options.get('performance', True):
                logger.info("⚡ Running performance tests...")
                performance_passed = await self._test_performance()
                all_passed = all_passed and performance_passed
            
            # Workflow tests
            if options.get('workflow', True):
                logger.info("🔄 Running workflow tests...")
                workflow_passed = await self._test_workflow()
                all_passed = all_passed and workflow_passed
            
            # End-to-end tests
            if options.get('e2e', True):
                logger.info("🎯 Running end-to-end tests...")
                e2e_passed = await self._test_end_to_end()
                all_passed = all_passed and e2e_passed
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            traceback.print_exc()
            all_passed = False
        
        # Print results
        self._print_test_results()
        
        return all_passed
    
    async def _test_configuration(self) -> bool:
        """Test configuration system."""
        try:
            # Test configuration loading
            config = load_config(self.config_file)
            assert isinstance(config, AIVideoConfig), "Configuration should be AIVideoConfig instance"
            
            # Test configuration manager
            manager = ConfigManager(self.config_file)
            config_from_manager = manager.get_config()
            assert isinstance(config_from_manager, AIVideoConfig), "Manager should return AIVideoConfig"
            
            # Test configuration validation
            assert config.workflow.max_concurrent_workflows > 0, "Max concurrent workflows should be positive"
            assert config.ai.max_tokens > 0, "Max tokens should be positive"
            assert 0 <= config.ai.temperature <= 2, "Temperature should be between 0 and 2"
            
            # Test configuration serialization
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict), "Configuration should serialize to dict"
            
            # Test configuration deserialization
            config_from_dict = AIVideoConfig.from_dict(config_dict)
            assert isinstance(config_from_dict, AIVideoConfig), "Configuration should deserialize from dict"
            
            self.test_results['configuration_tests'] = {
                'config_loading': True,
                'config_validation': True,
                'config_serialization': True,
                'config_manager': True
            }
            
            logger.info("✅ Configuration tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration tests failed: {e}")
            self.test_results['configuration_tests'] = {
                'config_loading': False,
                'config_validation': False,
                'config_serialization': False,
                'config_manager': False,
                'error': str(e)
            }
            return False
    
    async def _test_plugin_system(self) -> bool:
        """Test plugin system."""
        try:
            # Test plugin manager creation
            plugin_config = ManagerConfig(
                auto_discover=True,
                auto_load=True,
                validation_level=ValidationLevel.STANDARD
            )
            
            manager = PluginManager(plugin_config)
            assert manager is not None, "Plugin manager should be created"
            
            # Test plugin manager startup
            success = await manager.start()
            assert success, "Plugin manager should start successfully"
            
            # Test plugin discovery
            discovered = await manager.discover_plugins()
            assert isinstance(discovered, list), "Plugin discovery should return list"
            
            # Test plugin loading
            loaded = await manager.load_all_plugins()
            assert isinstance(loaded, list), "Plugin loading should return list"
            
            # Test plugin statistics
            stats = manager.get_stats()
            assert isinstance(stats, dict), "Plugin stats should be dict"
            
            # Test health report
            health = manager.get_health_report()
            assert isinstance(health, dict), "Health report should be dict"
            
            # Test plugin integration
            integration_manager, bridge = await create_plugin_integration(plugin_config)
            assert integration_manager is not None, "Integration manager should be created"
            assert bridge is not None, "Bridge should be created"
            
            # Test bridged components
            components = await bridge.create_bridged_components()
            assert isinstance(components, dict), "Bridged components should be dict"
            
            # Test shutdown
            await manager.shutdown()
            
            self.test_results['plugin_tests'] = {
                'manager_creation': True,
                'manager_startup': True,
                'plugin_discovery': True,
                'plugin_loading': True,
                'plugin_stats': True,
                'health_report': True,
                'integration': True,
                'bridged_components': True,
                'shutdown': True
            }
            
            logger.info("✅ Plugin system tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Plugin system tests failed: {e}")
            self.test_results['plugin_tests'] = {
                'manager_creation': False,
                'manager_startup': False,
                'plugin_discovery': False,
                'plugin_loading': False,
                'plugin_stats': False,
                'health_report': False,
                'integration': False,
                'bridged_components': False,
                'shutdown': False,
                'error': str(e)
            }
            return False
    
    async def _test_units(self) -> bool:
        """Test individual units."""
        try:
            results = {}
            
            # Test AIVideo model
            try:
                video = AIVideo(
                    title="Test Video",
                    description="Test description",
                    prompts=["Test prompt"],
                    ai_model="test-model",
                    duration=30.0,
                    resolution="1920x1080"
                )
                assert video.title == "Test Video", "Video title should match"
                assert video.duration == 30.0, "Video duration should match"
                results['ai_video_model'] = True
            except Exception as e:
                results['ai_video_model'] = False
                results['ai_video_model_error'] = str(e)
            
            # Test configuration validation
            try:
                config = AIVideoConfig()
                config._validate_config()
                results['config_validation'] = True
            except Exception as e:
                results['config_validation'] = False
                results['config_validation_error'] = str(e)
            
            # Test plugin metadata
            try:
                metadata = PluginMetadata(
                    name="test_plugin",
                    version="1.0.0",
                    description="Test plugin",
                    author="Test Author",
                    category="test"
                )
                assert metadata.name == "test_plugin", "Plugin name should match"
                results['plugin_metadata'] = True
            except Exception as e:
                results['plugin_metadata'] = False
                results['plugin_metadata_error'] = str(e)
            
            self.test_results['unit_tests'] = results
            
            all_passed = all(results.values())
            if all_passed:
                logger.info("✅ Unit tests passed")
            else:
                logger.warning("⚠️ Some unit tests failed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Unit tests failed: {e}")
            self.test_results['unit_tests'] = {'error': str(e)}
            return False
    
    async def _test_integration(self) -> bool:
        """Test component integration."""
        try:
            results = {}
            
            # Test integrated workflow creation
            try:
                workflow = await create_integrated_workflow()
                assert workflow is not None, "Integrated workflow should be created"
                results['workflow_creation'] = True
            except Exception as e:
                results['workflow_creation'] = False
                results['workflow_creation_error'] = str(e)
            
            # Test plugin integration
            try:
                plugin_config = ManagerConfig()
                integration_manager, bridge = await create_plugin_integration(plugin_config)
                assert integration_manager is not None, "Integration manager should be created"
                assert bridge is not None, "Bridge should be created"
                results['plugin_integration'] = True
            except Exception as e:
                results['plugin_integration'] = False
                results['plugin_integration_error'] = str(e)
            
            # Test configuration integration
            try:
                config = load_config(self.config_file)
                manager = ConfigManager(self.config_file)
                assert config is not None, "Configuration should be loaded"
                assert manager is not None, "Configuration manager should be created"
                results['config_integration'] = True
            except Exception as e:
                results['config_integration'] = False
                results['config_integration_error'] = str(e)
            
            self.test_results['integration_tests'] = results
            
            all_passed = all(results.values())
            if all_passed:
                logger.info("✅ Integration tests passed")
            else:
                logger.warning("⚠️ Some integration tests failed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
            self.test_results['integration_tests'] = {'error': str(e)}
            return False
    
    async def _test_performance(self) -> bool:
        """Test system performance."""
        try:
            results = {}
            
            # Test configuration loading performance
            start_time = time.time()
            config = load_config(self.config_file)
            config_load_time = time.time() - start_time
            results['config_load_time'] = config_load_time
            results['config_load_fast'] = config_load_time < 1.0
            
            # Test plugin manager startup performance
            start_time = time.time()
            plugin_config = ManagerConfig()
            manager = PluginManager(plugin_config)
            await manager.start()
            plugin_startup_time = time.time() - start_time
            results['plugin_startup_time'] = plugin_startup_time
            results['plugin_startup_fast'] = plugin_startup_time < 5.0
            
            # Test plugin discovery performance
            start_time = time.time()
            discovered = await manager.discover_plugins()
            discovery_time = time.time() - start_time
            results['discovery_time'] = discovery_time
            results['discovery_fast'] = discovery_time < 2.0
            
            # Test plugin loading performance
            start_time = time.time()
            loaded = await manager.load_all_plugins()
            loading_time = time.time() - start_time
            results['loading_time'] = loading_time
            results['loading_fast'] = loading_time < 3.0
            
            # Test shutdown performance
            start_time = time.time()
            await manager.shutdown()
            shutdown_time = time.time() - start_time
            results['shutdown_time'] = shutdown_time
            results['shutdown_fast'] = shutdown_time < 2.0
            
            self.test_results['performance_tests'] = results
            
            all_passed = all([
                results['config_load_fast'],
                results['plugin_startup_fast'],
                results['discovery_fast'],
                results['loading_fast'],
                results['shutdown_fast']
            ])
            
            if all_passed:
                logger.info("✅ Performance tests passed")
            else:
                logger.warning("⚠️ Some performance tests failed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Performance tests failed: {e}")
            self.test_results['performance_tests'] = {'error': str(e)}
            return False
    
    async def _test_workflow(self) -> bool:
        """Test workflow functionality."""
        try:
            results = {}
            
            # Test workflow creation
            try:
                workflow = await create_integrated_workflow()
                assert workflow is not None, "Workflow should be created"
                results['workflow_creation'] = True
            except Exception as e:
                results['workflow_creation'] = False
                results['workflow_creation_error'] = str(e)
            
            # Test system statistics
            try:
                stats = workflow.get_system_stats()
                assert isinstance(stats, dict), "System stats should be dict"
                results['system_stats'] = True
            except Exception as e:
                results['system_stats'] = False
                results['system_stats_error'] = str(e)
            
            # Test health status
            try:
                health = workflow.get_health_status()
                assert isinstance(health, dict), "Health status should be dict"
                results['health_status'] = True
            except Exception as e:
                results['health_status'] = False
                results['health_status_error'] = str(e)
            
            # Test shutdown
            try:
                await workflow.shutdown()
                results['workflow_shutdown'] = True
            except Exception as e:
                results['workflow_shutdown'] = False
                results['workflow_shutdown_error'] = str(e)
            
            self.test_results['workflow_tests'] = results
            
            all_passed = all(results.values())
            if all_passed:
                logger.info("✅ Workflow tests passed")
            else:
                logger.warning("⚠️ Some workflow tests failed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Workflow tests failed: {e}")
            self.test_results['workflow_tests'] = {'error': str(e)}
            return False
    
    async def _test_end_to_end(self) -> bool:
        """Test end-to-end functionality."""
        try:
            results = {}
            
            # Test complete system initialization
            try:
                system = AIVideoSystem(self.config_file)
                success = await system.initialize()
                assert success, "System should initialize successfully"
                results['system_initialization'] = True
            except Exception as e:
                results['system_initialization'] = False
                results['system_initialization_error'] = str(e)
            
            # Test system statistics
            try:
                stats = system.get_system_stats()
                assert isinstance(stats, dict), "System stats should be dict"
                results['system_stats'] = True
            except Exception as e:
                results['system_stats'] = False
                results['system_stats_error'] = str(e)
            
            # Test health status
            try:
                health = system.get_health_status()
                assert isinstance(health, dict), "Health status should be dict"
                results['system_health'] = True
            except Exception as e:
                results['system_health'] = False
                results['system_health_error'] = str(e)
            
            # Test system shutdown
            try:
                await system.shutdown()
                results['system_shutdown'] = True
            except Exception as e:
                results['system_shutdown'] = False
                results['system_shutdown_error'] = str(e)
            
            self.test_results['e2e_tests'] = results
            
            all_passed = all(results.values())
            if all_passed:
                logger.info("✅ End-to-end tests passed")
            else:
                logger.warning("⚠️ Some end-to-end tests failed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ End-to-end tests failed: {e}")
            self.test_results['e2e_tests'] = {'error': str(e)}
            return False
    
    def _print_test_results(self) -> Any:
        """Print comprehensive test results."""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("🧪 AI Video System Test Results")
        print("="*80)
        
        print(f"\n⏱️ Total Test Time: {total_time:.2f} seconds")
        
        # Summary
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in self.test_results.items():
            if isinstance(results, dict) and 'error' not in results:
                category_tests = len(results)
                category_passed = sum(1 for v in results.values() if v is True)
                total_tests += category_tests
                passed_tests += category_passed
                
                status = "✅ PASSED" if category_passed == category_tests else "⚠️ PARTIAL" if category_passed > 0 else "❌ FAILED"
                print(f"\n{test_category.replace('_', ' ').title()}: {status}")
                print(f"  Tests: {category_passed}/{category_tests} passed")
                
                # Show failed tests
                failed_tests = [k for k, v in results.items() if v is False]
                if failed_tests:
                    print(f"  Failed: {', '.join(failed_tests)}")
            elif 'error' in results:
                print(f"\n{test_category.replace('_', ' ').title()}: ❌ FAILED")
                print(f"  Error: {results['error']}")
        
        # Overall result
        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
            print(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
            
            if success_rate == 100:
                print("🎉 All tests passed! System is ready for use.")
            elif success_rate >= 80:
                print("✅ Most tests passed. System is mostly functional.")
            elif success_rate >= 60:
                print("⚠️ Some tests failed. System may have issues.")
            else:
                print("❌ Many tests failed. System needs attention.")
        
        # Performance summary
        if 'performance_tests' in self.test_results:
            perf = self.test_results['performance_tests']
            if 'error' not in perf:
                print(f"\n⚡ Performance Summary:")
                print(f"  Config Loading: {perf.get('config_load_time', 0):.3f}s")
                print(f"  Plugin Startup: {perf.get('plugin_startup_time', 0):.3f}s")
                print(f"  Plugin Discovery: {perf.get('discovery_time', 0):.3f}s")
                print(f"  Plugin Loading: {perf.get('loading_time', 0):.3f}s")
                print(f"  Shutdown: {perf.get('shutdown_time', 0):.3f}s")
        
        print("\n" + "="*80)


def main():
    """Main test function."""
    if not IMPORTS_SUCCESS:
        print("❌ Cannot run tests due to import errors")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="AI Video System Test Suite")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--no-config", action="store_true", help="Skip configuration tests")
    parser.add_argument("--no-plugins", action="store_true", help="Skip plugin tests")
    parser.add_argument("--no-unit", action="store_true", help="Skip unit tests")
    parser.add_argument("--no-integration", action="store_true", help="Skip integration tests")
    parser.add_argument("--no-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--no-workflow", action="store_true", help="Skip workflow tests")
    parser.add_argument("--no-e2e", action="store_true", help="Skip end-to-end tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", help="Output results to file")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create tester
    tester = AIVideoTester(args.config)
    
    # Prepare options
    options = {
        'configuration': not args.no_config,
        'plugins': not args.no_plugins,
        'unit': not args.no_unit,
        'integration': not args.no_integration,
        'performance': not args.no_performance,
        'workflow': not args.no_workflow,
        'e2e': not args.no_e2e
    }
    
    # Run tests
    success = asyncio.run(tester.run_all_tests(options))
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(tester.test_results, f, indent=2, default=str)
        print(f"\n📄 Test results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


match __name__:
    case "__main__":
    main() 