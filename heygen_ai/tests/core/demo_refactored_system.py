"""
Demo of Refactored Test Generation System
========================================

This demo showcases the refactored test generation system with its
new architecture, configuration management, and plugin system.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from .unified_api import TestGenerationAPI, create_api, quick_generate, batch_generate
from .configuration import ConfigurationManager, ConfigurationPreset
from .plugin_system import PluginManager, PluginType
from .factory import TestGeneratorFactory, component_registry
from .implementations import EnhancedTestGenerator


async def demo_basic_usage():
    """Demo basic usage of the refactored system"""
    print("🚀 Demo: Basic Usage of Refactored Test Generation System")
    print("=" * 60)
    
    # Create API instance
    api = create_api()
    
    # Use a preset configuration
    api.use_preset("standard")
    
    # Generate tests for a simple function
    function_signature = "def calculate_sum(a: int, b: int) -> int:"
    docstring = "Calculate the sum of two integers."
    
    print(f"📝 Function: {function_signature}")
    print(f"📖 Docstring: {docstring}")
    print()
    
    result = await api.generate_tests(function_signature, docstring, "enhanced")
    
    if result["success"]:
        print(f"✅ Generated {len(result['test_cases'])} test cases")
        print(f"⏱️  Generation time: {result['generation_time']:.3f}s")
        print()
        
        # Show first few test cases
        for i, test_case in enumerate(result["test_cases"][:3]):
            print(f"🧪 Test {i+1}: {test_case.name}")
            print(f"   Description: {test_case.description}")
            print(f"   Category: {test_case.category.value}")
            print(f"   Priority: {test_case.priority.value}")
            print()
    else:
        print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")


async def demo_advanced_configuration():
    """Demo advanced configuration management"""
    print("⚙️  Demo: Advanced Configuration Management")
    print("=" * 60)
    
    # Create API with custom configuration
    api = create_api()
    
    # Load configuration from file (if exists)
    config_path = "test_config.yaml"
    if Path(config_path).exists():
        api.load_configuration(config_path, "yaml")
        print(f"📁 Loaded configuration from {config_path}")
    else:
        # Use enterprise preset
        api.use_preset("enterprise")
        print("🏢 Using enterprise preset configuration")
    
    # Show configuration summary
    status = api.get_system_status()
    config_summary = status["configuration"]
    
    print("\n📊 Configuration Summary:")
    print(f"   Target Coverage: {config_summary['generation_settings']['target_coverage']}")
    print(f"   Max Test Cases: {config_summary['generation_settings']['max_test_cases']}")
    print(f"   Complexity Level: {config_summary['generation_settings']['complexity_level']}")
    print(f"   Parallel Generation: {config_summary['generation_settings']['parallel_generation']}")
    print(f"   Code Style: {config_summary['code_settings']['code_style']}")
    print(f"   Output Directory: {config_summary['output_settings']['output_directory']}")
    print()


async def demo_batch_generation():
    """Demo batch test generation"""
    print("📦 Demo: Batch Test Generation")
    print("=" * 60)
    
    # Define multiple functions to test
    functions = [
        {
            "name": "calculate_sum",
            "signature": "def calculate_sum(a: int, b: int) -> int:",
            "docstring": "Calculate the sum of two integers."
        },
        {
            "name": "validate_email",
            "signature": "def validate_email(email: str) -> bool:",
            "docstring": "Validate if an email address is properly formatted."
        },
        {
            "name": "process_list",
            "signature": "def process_list(items: List[str]) -> List[str]:",
            "docstring": "Process a list of strings and return filtered results."
        }
    ]
    
    print(f"📝 Processing {len(functions)} functions...")
    
    # Generate tests for all functions
    result = batch_generate(functions, "enhanced", "comprehensive")
    
    if result["success"]:
        print(f"✅ Batch generation completed in {result['total_time']:.3f}s")
        print()
        
        # Show results for each function
        for func_name, func_result in result["results"].items():
            if func_result["success"]:
                test_count = len(func_result["test_cases"])
                print(f"🧪 {func_name}: {test_count} test cases generated")
            else:
                print(f"❌ {func_name}: {func_result.get('error', 'Unknown error')}")
    else:
        print(f"❌ Batch generation failed: {result.get('error', 'Unknown error')}")


async def demo_plugin_system():
    """Demo plugin system capabilities"""
    print("🔌 Demo: Plugin System")
    print("=" * 60)
    
    # Get available plugins
    api = create_api()
    available_plugins = api.get_available_plugins()
    
    print(f"📋 Available Plugins: {len(available_plugins)}")
    for plugin in available_plugins:
        print(f"   🔧 {plugin['name']} v{plugin['version']}")
        print(f"      Type: {plugin['type']}")
        print(f"      Description: {plugin['description']}")
        print()
    
    # Show plugin status
    status = api.get_system_status()
    plugin_status = status["plugins"]
    
    print(f"📊 Plugin Status:")
    print(f"   Total Plugins: {plugin_status['total_plugins']}")
    print(f"   Active Plugins: {plugin_status['active_plugins']}")
    print(f"   Plugin Directories: {len(plugin_status['plugin_directories'])}")
    print()


async def demo_export_capabilities():
    """Demo test export capabilities"""
    print("📤 Demo: Test Export Capabilities")
    print("=" * 60)
    
    # Generate some test cases
    function_signature = "def advanced_calculation(x: float, y: float, operation: str) -> float:"
    docstring = "Perform advanced mathematical calculations based on operation type."
    
    result = await quick_generate(function_signature, docstring, "enhanced", "standard")
    
    if result["success"] and result["test_cases"]:
        test_cases = result["test_cases"]
        
        # Export to Python file
        python_output = "generated_tests.py"
        api = create_api()
        
        if api.export_tests(test_cases, python_output, "python"):
            print(f"✅ Exported {len(test_cases)} test cases to {python_output}")
        
        # Export to JSON file
        json_output = "generated_tests.json"
        if api.export_tests(test_cases, json_output, "json"):
            print(f"✅ Exported {len(test_cases)} test cases to {json_output}")
        
        print()
    else:
        print("❌ No test cases to export")


async def demo_performance_comparison():
    """Demo performance comparison between different generators"""
    print("⚡ Demo: Performance Comparison")
    print("=" * 60)
    
    function_signature = "def complex_algorithm(data: List[Dict], threshold: float) -> Dict[str, Any]:"
    docstring = "Process complex data with multiple validation steps and transformations."
    
    generators = ["enhanced", "basic", "advanced"]
    results = {}
    
    for generator_type in generators:
        print(f"🔄 Testing {generator_type} generator...")
        
        start_time = time.time()
        result = await quick_generate(function_signature, docstring, generator_type, "standard")
        end_time = time.time()
        
        if result["success"]:
            results[generator_type] = {
                "test_count": len(result["test_cases"]),
                "generation_time": end_time - start_time,
                "success": True
            }
        else:
            results[generator_type] = {
                "test_count": 0,
                "generation_time": end_time - start_time,
                "success": False,
                "error": result.get("error", "Unknown error")
            }
    
    print("\n📊 Performance Results:")
    print(f"{'Generator':<12} {'Tests':<8} {'Time (s)':<10} {'Status':<8}")
    print("-" * 45)
    
    for generator_type, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{generator_type:<12} {result['test_count']:<8} {result['generation_time']:<10.3f} {status:<8}")
    
    print()


async def demo_system_status():
    """Demo system status and information"""
    print("📊 Demo: System Status and Information")
    print("=" * 60)
    
    api = create_api()
    status = api.get_system_status()
    
    # Configuration info
    config = status["configuration"]
    print("⚙️  Configuration:")
    print(f"   Target Coverage: {config['generation_settings']['target_coverage']}")
    print(f"   Max Test Cases: {config['generation_settings']['max_test_cases']}")
    print(f"   Parallel Workers: {config['performance_settings']['parallel_workers']}")
    print(f"   Cache Enabled: {config['performance_settings']['enable_caching']}")
    print()
    
    # Component info
    components = status["components"]
    print("🧩 Components:")
    print(f"   Generators: {components['generators']['count']} registered")
    print(f"   Patterns: {components['patterns']['count']} registered")
    print(f"   Validators: {components['validators']['count']} registered")
    print(f"   Optimizers: {components['optimizers']['count']} registered")
    print()
    
    # Plugin info
    plugins = status["plugins"]
    print("🔌 Plugins:")
    print(f"   Total: {plugins['total_plugins']}")
    print(f"   Active: {plugins['active_plugins']}")
    print(f"   Directories: {len(plugins['plugin_directories'])}")
    print()
    
    # Available options
    print("📋 Available Options:")
    print(f"   Generators: {', '.join(status['available_generators'])}")
    print(f"   Presets: {', '.join(status['available_presets'])}")
    print()


async def main():
    """Run all demos"""
    print("🎯 Refactored Test Generation System - Comprehensive Demo")
    print("=" * 80)
    print()
    
    try:
        # Run all demos
        await demo_basic_usage()
        await demo_advanced_configuration()
        await demo_batch_generation()
        await demo_plugin_system()
        await demo_export_capabilities()
        await demo_performance_comparison()
        await demo_system_status()
        
        print("🎉 All demos completed successfully!")
        print()
        print("📚 Key Features Demonstrated:")
        print("   ✅ Unified API for easy usage")
        print("   ✅ Advanced configuration management")
        print("   ✅ Batch test generation")
        print("   ✅ Plugin system extensibility")
        print("   ✅ Test export capabilities")
        print("   ✅ Performance monitoring")
        print("   ✅ System status reporting")
        print()
        print("🚀 The refactored system is ready for production use!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())









