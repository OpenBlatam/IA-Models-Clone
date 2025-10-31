# Refactored Test Generation System

## Overview

The refactored test generation system provides a comprehensive, modular, and extensible architecture for generating high-quality test cases. This system has been completely redesigned from the ground up to provide better organization, maintainability, and performance.

## Architecture

### Core Components

The system is built around several core components:

1. **Base Architecture** (`base_architecture.py`)
   - Defines abstract base classes and interfaces
   - Provides common data structures and enums
   - Establishes the foundation for all components

2. **Factory Pattern** (`factory.py`)
   - Implements factory pattern for component creation
   - Manages component registration and discovery
   - Provides centralized component management

3. **Configuration System** (`configuration.py`)
   - Manages configuration loading and validation
   - Supports multiple configuration formats (JSON, YAML)
   - Provides environment variable integration
   - Includes predefined configuration presets

4. **Plugin System** (`plugin_system.py`)
   - Enables extensible plugin architecture
   - Supports dynamic plugin loading and management
   - Provides plugin lifecycle management

5. **Unified API** (`unified_api.py`)
   - Provides a single entry point for all functionality
   - Integrates all components seamlessly
   - Offers both programmatic and convenience interfaces

6. **Concrete Implementations** (`implementations.py`)
   - Contains concrete implementations of abstract classes
   - Provides various test patterns and generators
   - Includes validators and optimizers

## Key Features

### 🏗️ Modular Architecture
- Clean separation of concerns
- Easy to extend and maintain
- Pluggable components

### ⚙️ Advanced Configuration
- Multiple configuration sources
- Environment variable support
- Predefined presets (minimal, standard, comprehensive, enterprise)
- Runtime configuration updates

### 🔌 Plugin System
- Dynamic plugin loading
- Plugin lifecycle management
- Extensible architecture
- Plugin configuration management

### 🚀 Performance Optimized
- Parallel test generation
- Caching mechanisms
- Memory management
- Performance monitoring

### 📊 Comprehensive Monitoring
- Generation metrics
- Performance tracking
- System status reporting
- Error handling and logging

### 📤 Export Capabilities
- Multiple export formats (Python, JSON)
- Customizable output structure
- Metadata inclusion

## Quick Start

### Basic Usage

```python
from core.unified_api import quick_generate

# Generate tests for a function
result = await quick_generate(
    function_signature="def calculate_sum(a: int, b: int) -> int:",
    docstring="Calculate the sum of two integers.",
    generator_type="enhanced",
    preset="standard"
)

if result["success"]:
    print(f"Generated {len(result['test_cases'])} test cases")
    for test_case in result["test_cases"]:
        print(f"- {test_case.name}: {test_case.description}")
```

### Advanced Usage

```python
from core.unified_api import create_api

# Create API instance
api = create_api()

# Load custom configuration
api.load_configuration("my_config.yaml", "yaml")

# Generate tests with custom settings
result = await api.generate_tests(
    function_signature="def complex_function(data: List[Dict]) -> Dict:",
    docstring="Process complex data with validation.",
    generator_type="enhanced",
    config_override={
        "target_coverage": 0.95,
        "include_security_tests": True,
        "max_test_cases": 50
    }
)

# Export results
api.export_tests(result["test_cases"], "output_tests.py", "python")
```

### Batch Generation

```python
from core.unified_api import batch_generate

functions = [
    {
        "name": "function1",
        "signature": "def function1(x: int) -> int:",
        "docstring": "First function description."
    },
    {
        "name": "function2", 
        "signature": "def function2(y: str) -> bool:",
        "docstring": "Second function description."
    }
]

result = batch_generate(functions, "enhanced", "comprehensive")
```

## Configuration

### Configuration Presets

The system provides several predefined configuration presets:

- **minimal**: Basic test generation with minimal coverage
- **standard**: Balanced test generation with good coverage
- **comprehensive**: Extensive test generation with high coverage
- **enterprise**: Full-featured test generation for enterprise use

### Custom Configuration

Create a configuration file in YAML or JSON format:

```yaml
# config.yaml
main_config:
  target_coverage: 0.9
  max_test_cases: 100
  include_edge_cases: true
  include_performance_tests: true
  include_security_tests: true
  complexity_level: "advanced"
  naming_convention: "descriptive"
  code_style: "pytest"
  parallel_generation: true

generators:
  enhanced:
    enabled: true
    priority: 1
    custom_config:
      max_depth: 5
      include_mocks: true

patterns:
  basic:
    enabled: true
    weight: 1.0
  edge_case:
    enabled: true
    weight: 0.8
  performance:
    enabled: true
    weight: 0.6

advanced:
  max_parallel_workers: 8
  memory_limit_mb: 2048
  timeout_seconds: 600
  cache_enabled: true
  debug_mode: false
```

### Environment Variables

You can also configure the system using environment variables:

```bash
export TEST_GEN_TARGET_COVERAGE=0.9
export TEST_GEN_MAX_TEST_CASES=100
export TEST_GEN_INCLUDE_EDGE_CASES=true
export TEST_GEN_INCLUDE_PERFORMANCE=true
export TEST_GEN_INCLUDE_SECURITY=true
export TEST_GEN_COMPLEXITY_LEVEL=advanced
export TEST_GEN_PARALLEL_GENERATION=true
export TEST_GEN_MAX_WORKERS=8
export TEST_GEN_DEBUG_MODE=false
```

## Plugin Development

### Creating a Custom Plugin

```python
from core.plugin_system import BasePlugin, PluginInfo, PluginType

class MyCustomGenerator(BasePlugin):
    def _create_plugin_info(self) -> PluginInfo:
        return PluginInfo(
            name="my_custom_generator",
            version="1.0.0",
            description="My custom test generator",
            author="Your Name",
            plugin_type=PluginType.GENERATOR
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        # Initialize your plugin
        return True
    
    async def generate_tests(self, function_signature: str, docstring: str, config: TestGenerationConfig) -> List[TestCase]:
        # Implement your test generation logic
        return []
```

### Plugin Configuration

```yaml
# plugin_config.yaml
plugins:
  my_custom_generator:
    enabled: true
    priority: 1
    config:
      custom_setting: "value"
    auto_load: true
```

## API Reference

### TestGenerationAPI

The main API class that provides access to all system functionality.

#### Methods

- `generate_tests(function_signature, docstring, generator_type, config_override)` - Generate test cases
- `generate_tests_batch(functions, generator_type, config_override)` - Batch test generation
- `load_configuration(config_path, config_format)` - Load configuration from file
- `save_configuration(output_path, config_format)` - Save configuration to file
- `use_preset(preset_name)` - Apply predefined configuration preset
- `get_available_generators()` - Get list of available generators
- `get_available_plugins()` - Get list of available plugins
- `get_system_status()` - Get system status and information
- `export_tests(test_cases, output_path, format)` - Export test cases to file

### Convenience Functions

- `quick_generate(function_signature, docstring, generator_type, preset)` - Quick test generation
- `batch_generate(functions, generator_type, preset)` - Batch test generation
- `create_api(config)` - Create API instance

## Performance Considerations

### Parallel Generation

The system supports parallel test generation to improve performance:

```python
# Enable parallel generation
config = {
    "parallel_generation": True,
    "max_parallel_workers": 8
}

result = await api.generate_tests(signature, docstring, "enhanced", config)
```

### Caching

Enable caching to improve performance for repeated operations:

```python
# Enable caching
config = {
    "cache_enabled": True,
    "cache_ttl_seconds": 3600
}
```

### Memory Management

Configure memory limits to prevent resource exhaustion:

```python
# Set memory limits
config = {
    "memory_limit_mb": 2048,
    "timeout_seconds": 600
}
```

## Error Handling

The system provides comprehensive error handling:

```python
try:
    result = await api.generate_tests(signature, docstring, "enhanced")
    if not result["success"]:
        print(f"Generation failed: {result['error']}")
except Exception as e:
    print(f"System error: {e}")
```

## Logging

Configure logging for debugging and monitoring:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug mode
api = create_api()
api.config_manager.update_config({"debug_mode": True})
```

## Examples

See the `demo_refactored_system.py` file for comprehensive examples of system usage.

## Migration from Legacy System

If you're migrating from the legacy test generation system:

1. Update imports to use the new unified API
2. Replace direct generator instantiation with factory pattern
3. Update configuration to use the new configuration system
4. Migrate custom plugins to the new plugin architecture

## Support

For questions, issues, or contributions, please refer to the project documentation or contact the development team.

## License

This system is part of the larger project and follows the same licensing terms.









