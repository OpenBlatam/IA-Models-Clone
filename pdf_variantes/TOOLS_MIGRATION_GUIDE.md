# Tools Migration Guide - PDF Variantes

## ✅ Recommended Tools Structure

### `tools/` Directory - **USE THIS**

The refactored tools structure provides a modular, extensible architecture:

```python
from tools.manager import ToolManager
from tools.refactored_health_checker import HealthChecker
from tools.refactored_benchmark import Benchmark
from tools.refactored_test_suite import TestSuite
```

**Available Refactored Tools:**
- `tools.refactored_health_checker.HealthChecker` - Health checking
- `tools.refactored_benchmark.Benchmark` - Performance benchmarking
- `tools.refactored_test_suite.TestSuite` - Automated testing

**Tool Manager:**
```python
from tools.manager import ToolManager

manager = ToolManager()
result = manager.run_tool("health")
result = manager.run_tool("benchmark")
result = manager.run_tool("test")
```

## ⚠️ Deprecated Tool Files

The following `api_*.py` files in the root directory are **deprecated**:

### Tools with Refactored Versions

#### `api_health_checker.py`
- **Status**: Deprecated
- **Replacement**: `tools.refactored_health_checker.HealthChecker`
- **Migration**: Use `tools.refactored_health_checker.HealthChecker` or `ToolManager().run_tool("health")`

#### `api_benchmark.py`
- **Status**: Deprecated
- **Replacement**: `tools.refactored_benchmark.Benchmark`
- **Migration**: Use `tools.refactored_benchmark.Benchmark` or `ToolManager().run_tool("benchmark")`

#### `api_test_suite.py`
- **Status**: Deprecated
- **Replacement**: `tools.refactored_test_suite.TestSuite`
- **Migration**: Use `tools.refactored_test_suite.TestSuite` or `ToolManager().run_tool("test")`

### Tools Pending Migration

The following tools are deprecated but don't have refactored versions yet. They should be migrated to the new structure:

#### `api_monitor.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_profiler.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_dashboard.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_logger.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_comparator.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_reporter.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_analyzer.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_alerts.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_visualizer.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

#### `api_notifier.py`
- **Status**: Deprecated (pending migration)
- **Migration**: Should be migrated to `tools/` structure

### Utility Files

#### `api_utils.py`
- **Status**: Deprecated
- **Replacement**: `tools.utils`
- **Migration**: Use `tools.utils` for utility functions

#### `api_config.py`
- **Status**: Deprecated
- **Replacement**: `tools.config`
- **Migration**: Use `tools.config.ToolConfig` and `get_config()`

## 🏗️ Tools Structure

```
pdf_variantes/
├── tools/                          # ✅ Refactored tools structure
│   ├── base.py                     # ✅ BaseAPITool, ToolResult
│   ├── config.py                   # ✅ ToolConfig
│   ├── registry.py                 # ✅ ToolRegistry
│   ├── factory.py                  # ✅ ToolFactory
│   ├── chain.py                    # ✅ ToolChain
│   ├── executor.py                 # ✅ ToolExecutor
│   ├── manager.py                  # ✅ ToolManager
│   ├── plugins.py                  # ✅ Plugin System
│   ├── utils.py                    # ✅ Utility functions
│   ├── refactored_health_checker.py # ✅ Health Checker
│   ├── refactored_benchmark.py     # ✅ Benchmark
│   └── refactored_test_suite.py    # ✅ Test Suite
├── api_health_checker.py          # ⚠️ Deprecated
├── api_benchmark.py                # ⚠️ Deprecated
├── api_test_suite.py               # ⚠️ Deprecated
├── api_monitor.py                  # ⚠️ Deprecated
├── api_profiler.py                 # ⚠️ Deprecated
├── api_dashboard.py                # ⚠️ Deprecated
├── api_logger.py                   # ⚠️ Deprecated
├── api_comparator.py               # ⚠️ Deprecated
├── api_reporter.py                 # ⚠️ Deprecated
├── api_analyzer.py                 # ⚠️ Deprecated
├── api_alerts.py                   # ⚠️ Deprecated
├── api_visualizer.py               # ⚠️ Deprecated
├── api_notifier.py                 # ⚠️ Deprecated
├── api_utils.py                    # ⚠️ Deprecated
└── api_config.py                   # ⚠️ Deprecated
```

## 📝 Usage Examples

### Using Refactored Tools

#### Health Checker
```python
# Option 1: Direct import
from tools.refactored_health_checker import HealthChecker

checker = HealthChecker(base_url="http://localhost:8000")
result = checker.run(endpoints=["/health", "/docs"])

# Option 2: Via ToolManager
from tools.manager import ToolManager

manager = ToolManager()
result = manager.run_tool("health", endpoints=["/health"])
```

#### Benchmark
```python
# Option 1: Direct import
from tools.refactored_benchmark import Benchmark

benchmark = Benchmark(base_url="http://localhost:8000")
result = benchmark.run(endpoint="/health", iterations=10)

# Option 2: Via ToolManager
from tools.manager import ToolManager

manager = ToolManager()
result = manager.run_tool("benchmark", endpoint="/health", iterations=10)
```

#### Test Suite
```python
# Option 1: Direct import
from tools.refactored_test_suite import TestSuite

suite = TestSuite(base_url="http://localhost:8000")
result = suite.run(tests=[...])

# Option 2: Via ToolManager
from tools.manager import ToolManager

manager = ToolManager()
result = manager.run_tool("test", tests=[...])
```

### Using Tool Manager

```python
from tools.manager import ToolManager

manager = ToolManager()

# List available tools
tools = manager.list_tools()

# Run a tool
result = manager.run_tool("health", endpoints=["/health"])

# Run multiple tools
results = manager.run_tools([
    {"name": "health", "kwargs": {"endpoints": ["/health"]}},
    {"name": "benchmark", "kwargs": {"endpoint": "/health", "iterations": 10}}
])
```

### Using Tool Chain

```python
from tools.chain import create_chain

chain = create_chain()
chain.add_tool("health", endpoints=["/health"])
chain.add_tool("benchmark", endpoint="/health", iterations=10)
results = chain.execute()
chain.print_summary()
```

### Using Tool Executor

```python
from tools.executor import ToolExecutor

executor = ToolExecutor()

# Simple execution
result = executor.execute("health")

# Parallel execution
results = executor.execute_parallel([
    {"name": "health", "kwargs": {}},
    {"name": "benchmark", "kwargs": {}}
])

# Chain execution
results = executor.execute_chain([
    {"name": "health", "kwargs": {}},
    {"name": "test", "kwargs": {}}
])
```

## 🔄 Migration Guide

### From `api_health_checker.py`
```python
# Old
from api_health_checker import APIHealthChecker
checker = APIHealthChecker()
result = checker.check_health_endpoint()

# New
from tools.refactored_health_checker import HealthChecker
checker = HealthChecker()
result = checker.run(endpoints=["/health"])
```

### From `api_benchmark.py`
```python
# Old
from api_benchmark import APIBenchmark
benchmark = APIBenchmark()
result = benchmark.benchmark_endpoint("/health", iterations=10)

# New
from tools.refactored_benchmark import Benchmark
benchmark = Benchmark()
result = benchmark.run(endpoint="/health", iterations=10)
```

### From `api_test_suite.py`
```python
# Old
from api_test_suite import APITestSuite
suite = APITestSuite()
result = suite.run_tests(tests)

# New
from tools.refactored_test_suite import TestSuite
suite = TestSuite()
result = suite.run(tests=tests)
```

### From `api_utils.py`
```python
# Old
from api_utils import format_response_time, validate_json

# New
from tools.utils import format_response_time, validate_json
```

### From `api_config.py`
```python
# Old
from api_config import APIConfigManager
config = APIConfigManager().get_config()

# New
from tools.config import ToolConfig, get_config
config = get_config()
```

## 🚀 Creating New Tools

To create a new tool in the refactored structure:

```python
from tools.base import BaseAPITool, ToolResult
from tools.registry import register_tool

@register_tool("my_tool")
class MyTool(BaseAPITool):
    def run(self, **kwargs) -> ToolResult:
        response = self.make_request("GET", "/endpoint")
        return ToolResult(
            success=response.status_code == 200,
            message="Success",
            data={"status": response.status_code}
        )
```

## 📚 Additional Resources

- See `tools/README.md` for tools package documentation
- See `tools/REFACTORING_V3.md` for refactoring details
- See `tools/ADVANCED_FEATURES.md` for advanced features
- See `tools/INTEGRATION_GUIDE.md` for integration guide
- See `REFACTORING_STATUS.md` for refactoring progress






