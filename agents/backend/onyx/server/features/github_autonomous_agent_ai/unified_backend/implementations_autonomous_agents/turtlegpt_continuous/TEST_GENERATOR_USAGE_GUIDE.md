# Unit Test Generator - Complete Usage Guide

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Best Practices](#best-practices)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Installation

The test generator is already part of the `turtlegpt_continuous` package. Simply import it:

```python
from turtlegpt_continuous import (
    UnitTestGenerator,
    TestFramework,
    TestComplexity,
    create_test_generator
)
```

### Basic Example

```python
from turtlegpt_continuous import create_test_generator, TestFramework, TestComplexity

# Create generator
generator = create_test_generator(
    test_framework=TestFramework.PYTEST,
    test_complexity=TestComplexity.COMPREHENSIVE,
    target_coverage=0.80
)

# Your function to test
def add_numbers(a: int, b: int) -> int:
    return a + b

# Generate tests
analysis = generator.analyze_function(add_numbers)
test_cases = generator.generate_test_cases(analysis)
suite = generator.generate_test_suite("calculator", [add_numbers])
test_code = generator.generate_test_code(suite)

# Save to file
with open("test_calculator.py", "w") as f:
    f.write(test_code)
```

## 📖 Basic Usage

### Step 1: Create Generator

```python
generator = create_test_generator(
    test_framework=TestFramework.PYTEST,  # or TestFramework.UNITTEST
    test_complexity=TestComplexity.COMPREHENSIVE,  # BASIC, COMPREHENSIVE, EXHAUSTIVE
    target_coverage=0.80  # 0.0 to 1.0
)
```

### Step 2: Analyze Function

```python
def my_function(param1: str, param2: int) -> bool:
    """My function description."""
    return len(param1) > param2

analysis = generator.analyze_function(my_function)
print(f"Complexity: {analysis.complexity}")
print(f"Parameters: {analysis.parameters}")
print(f"Dependencies: {analysis.dependencies}")
```

### Step 3: Generate Test Cases

```python
test_cases = generator.generate_test_cases(
    analysis,
    include_edge_cases=True,
    include_error_cases=True
)

for test_case in test_cases:
    print(f"{test_case.name}: {test_case.description}")
```

### Step 4: Generate Test Suite

```python
suite = generator.generate_test_suite(
    component_name="my_module",
    functions=[my_function, another_function],
    objectives=None  # Optional custom objectives
)
```

### Step 5: Generate Test Code

```python
test_code = generator.generate_test_code(suite)

# Save to file
Path("tests/test_my_module.py").write_text(test_code)
```

## 🎯 Advanced Features

### Custom Test Objectives

```python
from turtlegpt_continuous import TestObjective

custom_objectives = [
    TestObjective(
        description="Verify basic functionality",
        priority=10,
        category="functionality"
    ),
    TestObjective(
        description="Test edge cases",
        priority=8,
        category="edge_cases"
    ),
    TestObjective(
        description="Verify error handling",
        priority=9,
        category="error_handling"
    )
]

suite = generator.generate_test_suite(
    component_name="my_component",
    functions=[my_function],
    objectives=custom_objectives
)
```

### Async Function Testing

```python
async def async_function(url: str) -> dict:
    """Async function example."""
    await asyncio.sleep(0.1)
    return {"url": url}

# Generator automatically detects async functions
analysis = generator.analyze_function(async_function)
suite = generator.generate_test_suite("async_module", [async_function])
test_code = generator.generate_test_code(suite)

# Generated code will include async test support
```

### Multiple Functions

```python
def function1(x: int) -> int:
    return x * 2

def function2(x: str) -> str:
    return x.upper()

def function3(x: list) -> int:
    return len(x)

# Generate tests for all functions at once
suite = generator.generate_test_suite(
    component_name="utils",
    functions=[function1, function2, function3]
)

print(f"Generated {len(suite.test_cases)} test cases")
```

### Test Complexity Levels

#### BASIC
- Only happy path tests
- Minimal edge cases
- Fast generation

```python
generator = create_test_generator(
    test_complexity=TestComplexity.BASIC
)
```

#### COMPREHENSIVE (Recommended)
- Happy path + edge cases
- Error scenarios
- Good coverage

```python
generator = create_test_generator(
    test_complexity=TestComplexity.COMPREHENSIVE
)
```

#### EXHAUSTIVE
- All possible test scenarios
- Maximum edge cases
- Complete coverage

```python
generator = create_test_generator(
    test_complexity=TestComplexity.EXHAUSTIVE
)
```

## 🏆 Best Practices

### 1. Test Organization

Organize tests by module structure:

```
tests/
├── unit/
│   ├── test_core.py
│   ├── test_services.py
│   └── test_utils.py
├── integration/
│   └── test_workflows.py
└── conftest.py  # Shared fixtures
```

### 2. Naming Conventions

Generated tests follow pattern: `test_<function>_<scenario>`

```python
# Good
test_calculate_sum_basic
test_calculate_sum_edge_cases
test_calculate_sum_error_handling

# Avoid
test1
test_function
test_thing
```

### 3. AAA Pattern

All generated tests follow AAA (Arrange, Act, Assert):

```python
def test_function_basic(self):
    # Arrange - Setup test data
    input_data = {"key": "value"}
    
    # Act - Execute function
    result = function_under_test(input_data)
    
    # Assert - Verify results
    assert result is not None
```

### 4. Test Isolation

- Each test is independent
- No shared state
- Proper setup/teardown

### 5. Mocking Strategy

```python
# Generator automatically identifies dependencies
# and suggests appropriate mocks

from unittest.mock import Mock, patch

@patch('module.external_api')
def test_function_with_mock(mock_api):
    # Arrange
    mock_api.return_value = {"data": "test"}
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result is not None
```

## 📝 Examples

### Example 1: Simple Function

```python
def calculate_total(items: list[float]) -> float:
    """Calculate total of items."""
    return sum(items)

generator = create_test_generator()
analysis = generator.analyze_function(calculate_total)
suite = generator.generate_test_suite("calculator", [calculate_total])
test_code = generator.generate_test_code(suite)
```

### Example 2: Function with Error Handling

```python
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

generator = create_test_generator()
analysis = generator.analyze_function(divide)
# Generator automatically detects ValueError and creates error test
test_cases = generator.generate_test_cases(analysis, include_error_cases=True)
```

### Example 3: Class Methods

```python
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        return a - b

generator = create_test_generator()
calc = Calculator()
suite = generator.generate_test_suite(
    "calculator",
    [calc.add, calc.subtract]
)
```

### Example 4: Complete Module

```python
# my_module.py
def function1(x: int) -> int:
    return x * 2

def function2(x: str) -> str:
    return x.upper()

def function3(x: list) -> int:
    return len(x)

# Generate tests for entire module
from my_module import function1, function2, function3

generator = create_test_generator()
suite = generator.generate_test_suite(
    "my_module",
    [function1, function2, function3]
)

test_code = generator.generate_test_code(suite)
Path("tests/test_my_module.py").write_text(test_code)
```

## 🔧 Troubleshooting

### Issue: Tests not generating

**Solution**: Ensure function is importable and has valid Python syntax.

```python
# Check function is valid
import inspect
inspect.signature(my_function)  # Should not raise error
```

### Issue: Missing edge cases

**Solution**: Use `TestComplexity.EXHAUSTIVE`:

```python
generator = create_test_generator(
    test_complexity=TestComplexity.EXHAUSTIVE
)
```

### Issue: Async tests not working

**Solution**: Install pytest-asyncio and ensure async markers:

```bash
pip install pytest-asyncio
```

```python
# Generated code should include:
import pytest
@pytest.mark.asyncio
async def test_async_function():
    ...
```

### Issue: Mocks not generated correctly

**Solution**: Manually add mocks after generation:

```python
# After generating, add custom mocks:
test_code = test_code.replace(
    "# TODO: Add mocks",
    "with patch('module.dependency') as mock_dep:"
)
```

## 📊 Coverage Goals

### Recommended Coverage by Component Type

| Component Type | Target Coverage | Priority |
|----------------|----------------|----------|
| Critical Business Logic | 95%+ | High |
| API Endpoints | 90%+ | High |
| Utilities | 85%+ | Medium |
| Helpers | 80%+ | Medium |
| Simple Functions | 75%+ | Low |

### Running Coverage Reports

```bash
# Install coverage
pip install pytest-cov

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

## 🎓 Learning Resources

1. **Test Generator Strategy**: See `TEST_GENERATOR_STRATEGY.md`
2. **Examples**: Run `python -m turtlegpt_continuous.test_generator_example`
3. **Pytest Documentation**: https://docs.pytest.org/
4. **Unittest Documentation**: https://docs.python.org/3/library/unittest.html

## ✅ Checklist

Before using generated tests:

- [ ] Review generated test code
- [ ] Fill in TODO comments
- [ ] Add specific assertions
- [ ] Verify mocks are correct
- [ ] Run tests to ensure they pass
- [ ] Check coverage meets target
- [ ] Update test descriptions if needed
- [ ] Add additional edge cases if necessary

## 🎉 Conclusion

The Unit Test Generator provides:

- ✅ Automated test generation
- ✅ Consistent test patterns
- ✅ Comprehensive coverage
- ✅ Industry best practices
- ✅ Time savings (70%+)

Start generating tests today and improve your code quality!
