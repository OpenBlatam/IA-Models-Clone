# Test Case Generation Guide

## Overview

This guide explains how to use the advanced test generation utilities to create unique, diverse, and intuitive unit tests for functions.

## Principles

### 1. Unique Tests
Each test case should cover a **distinct scenario** that hasn't been covered by other tests. Avoid redundant test cases that test the same behavior.

### 2. Diverse Coverage
Tests should cover:
- **Happy paths**: Normal, expected usage
- **Edge cases**: Boundary conditions, unusual but valid inputs
- **Error conditions**: Invalid inputs, missing data, exceptions
- **Type validation**: Wrong types, None values
- **State changes**: Before/after state verification
- **Side effects**: External changes, file operations, etc.

### 3. Intuitive Tests
- **Clear naming**: Test names should describe what is being tested
- **Descriptive assertions**: Assertions should clearly express expected behavior
- **Good documentation**: Docstrings explain the test's purpose
- **Readable structure**: Easy to understand test flow

## Usage

### Basic Usage

```python
from tests.test_generation_utils import generate_tests_for_function
from core.project_generator import ProjectGenerator

# Generate tests for a function
test_code = generate_tests_for_function(ProjectGenerator._sanitize_name)
print(test_code)
```

### Generating Tests for a Class

```python
from tests.test_generation_utils import generate_tests_for_class
from core.project_generator import ProjectGenerator

# Generate tests for all public methods
test_code = generate_tests_for_class(ProjectGenerator)
print(test_code)
```

### Using TestGenerator Directly

```python
from tests.test_generation_utils import TestGenerator
from core.project_generator import ProjectGenerator

generator = TestGenerator()

# Analyze a function
signature = generator.analyze_function(ProjectGenerator._sanitize_name)

# Extract test scenarios
test_cases = generator.extract_test_scenarios(signature)

# Generate test code
test_code = generator.generate_test_code(
    ProjectGenerator._sanitize_name,
    class_name="ProjectGenerator"
)
```

## Test Categories

### Happy Path Tests
Test normal, expected usage with valid inputs.

```python
def test_sanitize_name_with_valid_inputs(self):
    """Test _sanitize_name with valid typical inputs"""
    result = ProjectGenerator._sanitize_name("Test Project")
    assert result == "test_project"
```

### Edge Case Tests
Test boundary conditions and unusual but valid inputs.

```python
def test_sanitize_name_with_long_name(self):
    """Test _sanitize_name with very long name"""
    long_name = "a" * 10000
    result = ProjectGenerator._sanitize_name(long_name)
    assert len(result) <= 50  # MAX_PROJECT_NAME_LENGTH
```

### Error Condition Tests
Test invalid inputs and exception handling.

```python
def test_sanitize_name_raises_error_with_empty_name(self):
    """Test _sanitize_name raises ValueError with empty name"""
    with pytest.raises(ValueError):
        ProjectGenerator._sanitize_name("")
```

### Boundary Value Tests
Test values at the boundaries of valid ranges.

```python
def test_sanitize_name_with_single_char(self):
    """Test _sanitize_name with single character"""
    result = ProjectGenerator._sanitize_name("a")
    assert result == "a"
```

### Type Validation Tests
Test type checking and wrong type handling.

```python
def test_sanitize_name_with_wrong_type(self):
    """Test _sanitize_name with wrong type"""
    with pytest.raises(TypeError):
        ProjectGenerator._sanitize_name(123)
```

## Best Practices

### 1. Test Naming Convention
- Use descriptive names: `test_function_name_scenario`
- Include the condition: `test_function_name_with_empty_input`
- Include expected outcome: `test_function_name_raises_value_error`

### 2. Test Structure
```python
def test_example(self):
    """Clear description of what is being tested"""
    # Arrange: Set up test data
    input_value = "test"
    
    # Act: Execute the function
    result = function_under_test(input_value)
    
    # Assert: Verify the result
    assert result == expected_value
```

### 3. Test Documentation
- Always include a docstring explaining the test
- Document why this test case is important
- Mention any special setup or teardown

### 4. Assertions
- Use specific assertions: `assert result == expected` not `assert result`
- Include helpful messages: `assert result == expected, f"Expected {expected}, got {result}"`
- Test multiple aspects when relevant

### 5. Test Independence
- Each test should be independent
- Don't rely on test execution order
- Use fixtures for shared setup

## Examples

### Example 1: Simple Function

```python
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b
```

Generated tests:
- `test_add_with_valid_inputs`: Test with typical values (42, 10)
- `test_add_with_zero_a`: Test with zero
- `test_add_with_negative_a`: Test with negative values
- `test_add_with_max_int_a`: Test with maximum integer
- `test_add_with_wrong_type_a`: Test with wrong type

### Example 2: Function with Exceptions

```python
def divide(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result of division
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

Generated tests:
- `test_divide_with_valid_inputs`: Normal division
- `test_divide_raises_value_error_with_zero_b`: Division by zero
- `test_divide_with_negative_b`: Negative denominator
- `test_divide_with_wrong_type_b`: Wrong type

### Example 3: Async Function

```python
async def fetch_data(url: str) -> dict:
    """Fetch data from URL.
    
    Args:
        url: URL to fetch from
        
    Returns:
        Fetched data as dictionary
        
    Raises:
        ValueError: If url is empty
    """
    if not url:
        raise ValueError("URL cannot be empty")
    # ... implementation
```

Generated tests:
- `test_fetch_data_with_valid_inputs`: Normal fetch
- `test_fetch_data_raises_value_error_with_empty_url`: Empty URL
- `test_fetch_data_with_special_chars_in_url`: Special characters
- `test_fetch_data_with_wrong_type_url`: Wrong type

## Integration with pytest

The generated tests are compatible with pytest and can be used directly:

```python
# Save generated tests to a file
with open("test_generated.py", "w") as f:
    f.write(generate_tests_for_function(my_function))

# Run the tests
# pytest test_generated.py
```

## Customization

You can customize test generation by:

1. **Extending TestGenerator**: Add custom test case generators
2. **Modifying test templates**: Change how tests are formatted
3. **Adding domain-specific tests**: Add tests specific to your domain

## Tips

1. **Review generated tests**: Always review and refine generated tests
2. **Add manual tests**: Some scenarios may need manual test cases
3. **Keep tests updated**: Update tests when function signatures change
4. **Use fixtures**: Leverage pytest fixtures for complex setup
5. **Test edge cases**: Don't forget to test edge cases manually

## Common Patterns

### Testing with Mocks
```python
from unittest.mock import Mock, patch

def test_function_with_mock(self):
    """Test function with mocked dependency"""
    with patch('module.dependency') as mock_dep:
        mock_dep.return_value = "mocked"
        result = function_under_test()
        assert result == "expected"
```

### Testing Async Functions
```python
@pytest.mark.asyncio
async def test_async_function(self):
    """Test async function"""
    result = await async_function_under_test()
    assert result == expected
```

### Testing with Fixtures
```python
def test_function_with_fixture(self, temp_dir):
    """Test function using fixture"""
    result = function_under_test(temp_dir)
    assert result is not None
```

## Conclusion

The test generation utilities help create comprehensive test suites, but they should be used as a starting point. Always review, refine, and add manual test cases as needed to ensure complete coverage of your code.


