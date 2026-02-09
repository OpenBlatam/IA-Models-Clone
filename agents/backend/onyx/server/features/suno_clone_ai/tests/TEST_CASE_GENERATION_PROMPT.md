# Test Case Generation Prompt

## Prompt para Generación de Casos de Prueba

You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring.

## Instructions

When given a function signature and docstring, generate comprehensive test cases that cover:

1. **Happy Path**: Normal successful execution with valid inputs
2. **Edge Cases**: Boundary values, extreme but valid inputs
3. **Error Handling**: Invalid inputs, missing parameters, type errors
4. **Boundary Conditions**: Min/max values, empty inputs, null values
5. **Type Validation**: Wrong types, None values where not allowed
6. **Integration Scenarios**: Interactions with dependencies

## Output Format

For each test case, provide:

- **Test Name**: Descriptive name following `test_<function>_<scenario>` pattern
- **Test Type**: One of: `happy_path`, `edge_case`, `error_handling`, `boundary`, `null_empty`, `type_validation`
- **Description**: Clear explanation of what is being tested
- **Parameters**: Input values for the test
- **Expected Result**: What should be returned (if applicable)
- **Expected Exception**: What exception should be raised (if applicable)
- **Assertions**: List of assertions to verify the result

## Example

### Input Function

```python
def generate_song_id() -> str:
    """Genera un ID único para una canción"""
    return str(uuid.uuid4())
```

### Generated Test Cases

```python
# Happy Path
def test_generate_song_id_happy_path_basic():
    """Test básico de generate_song_id con parámetros válidos"""
    result = generate_song_id()
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

# Edge Case
def test_generate_song_id_unique():
    """Test que genera IDs únicos"""
    ids = {generate_song_id() for _ in range(100)}
    assert len(ids) == 100

# Type Validation
def test_generate_song_id_format():
    """Test que el formato es válido UUID"""
    song_id = generate_song_id()
    uuid.UUID(song_id)  # Should not raise
```

## Guidelines

1. **Be Comprehensive**: Cover all code paths and edge cases
2. **Be Specific**: Use concrete values, not generic placeholders
3. **Be Independent**: Each test should be able to run alone
4. **Be Clear**: Test names and descriptions should be self-explanatory
5. **Be Realistic**: Use realistic test data that reflects actual usage
6. **Be Diverse**: Vary test inputs to catch different scenarios
7. **Be Intuitive**: Tests should be easy to understand and maintain

## Function Analysis

Before generating tests, analyze:

- **Parameters**: Types, defaults, constraints (min/max, length, etc.)
- **Return Type**: What the function returns
- **Docstring**: Extract requirements, examples, edge cases mentioned
- **Dependencies**: What external services/functions it uses
- **Side Effects**: What it modifies (files, state, etc.)
- **Error Conditions**: When it might fail

## Test Structure

```python
class TestFunctionName:
    """Tests para function_name"""
    
    @pytest.mark.happy_path
    def test_function_name_happy_path_basic(self):
        """Test básico exitoso"""
        # Arrange
        # Act
        # Assert
    
    @pytest.mark.error_handling
    def test_function_name_error_invalid_input(self):
        """Test con input inválido"""
        # Arrange
        # Act & Assert
        with pytest.raises(ValueError):
            function_name(invalid_input)
```

## Special Considerations

### Async Functions
- Use `@pytest.mark.asyncio`
- Use `await` when calling the function
- Test cancellation scenarios

### Functions with Dependencies
- Mock external dependencies
- Test with different dependency behaviors
- Test dependency failures

### Functions with Side Effects
- Test file creation/deletion
- Test state changes
- Test cleanup

### Functions with Validation
- Test all validation rules
- Test boundary values
- Test invalid combinations

## Quality Checklist

- [ ] All happy paths covered
- [ ] All error cases covered
- [ ] Boundary values tested
- [ ] Type validation tested
- [ ] Edge cases considered
- [ ] Dependencies mocked appropriately
- [ ] Assertions are specific and meaningful
- [ ] Test names are descriptive
- [ ] Tests are independent
- [ ] Tests use appropriate fixtures

## Usage with Test Case Generator

The `test_case_generator.py` module implements this prompt as code:

```python
from tests.test_case_generator import generate_tests_for_function
from api.helpers import generate_song_id

# Generate test cases and code
test_cases, code = generate_tests_for_function(
    generate_song_id,
    num_cases=10,
    output_file="test_generated.py"
)
```

This will:
1. Analyze the function signature and docstring
2. Generate diverse test cases
3. Output Python test code ready to use

