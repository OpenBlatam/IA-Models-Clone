# Test Suite for Addiction Recovery AI

This directory contains comprehensive unit tests for the Addiction Recovery AI module.

## Test Structure

### Core Tests
- **test_core_classes.py**: Basic tests for core classes
- **test_core_classes_enhanced.py**: Enhanced tests with fixtures and parametrization
- **test_base_models.py**: Tests for base model classes (BaseModel, BasePredictor, BaseGenerator, BaseAnalyzer)
- **test_models.py**: Tests for ML models (sentiment analyzer, progress predictor, inference engines)

### Utility Tests
- **test_utilities.py**: Basic tests for utility functions
- **test_utilities_enhanced.py**: Enhanced tests with parametrization and fixtures

### Integration & Validation Tests
- **test_integration.py**: Integration tests and complete workflows
- **test_validation.py**: Input validation and error handling tests

### Advanced & Helper Tests
- **test_advanced_cases.py**: Complex scenarios, edge cases, and performance tests
- **test_helper_functions.py**: Tests for internal helper methods and utilities

### Configuration
- **conftest.py**: Pytest fixtures and configuration shared across all tests

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_core_classes.py -v
```

### Run specific test class
```bash
pytest tests/test_core_classes.py::TestAddictionAnalyzer -v
```

### Run specific test
```bash
pytest tests/test_core_classes.py::TestAddictionAnalyzer::test_assess_addiction_minimal_data -v
```

### Run with coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Unit Tests
- Test individual functions and methods in isolation
- Mock external dependencies
- Test edge cases and boundary conditions

### Integration Tests
- Test complete workflows from assessment to tracking
- Test component interactions
- Test data consistency across components

### Validation Tests
- Test input validation
- Test error handling and recovery
- Test type coercion and conversion

## Test Coverage Goals

- **Core Classes**: 90%+ coverage
- **Utility Functions**: 85%+ coverage
- **Base Models**: 80%+ coverage
- **Integration**: 70%+ coverage
- **Helper Functions**: 85%+ coverage
- **Advanced Cases**: 75%+ coverage

## Test Statistics

- **Total Test Files**: 10+
- **Total Test Cases**: 300+
- **Fixtures Available**: 20+
- **Parametrized Tests**: 50+

### Test Categories Breakdown

1. **Core Classes**: ~100 tests
   - AddictionAnalyzer: 25+ tests
   - RecoveryPlanner: 20+ tests
   - ProgressTracker: 30+ tests
   - RelapsePrevention: 25+ tests

2. **Base Models**: ~40 tests
   - BaseModel: 10+ tests
   - BasePredictor: 8+ tests
   - BaseGenerator: 6+ tests
   - BaseAnalyzer: 6+ tests
   - Integration: 10+ tests

3. **Utilities**: ~60 tests
   - Data utilities: 30+ tests
   - Helper functions: 30+ tests

4. **Advanced Cases**: ~50 tests
   - Complex scenarios: 15+ tests
   - Performance tests: 10+ tests
   - Data integrity: 10+ tests
   - Boundary cases: 15+ tests

5. **Helper Functions**: ~40 tests
   - Internal methods: 40+ tests

6. **Integration & Validation**: ~50 tests
   - Integration workflows: 20+ tests
   - Validation: 20+ tests
   - Error handling: 10+ tests

## Writing New Tests

When adding new tests:

1. Follow the existing test structure
2. Use descriptive test names
3. Test both happy paths and edge cases
4. Include error handling tests
5. Add docstrings explaining what is being tested
6. **Use fixtures from conftest.py for common setup**
7. **Use @pytest.mark.parametrize for testing multiple scenarios**
8. Mock external dependencies

### Example Test Structure

```python
def test_function_name_scenario(self, fixture_name):
    """Test description of what is being tested"""
    # Arrange - use fixtures
    setup_data = fixture_name
    
    # Act
    result = function_under_test(setup_data)
    
    # Assert
    assert result is not None
    assert result["expected_key"] == expected_value

@pytest.mark.parametrize("input_value,expected_output", [
    (1, "one"),
    (2, "two"),
    (3, "three"),
])
def test_parametrized_example(self, input_value, expected_output):
    """Test multiple scenarios with parametrization"""
    result = function_under_test(input_value)
    assert result == expected_output
```

### Using Fixtures

Fixtures are defined in `conftest.py` and can be used in any test:

```python
def test_with_fixture(self, addiction_analyzer, minimal_assessment_data):
    """Test using fixtures"""
    result = addiction_analyzer.assess_addiction(minimal_assessment_data)
    assert result["success"] is True
```

### Available Fixtures

- **Core Classes**: `addiction_analyzer`, `addiction_analyzer_with_ai`, `recovery_planner`, `progress_tracker`, `relapse_prevention`
- **Test Data**: `minimal_assessment_data`, `complete_assessment_data`, `sample_progress_entries`, `current_state_low_risk`, etc.
- **Edge Cases**: `empty_assessment_data`, `invalid_assessment_data`, `malformed_progress_entries`
- **Performance**: `large_progress_entries`, `large_data_list`
- **Integration**: `complete_user_journey`

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Commits to main branch
- Nightly builds

## Dependencies

Test dependencies are listed in `requirements-dev.txt`:
- pytest
- pytest-cov
- pytest-mock
- torch (for model tests)
- numpy (for utility tests)

## Notes

- Some tests may be skipped if optional dependencies are not available
- Model tests require PyTorch and may require GPU for some tests
- Integration tests may take longer to run

