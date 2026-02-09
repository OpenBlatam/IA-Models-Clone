# Blaze AI Test Suite

This directory contains comprehensive tests for the Blaze AI system, focusing on the plugin system, caching mechanisms, and core engine functionality.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_plugins.py          # Plugin system tests
├── test_llm_engine_cache.py # LLM engine cache tests
├── requirements-test.txt     # Testing dependencies
└── README.md               # This file
```

## Test Categories

### Unit Tests
- **Plugin System**: Tests for `PluginConfig`, `PluginMetadata`, `PluginInfo`, `PluginLoader`, and `PluginManager`
- **Cache System**: Tests for LLM engine caching functionality
- **Core Components**: Tests for base engine classes and protocols

### Integration Tests
- **Plugin Loading**: End-to-end tests for plugin discovery and loading
- **Cache Persistence**: Tests for cache persistence across system restarts
- **Engine Management**: Tests for engine factory and management systems

## Running Tests

### Prerequisites

Install the required testing dependencies:

```bash
pip install -r tests/requirements-test.txt
```

### Using the Test Runner

The easiest way to run tests is using the provided test runner:

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run tests with coverage
python run_tests.py --coverage

# Run tests in parallel
python run_tests.py --parallel

# Run a specific test file
python run_tests.py --file tests/test_plugins.py
```

### Using Pytest Directly

You can also run tests directly with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_plugins.py -v

# Run tests with coverage
pytest tests/ --cov=engines --cov-report=html

# Run tests in parallel
pytest tests/ -n auto

# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration
```

## Test Configuration

### Pytest Configuration

The `conftest.py` file provides:
- Automatic logging mocking to reduce test output noise
- Common fixtures for test data and mock objects
- Automatic test categorization (unit/integration)
- Temporary directory management

### Test Fixtures

Common fixtures available in `conftest.py`:

- `temp_test_dir`: Creates a temporary directory for testing
- `sample_plugin_config`: Sample plugin configuration
- `sample_plugin_metadata`: Sample plugin metadata
- `mock_engine_class`: Mock engine class for testing
- `sample_plugin_file`: Sample Python plugin file
- `sample_plugin_json`: Sample plugin.json file

## Writing Tests

### Test Naming Convention

- Test files should be named `test_*.py`
- Test classes should be named `Test*`
- Test methods should be named `test_*`

### Example Test Structure

```python
class TestMyFeature(unittest.TestCase):
    """Test cases for MyFeature."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up test fixtures."""
        pass
    
    def test_feature_behavior(self):
        """Test that the feature behaves correctly."""
        # Arrange
        # Act
        # Assert
        pass
```

### Using Fixtures

```python
def test_with_fixtures(sample_plugin_config, temp_test_dir):
    """Test using pytest fixtures."""
    assert sample_plugin_config.plugin_directories == ["test_plugins"]
    assert temp_test_dir.exists()
```

## Test Coverage

To generate coverage reports:

```bash
# Generate HTML coverage report
pytest tests/ --cov=engines --cov-report=html

# Generate terminal coverage report
pytest tests/ --cov=engines --cov-report=term-missing

# Generate both
pytest tests/ --cov=engines --cov-report=html --cov-report=term-missing
```

Coverage reports will be generated in the `htmlcov/` directory.

## Continuous Integration

The test suite is designed to work with CI/CD systems:

- Tests can run in parallel for faster execution
- Coverage reporting is available
- Exit codes properly indicate test success/failure
- Tests are isolated and don't depend on external services

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running tests from the correct directory
2. **Missing Dependencies**: Install test requirements with `pip install -r tests/requirements-test.txt`
3. **Permission Errors**: Ensure you have write access to create temporary directories
4. **Test Failures**: Check that the code being tested is compatible with the test expectations

### Debug Mode

To run tests with more verbose output:

```bash
pytest tests/ -v -s --tb=long
```

The `-s` flag shows print statements, and `--tb=long` shows full tracebacks.

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Ensure tests are isolated and don't depend on external state
4. Add appropriate test markers (unit/integration)
5. Update this README if adding new test categories

## Test Results

After running tests, you'll see:
- ✅ Passed tests
- ❌ Failed tests
- 📊 Coverage information (if enabled)
- ⏱️ Execution time
- 📝 Detailed error information for failures
