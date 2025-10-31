# HeyGen AI Testing Guide

## Overview

This guide provides comprehensive information about testing the HeyGen AI system, including how to run tests, write new tests, and understand the test structure.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── pytest.ini                 # Pytest configuration file
├── test_basic_imports.py       # Basic import validation tests
├── test_core_structures.py     # Core data structure tests
├── test_enterprise_features.py # Enterprise features tests
├── test_lifecycle_management.py # Service lifecycle tests
├── test_dependency_manager.py  # Dependency management tests
├── test_config_manager.py      # Configuration management tests
├── test_health_monitor.py      # Health monitoring tests
├── test_advanced_integration.py # Advanced integration tests
├── test_enhanced_system.py     # Enhanced system tests
├── test_performance_benchmarks.py # Performance tests
├── test_simple.py              # Simple validation tests
├── integration/                # Integration test suite
│   ├── test_e2e_*.py          # End-to-end tests
│   └── ...
├── unit/                       # Unit test suite
│   ├── domain/                # Domain layer tests
│   ├── application/           # Application layer tests
│   ├── infrastructure/        # Infrastructure layer tests
│   └── ...
└── performance/               # Performance test suite
    └── ...
```

## Running Tests

### Quick Start

```bash
# Navigate to the HeyGen AI directory
cd agents/backend/onyx/server/features/heygen_ai

# Run all tests
python run_tests.py

# Or use pytest directly
python -m pytest tests/ -v
```

### Test Categories

#### Unit Tests
```bash
# Run only unit tests
python -m pytest tests/ -m unit -v

# Run specific unit test files
python -m pytest tests/test_core_structures.py -v
python -m pytest tests/test_enterprise_features.py -v
```

#### Integration Tests
```bash
# Run integration tests
python -m pytest tests/ -m integration -v

# Run specific integration tests
python -m pytest tests/test_advanced_integration.py -v
```

#### Performance Tests
```bash
# Run performance tests
python -m pytest tests/ -m performance -v

# Run with benchmarking
python -m pytest tests/test_performance_benchmarks.py --benchmark-only
```

### Test Options

#### Verbose Output
```bash
python -m pytest tests/ -v
```

#### Stop on First Failure
```bash
python -m pytest tests/ -x
```

#### Run Specific Test
```bash
python -m pytest tests/test_enterprise_features.py::TestEnterpriseFeatures::test_create_user -v
```

#### Run Tests with Coverage
```bash
python -m pytest tests/ --cov=core --cov-report=html
```

## Test Configuration

### Pytest Configuration

The `pytest.ini` file contains the main test configuration:

- **Test Discovery**: Automatically finds test files
- **Markers**: Categorizes tests (unit, integration, performance, etc.)
- **Async Support**: Automatic async test handling
- **Timeouts**: Prevents hanging tests
- **Warnings**: Filters out unnecessary warnings

### Test Markers

Tests are categorized using pytest markers:

```python
@pytest.mark.unit
def test_basic_functionality():
    """Unit test for basic functionality"""
    pass

@pytest.mark.integration
def test_component_integration():
    """Integration test for component interaction"""
    pass

@pytest.mark.performance
def test_performance_benchmark():
    """Performance benchmark test"""
    pass

@pytest.mark.slow
def test_long_running_operation():
    """Test that takes more than 5 seconds"""
    pass
```

## Writing Tests

### Test Structure

#### Basic Test
```python
def test_basic_functionality():
    """Test basic functionality"""
    # Arrange
    expected = 4
    
    # Act
    result = 2 + 2
    
    # Assert
    assert result == expected
```

#### Async Test
```python
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality"""
    # Arrange
    service = EnterpriseFeatures()
    await service.initialize()
    
    try:
        # Act
        result = await service.create_user(
            username="test",
            email="test@example.com",
            full_name="Test User"
        )
        
        # Assert
        assert result is not None
    finally:
        await service.shutdown()
```

#### Test with Fixtures
```python
@pytest.fixture
async def enterprise_features():
    """Enterprise features fixture"""
    features = EnterpriseFeatures()
    await features.initialize()
    yield features
    await features.shutdown()

@pytest.mark.asyncio
async def test_with_fixture(enterprise_features):
    """Test using fixture"""
    user_id = await enterprise_features.create_user(
        username="test",
        email="test@example.com",
        full_name="Test User"
    )
    assert user_id is not None
```

#### Parametrized Test
```python
@pytest.mark.parametrize("username,email,expected", [
    ("user1", "user1@example.com", True),
    ("user2", "user2@example.com", True),
    ("", "invalid@example.com", False),
])
async def test_user_creation(enterprise_features, username, email, expected):
    """Test user creation with different parameters"""
    if expected:
        user_id = await enterprise_features.create_user(
            username=username,
            email=email,
            full_name="Test User"
        )
        assert user_id is not None
    else:
        with pytest.raises(ValueError):
            await enterprise_features.create_user(
                username=username,
                email=email,
                full_name="Test User"
            )
```

### Mocking and Stubbing

#### Using pytest-mock
```python
def test_with_mock(mocker):
    """Test with mocked dependencies"""
    # Mock external service
    mock_service = mocker.patch('core.external_api_integration.ExternalAPIManager')
    mock_service.return_value.get_data.return_value = {"status": "success"}
    
    # Test your code
    result = your_function()
    assert result["status"] == "success"
```

#### Using AsyncMock
```python
@pytest.mark.asyncio
async def test_async_mock():
    """Test with async mock"""
    mock_service = AsyncMock()
    mock_service.get_data.return_value = {"data": "test"}
    
    result = await your_async_function(mock_service)
    assert result["data"] == "test"
```

## Test Data Management

### Test Fixtures

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "role": "user"
    }

@pytest.fixture
async def enterprise_features():
    """Enterprise features instance for testing"""
    features = EnterpriseFeatures()
    await features.initialize()
    yield features
    await features.shutdown()
```

### Test Data Factories

Use factory-boy for generating test data:

```python
import factory

class UserFactory(factory.Factory):
    class Meta:
        model = User
    
    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    full_name = factory.Faker('name')
    role = "user"

# Usage in tests
def test_user_factory():
    user = UserFactory()
    assert user.username.startswith("user")
    assert "@example.com" in user.email
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=core --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m pytest tests/ -v
```

#### Async Test Issues
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Check async mode in pytest.ini
asyncio_mode = auto
```

#### Slow Tests
```bash
# Run only fast tests
python -m pytest tests/ -m "not slow" -v

# Run with timeout
python -m pytest tests/ --timeout=60
```

### Debug Mode

```bash
# Run with debug output
python -m pytest tests/ -v -s --tb=long

# Run single test with debug
python -m pytest tests/test_enterprise_features.py::TestEnterpriseFeatures::test_create_user -v -s
```

## Best Practices

### Test Organization
- Group related tests in classes
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Keep tests independent and isolated

### Test Data
- Use factories for generating test data
- Clean up test data after tests
- Use realistic test data
- Avoid hardcoded values

### Performance
- Mark slow tests appropriately
- Use fixtures for expensive setup
- Mock external dependencies
- Run performance tests separately

### Maintenance
- Keep tests up to date with code changes
- Remove obsolete tests
- Refactor tests when code changes
- Document complex test scenarios

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Factory Boy Documentation](https://factoryboy.readthedocs.io/)
- [Testing Best Practices](https://docs.python.org/3/library/unittest.html)





