# Testing Guide - Addiction Recovery AI

## ✅ Test Structure

### Test Organization

Tests are organized in the `tests/` directory:

```
tests/
├── conftest.py                  # ✅ Pytest configuration and fixtures
├── pytest.ini                   # ✅ Pytest configuration
├── README.md                    # ✅ Test documentation
├── run_tests.sh                 # ✅ Test runner (Linux/Mac)
├── run_tests.bat                # ✅ Test runner (Windows)
├── test_api_endpoints.py        # ✅ API endpoint tests
├── test_api_error_handling.py  # ✅ Error handling tests
├── test_performance.py          # ✅ Performance tests
├── test_integration_complete.py # ✅ Complete integration tests
├── test_integration.py          # ✅ Integration tests
├── test_services.py             # ✅ Service tests
├── test_middleware.py           # ✅ Middleware tests
├── test_schemas.py              # ✅ Schema tests
├── test_models.py               # ✅ Model tests
├── test_core_classes.py         # ✅ Core class tests
├── test_core_classes_enhanced.py # ✅ Enhanced core class tests
├── test_base_models.py          # ✅ Base model tests
├── test_utilities.py            # ✅ Utility tests
├── test_utilities_enhanced.py   # ✅ Enhanced utility tests
├── test_validation.py           # ✅ Validation tests
├── test_validators.py           # ✅ Validator tests
├── test_helper_functions.py     # ✅ Helper function tests
├── test_file_storage.py         # ✅ File storage tests
├── test_transformers.py         # ✅ Transformer tests
└── test_advanced_cases.py       # ✅ Advanced test cases
```

## 🧪 Running Tests

### Using Test Runners

**Linux/Mac:**
```bash
./tests/run_tests.sh
```

**Windows:**
```cmd
tests\run_tests.bat
```

### Using Pytest Directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_api_endpoints.py

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_api_endpoints.py::test_create_assessment
```

## 📋 Test Categories

### API Tests
- `test_api_endpoints.py` - API endpoint functionality
- `test_api_error_handling.py` - Error handling and responses

### Integration Tests
- `test_integration_complete.py` - Complete integration scenarios
- `test_integration.py` - Basic integration tests

### Component Tests
- `test_services.py` - Service layer tests
- `test_middleware.py` - Middleware tests
- `test_schemas.py` - Schema validation tests
- `test_models.py` - ML model tests
- `test_core_classes.py` - Core component tests
- `test_base_models.py` - Base model tests

### Utility Tests
- `test_utilities.py` - Utility function tests
- `test_validation.py` - Validation tests
- `test_validators.py` - Validator tests
- `test_helper_functions.py` - Helper function tests
- `test_file_storage.py` - File storage tests
- `test_transformers.py` - Transformer tests

### Performance Tests
- `test_performance.py` - Performance and benchmarking

### Advanced Tests
- `test_advanced_cases.py` - Advanced test scenarios
- `test_core_classes_enhanced.py` - Enhanced core tests
- `test_utilities_enhanced.py` - Enhanced utility tests

## 📝 Test Configuration

### `conftest.py`
- Contains shared fixtures
- Pytest configuration
- Common test utilities

### `pytest.ini`
- Pytest settings
- Test discovery patterns
- Coverage configuration

## 🎯 Writing Tests

### Example Test Structure
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_assessment():
    """Test assessment creation"""
    response = client.post(
        "/api/assessment/create",
        json={"user_id": "test", "substance_type": "alcohol"}
    )
    assert response.status_code == 201
```

### Using Fixtures
```python
@pytest.fixture
def mock_analyzer():
    """Mock addiction analyzer"""
    # Setup
    yield MockAnalyzer()
    # Teardown
```

## 📚 Additional Resources

- See `tests/README.md` for detailed test documentation
- See `tests/TEST_SUITE_SUMMARY.md` for test suite summary
- See `tests/IMPROVEMENTS.md` for test improvements
- See `tests/ADDITIONAL_TESTS.md` for additional test cases






