# Onyx AI Video System - Test Suite

This directory contains comprehensive tests for the Onyx AI Video System, including unit tests, integration tests, and system tests.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests for individual components
│   ├── test_core.py         # Core module tests
│   ├── test_config.py       # Configuration tests
│   └── test_utils.py        # Utility module tests
├── integration/             # Integration tests
│   ├── test_api.py          # API integration tests
│   └── test_workflows.py    # Workflow integration tests
├── system/                  # System tests
│   └── test_system.py       # End-to-end system tests
├── test_runner.py           # Comprehensive test runner
├── run_tests.py             # Simple test runner
├── requirements_test.txt    # Test dependencies
└── README_TESTS.md         # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Single functions, classes, or modules
- **Dependencies**: Mocked external dependencies
- **Speed**: Fast execution (< 1 second per test)
- **Coverage**: High code coverage for individual components

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and workflows
- **Scope**: Multiple components working together
- **Dependencies**: Some real dependencies, some mocked
- **Speed**: Medium execution (1-10 seconds per test)
- **Coverage**: Integration scenarios and workflows

### System Tests (`tests/system/`)
- **Purpose**: Test complete system end-to-end
- **Scope**: Full system functionality
- **Dependencies**: Real system components
- **Speed**: Slow execution (10+ seconds per test)
- **Coverage**: End-to-end scenarios and performance

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install -r tests/requirements_test.txt
```

2. Ensure the main system is installed:
```bash
pip install -e .
```

### Quick Test Run

Run all tests with the simple test runner:
```bash
python tests/run_tests.py
```

### Comprehensive Test Run

Use the full test runner for detailed reports:
```bash
python tests/test_runner.py
```

### Running Specific Test Categories

```bash
# Unit tests only
python tests/test_runner.py --categories unit

# Integration tests only
python tests/test_runner.py --categories integration

# System tests only
python tests/test_runner.py --categories system

# Multiple categories
python tests/test_runner.py --categories unit integration
```

### Running with Pytest Directly

```bash
# All tests
pytest tests/

# Specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/system/

# With coverage
pytest tests/ --cov=onyx_ai_video --cov-report=html

# With verbose output
pytest tests/ -v

# Parallel execution
pytest tests/ -n auto
```

## Test Configuration

### Environment Variables

Set these environment variables for testing:

```bash
# Test environment
export AI_VIDEO_ENVIRONMENT=testing
export AI_VIDEO_DEBUG=true

# Disable Onyx integration for faster tests
export AI_VIDEO_USE_ONYX=false

# Test output directories
export AI_VIDEO_TEST_OUTPUT_DIR=/tmp/ai_video_test_outputs
export AI_VIDEO_TEST_LOGS_DIR=/tmp/ai_video_test_logs
```

### Test Configuration File

Create `test_config.json` in the project root:

```json
{
  "test_outputs_dir": "/tmp/ai_video_test_outputs",
  "test_logs_dir": "/tmp/ai_video_test_logs",
  "test_reports_dir": "./test_reports",
  "coverage_enabled": true,
  "parallel_tests": true,
  "verbose_output": true,
  "timeout": {
    "unit": 300,
    "integration": 600,
    "system": 1200
  }
}
```

## Test Fixtures

The test suite provides several fixtures in `conftest.py`:

### Core Fixtures
- `temp_dir`: Temporary directory for test files
- `sample_config`: Sample configuration for testing
- `config_manager`: Test configuration manager
- `test_config`: Test configuration object

### Mock Fixtures
- `mock_logger`: Mock logger for testing
- `mock_performance_monitor`: Mock performance monitor
- `mock_security_manager`: Mock security manager
- `mock_onyx_integration`: Mock Onyx integration
- `mock_video_workflow`: Mock video workflow
- `mock_plugin_manager`: Mock plugin manager

### Data Fixtures
- `sample_video_request`: Sample video request
- `sample_video_response`: Sample video response
- `sample_plugin_config`: Sample plugin configuration

### System Fixtures
- `real_system`: Real system instance for integration tests
- `mock_system`: Mock system for testing

## Writing Tests

### Unit Test Example

```python
import pytest
from onyx_ai_video.core.models import VideoRequest, VideoQuality

@pytest.mark.unit
def test_video_request_creation():
    """Test VideoRequest creation with valid data."""
    request = VideoRequest(
        input_text="Test video",
        user_id="test_user",
        quality=VideoQuality.HIGH,
        duration=30,
        output_format=VideoFormat.MP4
    )
    
    assert request.input_text == "Test video"
    assert request.user_id == "test_user"
    assert request.quality == VideoQuality.HIGH
    assert request.duration == 30
    assert request.request_id is not None
```

### Integration Test Example

```python
import pytest
from onyx_ai_video.api.main import OnyxAIVideoSystem

@pytest.mark.integration
async def test_video_generation_workflow(real_system, sample_video_request):
    """Test complete video generation workflow."""
    await real_system.initialize()
    
    with patch.object(real_system.video_workflow, 'generate_video') as mock_generate:
        mock_generate.return_value = VideoResponse(
            request_id=sample_video_request.request_id,
            status="completed",
            output_url="http://example.com/video.mp4"
        )
        
        response = await real_system.generate_video(sample_video_request)
        
        assert response.status == "completed"
        assert response.request_id == sample_video_request.request_id
```

### System Test Example

```python
import pytest
from onyx_ai_video.api.main import OnyxAIVideoSystem

@pytest.mark.system
async def test_end_to_end_video_generation(temp_dir):
    """Test complete end-to-end video generation."""
    # Setup system
    system = OnyxAIVideoSystem(str(temp_dir / "config.yaml"))
    await system.initialize()
    
    # Create request
    request = VideoRequest(
        input_text="Create a test video",
        user_id="system_test_user",
        quality=VideoQuality.LOW,
        duration=10
    )
    
    # Generate video
    response = await system.generate_video(request)
    
    # Verify results
    assert response.status == "completed"
    assert Path(response.output_path).exists()
    
    # Check system metrics
    metrics = await system.get_metrics()
    assert metrics["request_metrics"]["total_requests"] >= 1
```

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit              # Unit tests
@pytest.mark.integration       # Integration tests
@pytest.mark.system           # System tests
@pytest.mark.performance      # Performance tests
@pytest.mark.security         # Security tests
@pytest.mark.slow             # Slow running tests
```

Run tests by marker:
```bash
pytest -m unit
pytest -m "not slow"
pytest -m "unit or integration"
```

## Coverage Reports

### HTML Coverage Report
```bash
pytest --cov=onyx_ai_video --cov-report=html
# Open: htmlcov/index.html
```

### JSON Coverage Report
```bash
pytest --cov=onyx_ai_video --cov-report=json
# Generates: coverage.json
```

### Terminal Coverage Report
```bash
pytest --cov=onyx_ai_video --cov-report=term-missing
```

## Performance Testing

### Benchmark Tests
```python
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

@pytest.mark.performance
def test_video_generation_performance(benchmark: BenchmarkFixture):
    """Benchmark video generation performance."""
    def generate_video():
        # Video generation code
        pass
    
    result = benchmark(generate_video)
    assert result.stats.mean < 1.0  # Should complete in under 1 second
```

### Load Testing
```python
@pytest.mark.performance
async def test_concurrent_requests(real_system):
    """Test system under concurrent load."""
    requests = [create_test_request() for _ in range(100)]
    
    start_time = time.time()
    responses = await asyncio.gather(*[
        real_system.generate_video(req) for req in requests
    ])
    end_time = time.time()
    
    assert end_time - start_time < 60  # Complete within 60 seconds
    assert all(r.status == "completed" for r in responses)
```

## Security Testing

### Input Validation Tests
```python
@pytest.mark.security
def test_malicious_input_handling(real_system):
    """Test handling of malicious input."""
    malicious_inputs = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "javascript:alert('xss')"
    ]
    
    for malicious_input in malicious_inputs:
        request = VideoRequest(
            input_text=malicious_input,
            user_id="security_test_user"
        )
        
        # Should handle malicious input safely
        response = await real_system.generate_video(request)
        assert response.status in ["completed", "error"]
```

### Access Control Tests
```python
@pytest.mark.security
async def test_access_control(real_system):
    """Test access control mechanisms."""
    # Test unauthorized access
    with pytest.raises(SecurityError):
        await real_system.generate_video(
            VideoRequest(input_text="test", user_id="unauthorized_user")
        )
    
    # Test rate limiting
    for _ in range(10):
        await real_system.generate_video(
            VideoRequest(input_text="test", user_id="rate_limit_user")
        )
    
    # 11th request should be rate limited
    with pytest.raises(RateLimitError):
        await real_system.generate_video(
            VideoRequest(input_text="test", user_id="rate_limit_user")
        )
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements_test.txt
      - name: Run tests
        run: python tests/run_tests.py
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Local CI Setup
```bash
# Install pre-commit hooks
pre-commit install

# Run all checks
pre-commit run --all-files

# Run tests with coverage
python tests/test_runner.py --categories unit integration --no-coverage
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure the package is installed in development mode
   pip install -e .
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Test Timeouts**
   ```bash
   # Increase timeout for slow tests
   pytest --timeout=300 tests/system/
   
   # Or set in pytest.ini
   [pytest]
   timeout = 300
   ```

3. **Memory Issues**
   ```bash
   # Run tests with memory profiling
   pytest --memray tests/
   
   # Or limit memory usage
   pytest --maxfail=10 tests/
   ```

4. **Coverage Issues**
   ```bash
   # Check coverage configuration
   coverage run -m pytest tests/
   coverage report
   coverage html
   ```

### Debug Mode

Run tests in debug mode:
```bash
# Enable debug logging
export AI_VIDEO_DEBUG=true

# Run with verbose output
pytest -v -s tests/

# Run specific test with debugger
python -m pdb -m pytest tests/unit/test_core.py::test_specific_function
```

## Test Reports

After running tests, check the following reports:

1. **HTML Reports**: `test_reports/report_*.html`
2. **Coverage Reports**: `test_reports/coverage_*/index.html`
3. **JUnit Reports**: `test_reports/junit_*.xml`
4. **JSON Reports**: `test_reports/coverage_*.json`

### Report Interpretation

- **Success Rate**: Percentage of tests that passed
- **Coverage**: Percentage of code covered by tests
- **Performance**: Test execution time and benchmarks
- **Security**: Security test results and vulnerabilities
- **Recommendations**: Suggestions for improving test coverage

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use appropriate test markers
3. Add comprehensive docstrings
4. Include both positive and negative test cases
5. Mock external dependencies appropriately
6. Ensure tests are fast and reliable
7. Update this README if adding new test categories

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Descriptive Names**: Use clear, descriptive test names
3. **Arrange-Act-Assert**: Structure tests with clear sections
4. **Mock Appropriately**: Mock external dependencies, not internal logic
5. **Test Edge Cases**: Include boundary conditions and error cases
6. **Performance**: Keep tests fast and efficient
7. **Maintenance**: Keep tests up to date with code changes 