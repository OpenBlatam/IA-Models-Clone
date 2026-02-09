# Quick Start Guide - Advanced Testing System

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
# Install test requirements
pip install -r tests/requirements-test.txt

# Or install specific categories
pip install pytest pytest-asyncio pytest-cov
pip install factory-boy faker hypothesis
pip install locust pytest-benchmark memory-profiler
```

### 2. Run All Tests
```bash
# Run complete test suite
python tests/run_advanced_tests.py

# Run with verbose output
python tests/run_advanced_tests.py --verbose
```

### 3. Run Specific Test Types
```bash
# Unit tests only
python tests/run_advanced_tests.py --test-type unit

# Integration tests
python tests/run_advanced_tests.py --test-type integration

# Load tests
python tests/run_advanced_tests.py --test-type load

# Performance tests
python tests/run_advanced_tests.py --test-type performance
```

## 🧪 Test Examples

### Unit Tests with Factory Boy
```python
def test_post_creation_with_factory():
    """Create test data using Factory Boy."""
    post_data = PostDataFactory()
    
    assert post_data["content"] is not None
    assert post_data["post_type"] in ["announcement", "educational", "update"]
    assert post_data["tone"] in ["professional", "casual", "friendly"]
```

### Property-Based Testing with Hypothesis
```python
@given(st.text(min_size=10, max_size=500))
def test_post_content_validation(content):
    """Test post content with automatically generated data."""
    post_data = {
        "content": content,
        "post_type": "announcement",
        "tone": "professional",
        "target_audience": "professionals",
        "industry": "technology"
    }
    
    if len(content) >= 10:
        post = LinkedInPostCreate(**post_data)
        assert post.content == content
    else:
        with pytest.raises(ValueError):
            LinkedInPostCreate(**post_data)
```

### Load Testing with Locust
```python
# Run Locust load test
locust -f tests/load/test_advanced_load.py --headless --users 10 --spawn-rate 2 --run-time 60s
```

### Performance Benchmarking
```python
def test_post_creation_performance(benchmark):
    """Benchmark post creation performance."""
    def create_post():
        return asyncio.run(use_cases.generate_post(
            content="Test content",
            post_type=PostType.ANNOUNCEMENT,
            tone=PostTone.PROFESSIONAL,
            target_audience="professionals",
            industry="technology"
        ))
    
    result = benchmark(create_post)
    assert result is not None
```

### Memory Profiling
```python
@pytest.mark.asyncio
@profile
async def test_memory_usage():
    """Profile memory usage during operations."""
    for i in range(100):
        post_data = PostDataFactory()
        # Your operations here
```

## 🔧 Common Commands

### Pytest Commands
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=linkedin_posts --cov-report=html

# Run specific test file
pytest tests/unit/test_advanced_unit.py -v

# Run with markers
pytest tests/ -m "not slow" -v

# Run in parallel
pytest tests/ -n auto -v
```

### Load Testing Commands
```bash
# Run Locust web interface
locust -f tests/load/test_advanced_load.py

# Run headless load test
locust -f tests/load/test_advanced_load.py --headless --users 50 --spawn-rate 5 --run-time 2m

# Run with custom parameters
locust -f tests/load/test_advanced_load.py --headless --users 100 --spawn-rate 10 --run-time 5m --html=reports/load_test.html
```

### Performance Testing Commands
```bash
# Run performance benchmarks
pytest tests/load/test_advanced_load.py::TestPerformanceBenchmarking -v --benchmark-only

# Run with specific benchmark
pytest tests/load/test_advanced_load.py::TestPerformanceBenchmarking::test_post_creation_benchmark -v --benchmark-only

# Generate benchmark report
pytest tests/load/test_advanced_load.py::TestPerformanceBenchmarking -v --benchmark-only --benchmark-json=reports/benchmarks.json
```

### Code Quality Commands
```bash
# Run black formatting check
black --check linkedin_posts/

# Run isort import check
isort --check-only linkedin_posts/

# Run flake8 linting
flake8 linkedin_posts/

# Run mypy type checking
mypy linkedin_posts/

# Run security checks
bandit -r linkedin_posts/
```

## 📊 Understanding Results

### Test Reports
After running tests, check the `reports/` directory for:
- `test_results.json` - Detailed test results
- `test_report.html` - HTML test report
- `test_report.md` - Markdown test report
- `coverage/` - Code coverage reports
- `load_tests.html` - Load test results
- `benchmarks.json` - Performance benchmarks

### Coverage Report
```bash
# Generate coverage report
pytest tests/ --cov=linkedin_posts --cov-report=html

# View coverage in browser
open reports/htmlcov/index.html
```

### Performance Metrics
```bash
# View benchmark results
pytest tests/load/test_advanced_load.py::TestPerformanceBenchmarking -v --benchmark-only

# Compare benchmarks
pytest tests/load/test_advanced_load.py::TestPerformanceBenchmarking -v --benchmark-only --benchmark-compare
```

## 🐛 Debugging

### Debug Test Failures
```python
# Add debug prints
def test_debug_example():
    post_data = PostDataFactory()
    print(f"Generated post data: {post_data}")
    
    # Your test logic here
    assert post_data["content"] is not None
```

### Use Debugger
```python
import pdb

def test_with_debugger():
    post_data = PostDataFactory()
    pdb.set_trace()  # Debugger will stop here
    
    # Your test logic here
    assert post_data["content"] is not None
```

### Memory Debugging
```python
from tests.debug.test_advanced_debug import AdvancedDebugger

def test_memory_debugging():
    debugger = AdvancedDebugger()
    
    with debugger.memory_tracking("test_operation"):
        # Your operations here
        post_data = PostDataFactory()
        # More operations...
```

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
name: Advanced Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r tests/requirements-test.txt
      
      - name: Run tests
        run: |
          python tests/run_advanced_tests.py --ci
      
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: test-reports
          path: reports/
```

### GitLab CI Example
```yaml
test:
  stage: test
  image: python:3.9
  script:
    - pip install -r tests/requirements-test.txt
    - python tests/run_advanced_tests.py --ci
  artifacts:
    reports:
      junit: reports/*.xml
    paths:
      - reports/
```

## 📝 Best Practices

### 1. Writing Tests
```python
# Good: Use descriptive names
def test_create_linkedin_post_with_valid_data_should_succeed():
    """Test that creating a post with valid data succeeds."""
    post_data = PostDataFactory()
    result = create_post(post_data)
    assert result.success is True

# Good: Use factories for test data
def test_post_creation():
    post_data = PostDataFactory(
        content="Test content",
        post_type="announcement"
    )
    # Test logic here

# Good: Test edge cases
@given(st.text(min_size=0, max_size=1000))
def test_post_content_length_validation(content):
    """Test post content length validation."""
    if len(content) == 0:
        with pytest.raises(ValueError):
            create_post({"content": content})
    elif len(content) > 500:
        with pytest.raises(ValueError):
            create_post({"content": content})
    else:
        result = create_post({"content": content})
        assert result.success is True
```

### 2. Performance Testing
```python
# Good: Test realistic scenarios
def test_batch_creation_performance(benchmark):
    """Test batch creation with realistic data."""
    batch_data = PostDataFactory.build_batch(10)
    
    def create_batch():
        return asyncio.run(use_cases.batch_create_posts(batch_data))
    
    result = benchmark(create_batch)
    assert len(result) == 10

# Good: Monitor system resources
def test_memory_usage_under_load():
    """Test memory usage during load."""
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss
    
    # Perform operations
    for _ in range(100):
        post_data = PostDataFactory()
        # Your operations here
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

### 3. Load Testing
```python
# Good: Test different scenarios
class LinkedInPostsLoadUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)  # High frequency
    def get_posts(self):
        self.client.get("/linkedin-posts/")
    
    @task(2)  # Medium frequency
    def create_post(self):
        post_data = PostDataFactory()
        self.client.post("/linkedin-posts/", json=post_data)
    
    @task(1)  # Low frequency
    def batch_create(self):
        batch_data = PostDataFactory.build_batch(5)
        self.client.post("/linkedin-posts/batch", json={"posts": batch_data})
```

## 🎯 Common Patterns

### 1. Test Data Generation
```python
# Use Factory Boy for consistent data
class PostDataFactory(Factory):
    class Meta:
        model = dict
    
    content = FactoryFaker('text', max_nb_chars=500)
    post_type = FactoryFaker('random_element', elements=['announcement', 'educational', 'update'])
    tone = FactoryFaker('random_element', elements=['professional', 'casual', 'friendly'])
    target_audience = FactoryFaker('random_element', elements=['tech professionals', 'marketers', 'developers'])
    industry = FactoryFaker('random_element', elements=['technology', 'marketing', 'finance'])
```

### 2. Async Testing
```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async operations."""
    result = await async_function()
    assert result is not None

# Test concurrent operations
@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent operations."""
    tasks = [async_function() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
```

### 3. Mocking External Services
```python
@pytest.fixture
def mock_external_api():
    """Mock external API calls."""
    with aioresponses() as m:
        m.post(
            "http://external-api/process",
            payload={"result": "success"},
            status=200
        )
        yield m

@pytest.mark.asyncio
async def test_with_mocked_api(mock_external_api):
    """Test with mocked external API."""
    result = await call_external_api()
    assert result["result"] == "success"
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd agents/backend/onyx/server/features/linkedin_posts
   
   # Add to PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Dependency Issues**
   ```bash
   # Update pip
   pip install --upgrade pip
   
   # Install with specific versions
   pip install -r tests/requirements-test.txt --force-reinstall
   ```

3. **Test Timeouts**
   ```bash
   # Increase timeout
   pytest tests/ --timeout=300
   
   # Run specific slow tests
   pytest tests/ -m "slow" --timeout=600
   ```

4. **Memory Issues**
   ```bash
   # Run with memory profiling
   python -m memory_profiler tests/debug/test_advanced_debug.py
   
   # Force garbage collection
   python -c "import gc; gc.collect()"
   ```

### Getting Help

1. **Check Logs**: Look at `reports/` directory for detailed logs
2. **Verbose Output**: Use `--verbose` flag for more details
3. **Debug Mode**: Use debug fixtures for detailed information
4. **Documentation**: Check the main testing documentation

## 🎉 Next Steps

1. **Explore Advanced Features**: Try property-based testing with Hypothesis
2. **Customize Tests**: Adapt tests for your specific use cases
3. **Add New Tests**: Create tests for new features
4. **Optimize Performance**: Use benchmarking to improve performance
5. **Integrate with CI/CD**: Set up automated testing in your pipeline

This quick start guide should get you up and running with the advanced testing system quickly. For more detailed information, refer to the main testing documentation. 