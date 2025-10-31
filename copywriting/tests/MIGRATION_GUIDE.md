# Test Migration Guide
=====================

Complete guide for migrating existing tests to the refactored architecture.

## Overview

This guide covers the migration from the original test structure to the new refactored architecture, which provides:

- **Unified Base Classes**: All tests inherit from `BaseTest`
- **Centralized Configuration**: Using `TestConfig` for all settings
- **Optimized Fixtures**: Shared fixtures in `refactored_conftest.py`
- **Data Management**: Automated test data handling
- **Performance Monitoring**: Built-in performance tracking
- **Parallel Execution**: Support for parallel test runs

## Migration Process

### 1. Pre-Migration Analysis

Before starting migration, analyze your current test structure:

```bash
# Analyze a specific test file
python tests/migration/migration_utils.py --analyze tests/test_api.py

# Analyze all test files
python tests/migration/migration_utils.py --migrate
```

### 2. Create Backup

Always create a backup before migration:

```bash
# The migration utility automatically creates backups
python tests/migration/migration_utils.py --migrate
```

Backups are stored in `tests/backup/backup_YYYYMMDD_HHMMSS/`

### 3. Run Migration

Execute the migration process:

```bash
# Migrate all tests
python tests/migration/migration_utils.py --migrate

# Migrate with custom options
python tests/migration/migration_utils.py --migrate --backup --validate
```

### 4. Validate Migration

After migration, validate the results:

```bash
# Validate migrated tests
python tests/migration/migration_utils.py --validate tests/migrated/

# Check for issues
python tests/validate_tests.py
```

## Migration Transformations

### 1. Import Updates

**Before:**
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
```

**After:**
```python
# Migrated to refactored architecture
from tests.base import BaseTest
from tests.config.test_config import TestConfig
from tests.data.test_data_manager import TestDataManager
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
```

### 2. Class Inheritance

**Before:**
```python
class TestCopywritingAPI:
    def test_generate_variants(self, client):
        # test implementation
```

**After:**
```python
class TestCopywritingAPI(BaseTest):
    def test_generate_variants(self, client):
        # test implementation
```

### 3. Setup Method Updates

**Before:**
```python
def setUp(self):
    self.client = TestClient(app)
    self.mock_celery = Mock()
```

**After:**
```python
def setUp(self):
    super().setUp()
    self.config = TestConfig()
    self.data_manager = TestDataManager()
    self.client = TestClient(app)
    self.mock_celery = Mock()
```

### 4. Configuration Usage

**Before:**
```python
def test_with_timeout(self):
    # Hard-coded timeout
    result = some_operation(timeout=30)
```

**After:**
```python
def test_with_timeout(self):
    # Using configuration
    result = some_operation(timeout=self.config.timeout_seconds)
```

### 5. Data Management

**Before:**
```python
def test_with_data(self):
    test_data = {
        "product": "Test Product",
        "description": "Test Description"
    }
    # Use test_data
```

**After:**
```python
def test_with_data(self):
    test_data = self.data_manager.get_test_data("product_variants")
    # Use test_data
```

## Manual Migration Steps

If automated migration doesn't work, follow these manual steps:

### 1. Update Imports

Add these imports to the top of each test file:

```python
from tests.base import BaseTest
from tests.config.test_config import TestConfig
from tests.data.test_data_manager import TestDataManager
```

### 2. Update Class Definitions

Change all test classes to inherit from `BaseTest`:

```python
# Change this:
class TestSomething:

# To this:
class TestSomething(BaseTest):
```

### 3. Update Setup Methods

Add configuration and data manager setup:

```python
def setUp(self):
    super().setUp()
    self.config = TestConfig()
    self.data_manager = TestDataManager()
    # ... existing setup code
```

### 4. Replace Hard-coded Values

Replace hard-coded configuration values:

```python
# Instead of:
timeout = 30
max_retries = 3

# Use:
timeout = self.config.timeout_seconds
max_retries = self.config.max_retries
```

### 5. Use Test Data Manager

Replace hard-coded test data:

```python
# Instead of:
test_data = {"key": "value"}

# Use:
test_data = self.data_manager.get_test_data("data_key")
```

## Migration Checklist

- [ ] **Backup Created**: Backup of original files created
- [ ] **Imports Updated**: All necessary imports added
- [ ] **Classes Updated**: All test classes inherit from `BaseTest`
- [ ] **Setup Methods**: Updated to call `super().setUp()`
- [ ] **Configuration**: Using `TestConfig` instead of hard-coded values
- [ ] **Data Management**: Using `TestDataManager` for test data
- [ ] **Fixtures**: Updated to use refactored fixtures
- [ ] **Validation**: All migrated tests pass validation
- [ ] **Testing**: All tests run successfully
- [ ] **Documentation**: Updated test documentation

## Rollback Process

If migration causes issues, rollback using the backup:

```bash
# Rollback to specific backup
python tests/migration/migration_utils.py --rollback tests/backup/backup_20231201_143022/

# Verify rollback
python tests/validate_tests.py
```

## Post-Migration Tasks

### 1. Update Test Runner

Use the new refactored test runner:

```bash
# Run all tests
python tests/refactored_test_runner.py

# Run with parallel execution
python tests/refactored_test_runner.py --parallel --workers 4

# Generate HTML report
python tests/refactored_test_runner.py --format html --output report.html
```

### 2. Update CI/CD

Update your CI/CD pipeline to use the new test runner:

```yaml
# .github/workflows/test.yml
- name: Run Refactored Tests
  run: |
    python tests/refactored_test_runner.py --parallel --format json --output test-results.json
```

### 3. Update Documentation

Update test documentation to reflect the new architecture:

- Update README files
- Update API documentation
- Update developer guides

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all required packages are installed
   - Check Python path configuration
   - Verify file locations

2. **Class Inheritance Issues**
   - Ensure all test classes inherit from `BaseTest`
   - Check method signatures
   - Verify super() calls

3. **Configuration Issues**
   - Check `TestConfig` initialization
   - Verify configuration file paths
   - Test configuration loading

4. **Data Management Issues**
   - Check test data file locations
   - Verify data manager initialization
   - Test data loading

### Debug Commands

```bash
# Check test discovery
python -m pytest --collect-only

# Run with verbose output
python tests/refactored_test_runner.py --format console

# Validate specific test file
python tests/validate_tests.py tests/migrated/test_api.py

# Check configuration
python -c "from tests.config.test_config import TestConfig; print(TestConfig())"
```

## Best Practices

### 1. Gradual Migration

- Migrate one test file at a time
- Test each migration before proceeding
- Keep backups of each step

### 2. Validation

- Always validate migrated tests
- Run tests after each migration
- Check for performance regressions

### 3. Documentation

- Document migration decisions
- Update test documentation
- Keep migration logs

### 4. Team Coordination

- Coordinate migration with team
- Share migration status
- Plan for rollback if needed

## Support

If you encounter issues during migration:

1. Check the migration report
2. Review the troubleshooting section
3. Use the validation tools
4. Consider rollback if necessary
5. Contact the development team

## Migration Examples

### Example 1: Simple Test Migration

**Before:**
```python
def test_simple_function():
    result = simple_function("test")
    assert result == "expected"
```

**After:**
```python
class TestSimpleFunction(BaseTest):
    def setUp(self):
        super().setUp()
        self.config = TestConfig()
    
    def test_simple_function(self):
        result = simple_function("test")
        assert result == "expected"
```

### Example 2: API Test Migration

**Before:**
```python
def test_api_endpoint(client):
    response = client.post("/api/test", json={"data": "test"})
    assert response.status_code == 200
```

**After:**
```python
class TestAPIEndpoint(BaseTest):
    def setUp(self):
        super().setUp()
        self.config = TestConfig()
        self.data_manager = TestDataManager()
    
    def test_api_endpoint(self, client):
        test_data = self.data_manager.get_test_data("api_request")
        response = client.post("/api/test", json=test_data)
        assert response.status_code == 200
```

### Example 3: Performance Test Migration

**Before:**
```python
def test_performance():
    start_time = time.time()
    # perform operation
    duration = time.time() - start_time
    assert duration < 1.0
```

**After:**
```python
class TestPerformance(BaseTest):
    def setUp(self):
        super().setUp()
        self.config = TestConfig()
    
    def test_performance(self):
        start_time = time.time()
        # perform operation
        duration = time.time() - start_time
        assert duration < self.config.performance_threshold
```

## Conclusion

The refactored test architecture provides significant improvements in:

- **Maintainability**: Centralized configuration and base classes
- **Performance**: Optimized fixtures and parallel execution
- **Reliability**: Better error handling and validation
- **Scalability**: Support for large test suites
- **Monitoring**: Built-in performance and memory tracking

Follow this guide to successfully migrate your tests and take advantage of these improvements.
