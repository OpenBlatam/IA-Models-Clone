# 🚀 Advanced Testing Features

## Overview

This document describes the advanced testing tools and features available for the TruthGPT test suite.

## 📊 Test Statistics

- **Total Tests**: 214+
- **Test Files**: 14
- **Utility Files**: 5
- **Test Categories**: 12

## 🛠️ Available Tools

### 1. HTML Report Generator

Generate beautiful, interactive HTML reports from test results.

**Usage:**
```bash
# Generate HTML report after running tests
python generate_html_report.py

# Or use the flag in the main runner
python run_unified_tests.py --html-report
```

**Features:**
- Visual statistics with progress bars
- Color-coded test results
- Detailed failure/error traces
- Responsive design
- Exportable reports

**Output:**
- HTML file saved as `test_report_YYYYMMDD_HHMMSS.html`
- Open in any web browser

### 2. Continuous Test Runner (Watch Mode)

Automatically run tests when files change - perfect for development!

**Usage:**
```bash
# Watch all files and run all tests
python run_tests_watch.py

# Watch and run specific category
python run_tests_watch.py core

# Custom watch interval (default: 1 second)
python run_tests_watch.py --interval 2.0

# Watch specific directories
python run_tests_watch.py --dirs core tests
```

**Features:**
- Automatic test execution on file changes
- Configurable watch directories
- Configurable check interval
- Category-specific test runs
- Clean output with timestamps

**Use Cases:**
- TDD (Test-Driven Development)
- Continuous integration during development
- Quick feedback loop

### 3. Test Coverage Analyzer

Analyze which parts of your code are covered by tests.

**Usage:**
```bash
# Basic coverage analysis
python test_coverage.py

# Use coverage.py library if available
python test_coverage.py --use-library
```

**Features:**
- File-by-file coverage analysis
- Test-to-code mapping
- Coverage percentages
- Summary statistics
- Integration with coverage.py library

**Output:**
- Text report: `coverage_report.txt`
- HTML report (if using coverage.py): `htmlcov/index.html`

**Installation (optional):**
```bash
pip install coverage
```

### 4. Test Profiler

Identify slow tests and performance bottlenecks.

**Usage:**
```bash
# Profile all tests
python test_profiler.py

# Profile specific category
python test_profiler.py core
```

**Features:**
- Execution time per test
- Category breakdown
- Slowest tests identification
- Performance recommendations
- Average test time analysis

**Output:**
- Text report: `test_profile_report.txt`
- Identifies top 20 slowest tests
- Category time distribution
- Performance recommendations

### 5. Enhanced Test Runner

The main test runner with advanced features.

**Usage:**
```bash
# Basic run
python run_unified_tests.py

# Run specific category
python run_unified_tests.py core

# Generate HTML report
python run_unified_tests.py --html-report

# Save detailed report
python run_unified_tests.py --save-report

# Parallel execution (future)
python run_unified_tests.py --parallel=4
```

**Features:**
- Unified test execution
- Category filtering
- HTML report generation
- Detailed text reports
- Parallel execution support (planned)

## 📈 Workflow Examples

### Development Workflow

```bash
# Terminal 1: Watch mode for continuous testing
python run_tests_watch.py core

# Terminal 2: Make changes to core files
# Tests automatically run when files change
```

### CI/CD Workflow

```bash
# Run all tests with HTML report
python run_unified_tests.py --html-report

# Generate coverage report
python test_coverage.py --use-library

# Profile tests for performance
python test_profiler.py
```

### Debugging Workflow

```bash
# Run specific failing category
python run_unified_tests.py integration

# Generate detailed report
python run_unified_tests.py integration --save-report

# Profile to find slow tests
python test_profiler.py integration
```

## 🎯 Best Practices

1. **During Development**: Use watch mode for instant feedback
2. **Before Committing**: Run full test suite with HTML report
3. **Performance Testing**: Use profiler to identify bottlenecks
4. **Coverage Goals**: Aim for 80%+ coverage on critical modules
5. **CI/CD Integration**: Use all tools in automated pipelines

## 📝 Integration Examples

### GitHub Actions

```yaml
- name: Run Tests
  run: python run_unified_tests.py --html-report

- name: Generate Coverage
  run: python test_coverage.py --use-library

- name: Upload HTML Report
  uses: actions/upload-artifact@v2
  with:
    name: test-report
    path: test_report_*.html
```

### GitLab CI

```yaml
test:
  script:
    - python run_unified_tests.py --html-report
    - python test_coverage.py --use-library
  artifacts:
    paths:
      - test_report_*.html
      - htmlcov/
```

## 🔮 Future Enhancements

- [ ] Full parallel test execution
- [ ] Test result caching
- [ ] Visual test coverage maps
- [ ] Test dependency analysis
- [ ] Mutation testing support
- [ ] Performance regression detection
- [ ] Test flakiness detection
- [ ] Automated test generation

## 📚 Related Documentation

- `README.md` - General project documentation
- `TEST_SUMMARY.md` - Test suite overview
- `QUICK_START.md` - Quick start guide
- `READY_TO_TEST.md` - Test readiness checklist

---

**Last Updated**: Advanced testing framework with 4 new tools
**Status**: ✅ Production Ready
