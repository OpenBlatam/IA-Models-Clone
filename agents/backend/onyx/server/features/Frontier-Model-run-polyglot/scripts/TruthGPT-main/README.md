# TruthGPT - Unified Test System

## 🚀 Quick Start

```bash
# 1. Check environment
python quick_check.py

# 2. Setup if needed
python setup_environment.py

# 3. Run tests
python run_unified_tests.py
```

See [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) for detailed instructions.

---

## TruthGPT

Is simple add a open source model then optimize or just create a ia model for your necessity.

## Testing System

### Quick Commands

```bash
# Run all tests
python run_unified_tests.py

# Quick environment check
python quick_check.py

# Setup environment
python setup_environment.py

# Check dependencies
python check_dependencies.py
```

### Test Categories

- `core` - Core component tests
- `optimization` - Optimization engine tests
- `models` - Model management tests
- `training` - Training system tests
- `inference` - Inference engine tests
- `monitoring` - Monitoring system tests
- `integration` - Integration tests
- `edge` - Edge cases and stress tests
- `performance` - Performance and benchmark tests
- `security` - Security and validation tests
- `compatibility` - Compatibility tests
- `regression` - Regression tests
- `validation` - Validation tests
- `benchmarks` - Benchmark tests

### Advanced Options

```bash
# Run specific category
python run_unified_tests.py core

# Stop on first failure
python run_unified_tests.py --failfast

# Verbose output
python run_unified_tests.py --verbose

# Quiet mode (for CI/CD)
python run_unified_tests.py --quiet

# Export to JSON
python run_unified_tests.py --json results.json

# Quick check before running
python run_unified_tests.py --check

# List all categories
python run_unified_tests.py --list
```

## Documentation

- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Quick start guide
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Initial improvements
- [MORE_IMPROVEMENTS.md](MORE_IMPROVEMENTS.md) - Additional improvements
- [READY_TO_TEST.md](READY_TO_TEST.md) - Test readiness checklist

## Helper Scripts

### Windows

```cmd
# Batch script
run_tests.bat                    # All tests
run_tests.bat core               # Core tests only
run_tests.bat --failfast         # Stop on first failure

# PowerShell script
.\run_tests.ps1                   # All tests
.\run_tests.ps1 core             # Core tests only
.\run_tests.ps1 --json report.json
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy

Install with:
```bash
pip install torch numpy
```

Or use the setup script:
```bash
python setup_environment.py
```

## Troubleshooting

### Python Not Found
1. Install Python from https://www.python.org/downloads/
2. Check "Add Python to PATH" during installation

### Missing Dependencies
```bash
python setup_environment.py
```

### Import Errors
Make sure you're in the TruthGPT-main directory and run:
```bash
python quick_check.py
```

## Status

✅ **204+ tests** ready to run
✅ **14 test categories** available
✅ **Advanced features** implemented
✅ **CI/CD ready** with JSON export

See [READY_TO_TEST.md](READY_TO_TEST.md) for complete status.
