# Run Tests and Debug - Quick Guide

## Quick Start

### Option 1: Simple Batch Script (Recommended)
```cmd
cd C:\blatam-academy\agents\backend\onyx\server\features\bulk
run_tests_simple.bat
```

This script will:
1. Find Python automatically
2. Run quick environment test
3. Verify test files
4. Run all tests
5. Show summary

### Option 2: Python Script
```cmd
cd C:\blatam-academy\agents\backend\onyx\server\features\bulk
python run_and_debug.py
```

### Option 3: Quick Test First
```cmd
python quick_test.py
```

This checks:
- Python installation
- Required modules
- Server status

## Individual Tests

Run tests individually for debugging:

```cmd
python test_api_responses.py
python test_api_advanced.py
python test_security.py
```

## Verify Test Files

Check if test files are syntactically correct:

```cmd
python verify_tests.py
```

## Debugging Tips

### If Python Not Found:
1. Install Python from https://www.python.org/downloads/
2. Make sure "Add Python to PATH" is checked
3. Restart terminal

### If Dependencies Missing:
```cmd
pip install -r requirements.txt
```

### If Server Not Running:
```cmd
python api_frontend_ready.py
```

Wait for "Application startup complete" message.

### Check Server Status:
```cmd
python -c "import requests; print(requests.get('http://localhost:8000/api/health').json())"
```

## Files Created

- `run_tests_simple.bat` - Simple batch script to run all tests
- `run_and_debug.py` - Python script with detailed debugging
- `verify_tests.py` - Verify test file syntax
- `quick_test.py` - Quick environment check

## Expected Output

After running tests, you should see:
- Test results for each test file
- Summary with pass/fail status
- Any error messages with details

## Troubleshooting

### Tests Timeout
- Check if server is running
- Check network connectivity
- Increase timeout in test files if needed

### Import Errors
- Install missing modules: `pip install -r requirements.txt`
- Check Python version (should be 3.8+)

### Connection Errors
- Start server: `python api_frontend_ready.py`
- Check port 8000 is not in use
- Check firewall settings









