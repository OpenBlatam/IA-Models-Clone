# Install Python - Quick Guide

## Why Python is Needed

The test scripts require Python to run. Currently, Python is not properly installed on your system.

## Quick Install

### Option 1: Official Python (Recommended)

1. **Download Python:**
   - Go to: https://www.python.org/downloads/
   - Download the latest Python 3.x version

2. **Install Python:**
   - Run the installer
   - ✅ **IMPORTANT:** Check "Add Python to PATH"
   - Click "Install Now"

3. **Verify Installation:**
   ```cmd
   python --version
   pip --version
   ```

4. **Install Dependencies:**
   ```cmd
   cd C:\blatam-academy\agents\backend\onyx\server\features\bulk
   pip install -r requirements.txt
   ```

### Option 2: Microsoft Store (Not Recommended)

The Microsoft Store Python alias is currently not working. If you want to use it:
1. Open Microsoft Store
2. Search for "Python 3.12" or "Python 3.11"
3. Install
4. Disable app execution aliases (see below)

### Disable Windows Store Python Aliases

If you install Python from python.org but Windows Store still intercepts:

1. Open **Settings** (Win + I)
2. Go to **Apps** → **Advanced app settings**
3. Click **App execution aliases**
4. Turn **OFF**:
   - `python.exe`
   - `python3.exe`
   - `pythonw.exe`

This allows real Python to work.

## After Installation

Once Python is installed:

```cmd
cd C:\blatam-academy\agents\backend\onyx\server\features\bulk

REM Verify
python --version
python quick_test.py

REM Install dependencies
pip install -r requirements.txt

REM Run tests
run_all_tests.bat
```

## Test Files Ready

All test files are fixed and ready:
- ✅ `test_api_responses.py`
- ✅ `test_api_advanced.py`
- ✅ `test_security.py`

They will run successfully once Python is installed.









