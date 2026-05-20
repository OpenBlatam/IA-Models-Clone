#!/bin/bash
# ============================================================================
# CI/CD Dependency Check
# Lightweight check for CI/CD pipelines
# ============================================================================

set -e

echo "=========================================="
echo "CI/CD Dependency Check"
echo "=========================================="

# Check Python version
python3 --version

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Basic validation
echo "Validating installation..."
python3 -c "import fastapi; import uvicorn; import pydantic; print('✓ Core packages OK')"

# Check for critical packages
echo "Checking critical packages..."
python3 -c "
import sys
critical = ['fastapi', 'uvicorn', 'pydantic', 'sqlalchemy']
missing = []
for pkg in critical:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'✗ Missing: {missing}')
    sys.exit(1)
print('✓ All critical packages installed')
"

# Security check (if available)
if command -v safety &> /dev/null; then
    echo "Running security check..."
    safety check -r requirements.txt --short-report || true
fi

echo "=========================================="
echo "✓ Dependency check completed"
echo "=========================================="



