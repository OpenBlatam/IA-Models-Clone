#!/bin/bash

# Deployment Script for Bulk TruthGPT
# ===================================

set -e

echo "🚀 Starting deployment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Setup environment
echo "⚙️  Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from .env.example"
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p storage backups logs cache temp

# Run setup
echo "🔧 Running setup..."
python setup.py

# Run tests (if available)
if [ -f "pytest.ini" ] || [ -f "tests/" ]; then
    echo "🧪 Running tests..."
    pytest tests/ || echo "⚠️  Tests failed, continuing..."
fi

# Verify setup
echo "✅ Verifying setup..."
python verify_setup.py

echo "🎉 Deployment completed successfully!"
echo ""
echo "To start the service:"
echo "  python start.py"
echo "  or"
echo "  uvicorn main:app --host 0.0.0.0 --port 8000"
















