#!/bin/bash

# Deployment script for Universal Model Benchmark AI

set -e

echo "🚀 Starting deployment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
cd python
pip install -r requirements.txt
cd ..

# Build Rust library
echo "🦀 Building Rust library..."
cd rust
cargo build --release
cd ..

# Run tests
echo "🧪 Running tests..."
cd python
pytest tests/ -v
cd ..

# Create directories
echo "📁 Creating directories..."
mkdir -p data results models logs

# Set permissions
chmod +x python/cli/main.py

echo "✅ Deployment complete!"
echo ""
echo "To start the API server:"
echo "  python -m api.rest_api"
echo ""
echo "To use CLI:"
echo "  python -m cli.main --help"












