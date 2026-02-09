#!/bin/bash

# Setup script for Universal Model Benchmark AI

set -e

echo "🔧 Setting up Universal Model Benchmark AI..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "⚠️  Rust is not installed. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd python
pip install --upgrade pip
pip install -r requirements.txt
cd ..

# Build Rust library
echo "🦀 Building Rust library..."
cd rust
cargo build --release
cd ..

# Create directories
echo "📁 Creating directories..."
mkdir -p data results models logs experiments model_registry

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Database
DATABASE_URL=sqlite:///data/results.db

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO

# Rust
RUST_BACKTRACE=1
EOF
fi

echo "✅ Setup complete!"
echo ""
echo "To activate virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To start the API server:"
echo "  python -m api.rest_api"
echo ""
echo "To use CLI:"
echo "  python -m cli.main --help"












