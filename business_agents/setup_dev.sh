#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up Ultimate Quantum AI API development environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cat > .env << EOF
# App Configuration
APP_NAME=Ultimate Quantum AI ML NLP Benchmark
ALLOWED_ORIGINS=*
RPS_LIMIT=200
MAX_BODY_BYTES=5242880
REQUEST_TIMEOUT_SECONDS=30

# Features
FEATURES=

# Cache
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=10

# Security
ENFORCE_AUTH=false
API_KEY=
JWT_SECRET=
JWT_ALGORITHM=HS256

# HTTP Client
HTTP_TIMEOUT_SECONDS=5
HTTP_RETRIES=3
CB_FAIL_THRESHOLD=5
CB_RECOVERY_SECONDS=30

# Logging
LOG_LEVEL=INFO
JSON_LOGGING=false
DEBUG=false
EOF
    echo "✅ .env file created!"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Start Redis (optional): redis-server"
echo "  3. Run API: python run.py --reload"
echo "  4. Run tests: pytest tests/"
echo ""


