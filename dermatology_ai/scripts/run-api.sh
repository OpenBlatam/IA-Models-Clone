#!/bin/bash
# ============================================================================
# Run API with Debugging
# Runs the Dermatology AI API with debugging enabled
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-debug}
RELOAD=${RELOAD:-true}

echo "=========================================="
echo "Starting Dermatology AI API"
echo "=========================================="
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Log Level: $LOG_LEVEL"
echo "  Reload: $RELOAD"
echo ""

# Check if virtual environment exists
if [ -d "venv" ] || [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Virtual environment found${NC}"
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        source .venv/bin/activate
    fi
else
    echo -e "${YELLOW}⚠ No virtual environment found${NC}"
    echo "  Running with system Python"
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠ FastAPI not found, installing dependencies...${NC}"
    pip install -r requirements-optimized.txt
fi

echo ""
echo -e "${BLUE}Starting server...${NC}"
echo ""

# Run with uvicorn
if [ "$RELOAD" = "true" ]; then
    uvicorn main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL" \
        --reload-dir . \
        --reload-include "*.py" \
        --reload-exclude "*.pyc" \
        --reload-exclude "__pycache__"
else
    uvicorn main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
fi



