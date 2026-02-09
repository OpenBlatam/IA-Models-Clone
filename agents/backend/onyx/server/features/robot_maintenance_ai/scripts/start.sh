#!/bin/bash

set -e

echo "Starting Robot Maintenance AI..."

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable is not set!"
    echo "Please set it with: export OPENROUTER_API_KEY='your-key-here'"
    exit 1
fi

if [ ! -d "logs" ]; then
    mkdir -p logs
fi

if [ ! -d "data" ]; then
    mkdir -p data
fi

python -m uvicorn main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --log-level "${LOG_LEVEL:-info}"






