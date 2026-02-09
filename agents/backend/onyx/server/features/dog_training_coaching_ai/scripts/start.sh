#!/bin/bash
# Start script for Dog Training Coaching AI

set -e

echo "Starting Dog Training Coaching AI..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Using environment variables."
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# Start server
python main.py

