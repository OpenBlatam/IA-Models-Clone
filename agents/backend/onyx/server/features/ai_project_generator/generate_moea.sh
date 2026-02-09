#!/bin/bash

# Script to generate MOEA project

echo "========================================"
echo "MOEA Project Generator"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found!"
    echo "Please install Python 3.8+ and add it to your PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)

echo "[1/3] Checking server status..."
if ! curl -s http://localhost:8020/health > /dev/null 2>&1; then
    echo "Server is not running. Starting server in background..."
    $PYTHON_CMD main.py &
    SERVER_PID=$!
    echo "Waiting for server to start (PID: $SERVER_PID)..."
    sleep 5
fi

echo "[2/3] Generating MOEA project..."
$PYTHON_CMD generate_moea.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Generation failed!"
    echo "Trying direct generation method..."
    $PYTHON_CMD generate_moea_direct.py
fi

echo ""
echo "[3/3] Checking generated project..."
if [ -d "generated_projects/moea_optimization_system" ]; then
    echo ""
    echo "========================================"
    echo "SUCCESS! Project generated at:"
    echo "generated_projects/moea_optimization_system"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. cd generated_projects/moea_optimization_system/backend"
    echo "2. pip install -r requirements.txt"
    echo "3. cd ../frontend"
    echo "4. npm install"
    echo ""
else
    echo ""
    echo "Project directory not found. Check logs above for errors."
    echo ""
fi

