#!/bin/bash
# Quick start script with debugging

echo "🚀 Quick Start - API with Debugging"
echo ""

# Check if API is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is already running on http://localhost:8000"
    echo ""
    echo "You can now:"
    echo "  - Use debug tool: python debug_api.py"
    echo "  - View docs: http://localhost:8000/docs"
    echo "  - View ReDoc: http://localhost:8000/redoc"
    echo ""
    read -p "Press Enter to open debug tool or Ctrl+C to exit..."
    python debug_api.py
else
    echo "Starting API with debugging..."
    echo ""
    python run_api_debug.py
fi



