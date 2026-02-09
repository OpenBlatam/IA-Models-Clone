#!/bin/bash
# Run API with debugging enabled

echo "🚀 Starting API with debugging..."
echo ""

# Set debugging environment variables
export DEBUG=true
export LOG_LEVEL=debug
export DETAILED_ERRORS=true
export LOG_REQUESTS=true
export LOG_RESPONSES=true
export ENABLE_METRICS=true
export ENABLE_PROFILING=true

# Run the API
python run_api_debug.py



