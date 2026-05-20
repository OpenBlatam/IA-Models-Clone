#!/bin/bash
# Start API and open debugging tools

echo "🚀 Starting API and Debugging Tools"
echo ""

# Start API in background
python start_api_and_debug.py --tool debug &

# Wait a moment
sleep 3

echo "✅ API started. Debug tool should open shortly."
echo ""
echo "To stop: Press Ctrl+C or run 'pkill -f start_api_and_debug'"



