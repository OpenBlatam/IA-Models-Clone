#!/bin/bash
# ============================================
# Unified AI Model - Quick Start (Linux/Mac)
# ============================================

echo ""
echo "============================================"
echo " UNIFIED AI MODEL - Starting..."
echo "============================================"
echo ""

# Check if API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "[WARNING] OPENROUTER_API_KEY not set!"
    echo ""
    echo "Set it with: export OPENROUTER_API_KEY=sk-or-v1-your-key"
    echo ""
fi

# Set default model to DeepSeek if not set
if [ -z "$UNIFIED_AI_DEFAULT_MODEL" ]; then
    export UNIFIED_AI_DEFAULT_MODEL="deepseek/deepseek-chat"
fi

# Set port if not set
if [ -z "$UNIFIED_AI_PORT" ]; then
    export UNIFIED_AI_PORT="8050"
fi

echo "Model: $UNIFIED_AI_DEFAULT_MODEL"
echo "Port: $UNIFIED_AI_PORT"
echo ""

# Run the server
python3 run.py



