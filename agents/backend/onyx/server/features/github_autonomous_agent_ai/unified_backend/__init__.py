"""
Unified Backend
================
Backend built for Unified AI Model

This backend provides a comprehensive API for AI chat and text generation.

Quick Start:
    uvicorn main:app --host 0.0.0.0 --port 8080
    
    Or:
    python main.py

API Endpoints:
    - /docs - Swagger UI documentation
    - /api/v1/health - Health check
    - /api/v1/chat - Chat with memory
    - /api/v1/generate - Text generation
    - /api/v1/agents - Continuous AI agents

Configuration:
    Set environment variables:
    - OPENROUTER_API_KEY or DEEPSEEK_API_KEY for LLM access
    - UNIFIED_AI_PORT (default: 8050)
    - UNIFIED_AI_HOST (default: 0.0.0.0)
"""

from unified_ai_model import __version__

__all__ = ["__version__"]
