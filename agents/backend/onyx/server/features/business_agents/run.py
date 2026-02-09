#!/usr/bin/env python3
"""
Quick start script for the Ultimate Quantum AI API.
Usage: python run.py [--port PORT] [--host HOST] [--reload]
"""
import argparse
import sys
import uvicorn
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ultimate Quantum AI API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    app_module = "agents.backend.onyx.server.features.business_agents.main:app"
    
    uvicorn.run(
        app_module,
        host=args.host,
        port=args.port,
        reload=args.reload and args.workers == 1,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        loop="uvloop",
        http="httptools",
        timeout_keep_alive=5,
    )
