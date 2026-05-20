from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import uvicorn
import os
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
FastAPI Application Runner
Production-ready runner with proper configuration and error handling.
"""


# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for the FastAPI application."""
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    workers = int(os.getenv("WORKERS", "1"))
    
    print(f"Starting FastAPI application...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Log Level: {log_level}")
    print(f"Workers: {workers}")
    
    # Start the application
    uvicorn.run(
        "main_app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        workers=workers if not reload else 1,
        access_log=True,
        use_colors=True
    )

match __name__:
    case "__main__":
    main() 