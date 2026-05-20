#!/usr/bin/env python3
"""
Run API with Debugging
======================
Script to run the API server with debugging enabled and tools.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the app
from api.main import app


def setup_debugging():
    """Setup debugging environment."""
    # Enable debug mode
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "debug"
    
    # Enable detailed error messages
    os.environ["DETAILED_ERRORS"] = "true"
    
    # Enable request/response logging
    os.environ["LOG_REQUESTS"] = "true"
    os.environ["LOG_RESPONSES"] = "true"
    
    # Enable performance monitoring
    os.environ["ENABLE_METRICS"] = "true"
    os.environ["ENABLE_PROFILING"] = "true"
    
    print("🐛 Debugging enabled:")
    print("   - Debug mode: ON")
    print("   - Detailed errors: ON")
    print("   - Request logging: ON")
    print("   - Response logging: ON")
    print("   - Metrics: ON")
    print("   - Profiling: ON")


def main():
    """Run the API server with debugging."""
    setup_debugging()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "debug").lower()
    
    print("=" * 60)
    print("🚀 PDF Variantes API - Starting Server (DEBUG MODE)")
    print("=" * 60)
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📊 Log Level: {log_level}")
    print(f"🌐 API URL: http://{host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    print(f"🔍 ReDoc: http://{host}:{port}/redoc")
    print("=" * 60)
    print()
    
    # Configure uvicorn with debugging
    uvicorn_config = {
        "app": app,
        "host": host,
        "port": port,
        "reload": reload,
        "log_level": log_level,
        "access_log": True,
        "use_colors": True,
        "reload_dirs": [str(project_root)] if reload else None,
        "reload_includes": ["*.py"] if reload else None,
    }
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



