"""
PDF Variantes API - Run Script
Main entry point for running the API server
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

def main():
    """Run the API server"""
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT", "development").lower() == "development"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("=" * 60)
    print("🚀 PDF Variantes API - Starting Server")
    print("=" * 60)
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"🔄 Reload: {reload}")
    print(f"📊 Log Level: {log_level}")
    print("=" * 60)
    print("📚 API Docs: http://{}:{}/docs".format(host if host != "0.0.0.0" else "localhost", port))
    print("🏥 Health Check: http://{}:{}/health".format(host if host != "0.0.0.0" else "localhost", port))
    print("=" * 60)
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()






