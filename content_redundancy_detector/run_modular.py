"""
Run Script for Modular Architecture
Entry point for the refactored modular system
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    from core.config import get_settings
    
    settings = get_settings()
    
    print("=" * 60)
    print("🚀 Content Redundancy Detector - Modular Architecture")
    print("=" * 60)
    print(f"📍 Host: {settings.host}")
    print(f"🔌 Port: {settings.port}")
    print(f"🌍 Environment: {settings.environment}")
    print(f"🔄 Reload: {settings.reload}")
    print(f"📊 Log Level: {settings.log_level}")
    print("=" * 60)
    print("📚 API Docs: http://{}:{}/docs".format(
        settings.host if settings.host != "0.0.0.0" else "localhost",
        settings.port
    ))
    print("🏥 Health: http://{}:{}/api/v1/health".format(
        settings.host if settings.host != "0.0.0.0" else "localhost",
        settings.port
    ))
    print("=" * 60)
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.reload
    )






