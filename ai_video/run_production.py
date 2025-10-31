from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import os
import sys
from pathlib import Path
        from production_ready_system import ProductionWorkflowManager, ProductionAPI, ProductionConfig
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Production Runner - Complete AI Video System

Simple script to run the production system with all optimizations.
"""


# Setup environment
os.environ.update({
    "ENVIRONMENT": "production",
    "DEBUG": "false",
    "HOST": "0.0.0.0",
    "PORT": "8001",  # API server port
    "JWT_SECRET": "production_secret_key_2024",
    "API_KEY_REQUIRED": "false",
    "ENABLE_NUMBA": "true",
    "ENABLE_DASK": "true", 
    "ENABLE_REDIS": "true",
    "ENABLE_PROMETHEUS": "true",
    "PROMETHEUS_PORT": "9090",  # Prometheus port
    "ENABLE_RAY": "false",
    "LOG_LEVEL": "INFO"
})

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    """Run production system."""
    try:
        
        print("🚀 Starting AI Video Production System...")
        
        # Create config
        config = ProductionConfig()
        config.port = 8001  # Ensure API uses port 8001
        
        # Create workflow manager
        workflow_manager = ProductionWorkflowManager(config)
        
        # Initialize system
        if await workflow_manager.initialize():
            print("✅ System initialized successfully")
            
            # Create API server
            api_server = ProductionAPI(workflow_manager, config)
            
            # Start server
            print("🌐 Starting API server on http://localhost:8001")
            print("📊 Prometheus metrics on http://localhost:9090")
            await api_server.start_server()
            
        else:
            print("❌ Failed to initialize system")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 