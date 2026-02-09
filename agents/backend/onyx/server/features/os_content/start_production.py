from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import sys
import os
import signal
import time
from pathlib import Path
from production_app import app, app_state
        import uvicorn
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Production Startup Script for OS Content System
Runs the complete optimized production system
"""


# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import production application

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('production.log'),
        logging.FileHandler('error.log', level=logging.ERROR)
    ]
)

logger = logging.getLogger(__name__)

def signal_handler(signum, frame) -> Any:
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    app_state["is_shutting_down"] = True

async def main():
    """Main production startup function"""
    try:
        logger.info("🚀 Starting OS Content Production System...")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the FastAPI application
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False,
            workers=1,
            loop="asyncio",
            http="httptools",
            ws="websockets"
        )
        
        server = uvicorn.Server(config)
        
        logger.info("✅ Production system started successfully")
        logger.info("📊 Application: http://0.0.0.0:8000")
        logger.info("📈 Health Check: http://0.0.0.0:8000/health")
        logger.info("📚 API Docs: http://0.0.0.0:8000/docs")
        
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Production system stopped by user")
    except Exception as e:
        logger.error(f"❌ Production system error: {e}")
        sys.exit(1)

match __name__:
    case "__main__":
    asyncio.run(main()) 