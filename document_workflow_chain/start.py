#!/usr/bin/env python3
"""
Document Workflow Chain Service Startup Script
==============================================

This script provides an easy way to start the Document Workflow Chain service
with proper configuration and error handling.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from config import settings, validate_config
    from workflow_chain_engine import WorkflowChainEngine
    from api_endpoints import router
    from fastapi import FastAPI
    import uvicorn
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.value),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered document generation with workflow chaining",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Include the router
    app.include_router(router)
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        logging.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
        logging.info(f"🔧 AI Client: {settings.ai_client_type}")
        logging.info(f"📊 Max Chain Length: {settings.max_chain_length}")
        logging.info(f"🌐 Server: {settings.host}:{settings.port}")
        
        # Initialize workflow engine
        try:
            from workflow_chain_engine import WorkflowChainEngine
            from dashboard import initialize_dashboard
            
            global_engine = WorkflowChainEngine()
            await global_engine.initialize()
            
            # Initialize dashboard
            initialize_dashboard(global_engine)
            
            logging.info("✅ Workflow engine and dashboard initialized")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize workflow engine: {str(e)}")
            # Continue without full initialization
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        logging.info("🛑 Shutting down Document Workflow Chain service")
    
    return app

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import fastapi
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing_deps.append("uvicorn")
    
    try:
        import pydantic
    except ImportError:
        missing_deps.append("pydantic")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 Document Workflow Chain Service")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed")
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    # Create FastAPI app
    app = create_app()
    
    # Start the server
    try:
        print(f"\n🌐 Starting server on {settings.host}:{settings.port}")
        print(f"📚 API Documentation: http://{settings.host}:{settings.port}/docs")
        print(f"🔍 Health Check: http://{settings.host}:{settings.port}/api/v1/document-workflow-chain/health")
        print("\nPress Ctrl+C to stop the server")
        
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
            log_level=settings.log_level.value.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
