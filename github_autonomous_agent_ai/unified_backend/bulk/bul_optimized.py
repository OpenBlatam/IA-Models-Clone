"""
BUL - Business Universal Language
Optimized Modular Version
=========================

Clean, modular AI-powered document generation system for SMEs.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import modular components
from modules import DocumentProcessor, QueryAnalyzer, BusinessAgentManager, APIHandler
from modules.api_handler import DocumentRequest, DocumentResponse, TaskStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bul.log')
    ]
)

logger = logging.getLogger(__name__)

class BULSystem:
    """Optimized BUL system with modular architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize modular components
        self.document_processor = DocumentProcessor(self.config)
        self.query_analyzer = QueryAnalyzer()
        self.agent_manager = BusinessAgentManager(self.config)
        self.api_handler = APIHandler(
            self.document_processor,
            self.query_analyzer,
            self.agent_manager
        )
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="BUL - Business Universal Language",
            description="Optimized AI-powered document generation system for SMEs",
            version="3.0.0"
        )
        
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("BUL System initialized with modular architecture")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'supported_formats': ['markdown', 'html', 'pdf'],
            'output_directory': 'generated_documents',
            'max_concurrent_tasks': 5,
            'task_timeout': 300,
            'business_areas': {
                'marketing': {'priority': 1},
                'sales': {'priority': 1},
                'operations': {'priority': 2},
                'hr': {'priority': 2},
                'finance': {'priority': 1}
            }
        }
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return self.api_handler.get_system_info()
        
        @self.app.get("/health")
        async def health_check():
            return self.api_handler.get_health_status()
        
        @self.app.post("/documents/generate", response_model=DocumentResponse)
        async def generate_document(request: DocumentRequest, background_tasks: BackgroundTasks):
            """Generate a document based on the provided query."""
            return await self.api_handler.generate_document(request)
        
        @self.app.get("/tasks/{task_id}/status", response_model=TaskStatus)
        async def get_task_status(task_id: str):
            """Get the status of a document generation task."""
            return await self.api_handler.get_task_status(task_id)
        
        @self.app.get("/tasks")
        async def list_tasks():
            """List all tasks."""
            return self.api_handler.list_tasks()
        
        @self.app.delete("/tasks/{task_id}")
        async def delete_task(task_id: str):
            """Delete a task."""
            return self.api_handler.delete_task(task_id)
        
        @self.app.get("/agents")
        async def get_agents():
            """Get available business area agents."""
            return {
                "agents": self.agent_manager.get_all_capabilities()
            }
        
        @self.app.get("/agents/{area}")
        async def get_agent_info(area: str):
            """Get information about a specific agent."""
            capabilities = self.agent_manager.get_agent_capabilities(area)
            if not capabilities:
                raise HTTPException(status_code=404, detail="Agent not found")
            return capabilities
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the BUL system."""
        logger.info(f"Starting optimized BUL system on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BUL - Business Universal Language (Optimized)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config and Path(args.config).exists():
        # Load config from file (implementation would depend on config format)
        logger.info(f"Loading configuration from {args.config}")
    
    # Create and run system
    system = BULSystem(config)
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

