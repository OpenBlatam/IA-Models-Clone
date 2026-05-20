"""
BUL - Business Universal Language
================================

A clean, consolidated AI-powered document generation system for SMEs.
This is the refactored version that consolidates all previous iterations.
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
from pydantic import BaseModel, Field
import uvicorn

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

# Pydantic Models
class DocumentRequest(BaseModel):
    """Request model for document generation."""
    query: str = Field(..., description="Business query for document generation")
    business_area: Optional[str] = Field(None, description="Specific business area")
    document_type: Optional[str] = Field(None, description="Type of document to generate")
    priority: int = Field(1, ge=1, le=5, description="Processing priority (1-5)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DocumentResponse(BaseModel):
    """Response model for document generation."""
    task_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None

class TaskStatus(BaseModel):
    """Task status response model."""
    task_id: str
    status: str
    progress: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BULSystem:
    """Main BUL system class - consolidated and clean."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL - Business Universal Language",
            description="AI-powered document generation system for SMEs",
            version="3.0.0"
        )
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.setup_middleware()
        self.setup_routes()
        logger.info("BUL System initialized")
    
    def setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "BUL - Business Universal Language",
                "version": "3.0.0",
                "status": "operational",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_tasks": len(self.tasks)
            }
        
        @self.app.post("/documents/generate", response_model=DocumentResponse)
        async def generate_document(request: DocumentRequest, background_tasks: BackgroundTasks):
            """Generate a document based on the provided query."""
            try:
                task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.tasks)}"
                
                # Initialize task
                self.tasks[task_id] = {
                    "status": "queued",
                    "progress": 0,
                    "request": request.dict(),
                    "created_at": datetime.now(),
                    "result": None,
                    "error": None
                }
                
                # Start background processing
                background_tasks.add_task(self.process_document, task_id, request)
                
                return DocumentResponse(
                    task_id=task_id,
                    status="queued",
                    message="Document generation started",
                    estimated_time=60  # seconds
                )
                
            except Exception as e:
                logger.error(f"Error starting document generation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks/{task_id}/status", response_model=TaskStatus)
        async def get_task_status(task_id: str):
            """Get the status of a document generation task."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            return TaskStatus(
                task_id=task_id,
                status=task["status"],
                progress=task["progress"],
                result=task["result"],
                error=task["error"]
            )
        
        @self.app.get("/tasks")
        async def list_tasks():
            """List all tasks."""
            return {
                "tasks": [
                    {
                        "task_id": task_id,
                        "status": task["status"],
                        "progress": task["progress"],
                        "created_at": task["created_at"].isoformat()
                    }
                    for task_id, task in self.tasks.items()
                ]
            }
        
        @self.app.delete("/tasks/{task_id}")
        async def delete_task(task_id: str):
            """Delete a task."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            del self.tasks[task_id]
            return {"message": "Task deleted successfully"}
    
    async def process_document(self, task_id: str, request: DocumentRequest):
        """Process document generation in the background."""
        try:
            logger.info(f"Starting document processing for task {task_id}")
            
            # Update status
            self.tasks[task_id]["status"] = "processing"
            self.tasks[task_id]["progress"] = 10
            
            # Simulate document analysis
            await asyncio.sleep(1)
            self.tasks[task_id]["progress"] = 30
            
            # Simulate AI processing
            await asyncio.sleep(2)
            self.tasks[task_id]["progress"] = 60
            
            # Simulate document generation
            await asyncio.sleep(2)
            self.tasks[task_id]["progress"] = 90
            
            # Generate mock result
            result = {
                "document_id": f"doc_{task_id}",
                "title": f"Generated Document for: {request.query[:50]}...",
                "content": f"This is a generated document based on your query: '{request.query}'. "
                          f"Business area: {request.business_area or 'General'}. "
                          f"Document type: {request.document_type or 'Report'}.",
                "format": "markdown",
                "word_count": 150,
                "generated_at": datetime.now().isoformat(),
                "business_area": request.business_area,
                "document_type": request.document_type
            }
            
            # Complete task
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["progress"] = 100
            self.tasks[task_id]["result"] = result
            
            logger.info(f"Document processing completed for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error processing document for task {task_id}: {e}")
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the BUL system."""
        logger.info(f"Starting BUL system on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BUL - Business Universal Language")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run system
    system = BULSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()


