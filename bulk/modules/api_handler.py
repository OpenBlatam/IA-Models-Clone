"""
API Handler Module
=================

Handles API requests and responses for the BUL system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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

class APIHandler:
    """Handles API operations for the BUL system."""
    
    def __init__(self, 
                 document_processor,
                 query_analyzer,
                 agent_manager):
        self.document_processor = document_processor
        self.query_analyzer = query_analyzer
        self.agent_manager = agent_manager
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    async def generate_document(self, request: DocumentRequest) -> DocumentResponse:
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
            asyncio.create_task(self._process_document(task_id, request))
            
            return DocumentResponse(
                task_id=task_id,
                status="queued",
                message="Document generation started",
                estimated_time=60  # seconds
            )
            
        except Exception as e:
            logger.error(f"Error starting document generation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_task_status(self, task_id: str) -> TaskStatus:
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
    
    def list_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
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
    
    def delete_task(self, task_id: str) -> Dict[str, str]:
        """Delete a task."""
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        del self.tasks[task_id]
        return {"message": "Task deleted successfully"}
    
    async def _process_document(self, task_id: str, request: DocumentRequest):
        """Process document generation in the background."""
        
        try:
            logger.info(f"Starting document processing for task {task_id}")
            
            # Update status
            self.tasks[task_id]["status"] = "processing"
            self.tasks[task_id]["progress"] = 10
            
            # Analyze query
            analysis = self.query_analyzer.analyze(request.query)
            self.tasks[task_id]["progress"] = 30
            
            # Determine business area and document type
            business_area = request.business_area or analysis.primary_area
            document_type = request.document_type or analysis.document_types[0]
            
            # Process with appropriate agent
            agent_result = await self.agent_manager.process_with_agent(
                business_area, request.query, document_type
            )
            self.tasks[task_id]["progress"] = 60
            
            # Generate document
            document = await self.document_processor.generate_document(
                query=request.query,
                business_area=business_area,
                document_type=document_type,
                metadata={
                    **request.metadata or {},
                    'analysis': {
                        'primary_area': analysis.primary_area,
                        'complexity': analysis.complexity.value,
                        'confidence': analysis.confidence
                    },
                    'agent_result': agent_result
                }
            )
            self.tasks[task_id]["progress"] = 90
            
            # Complete task
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["progress"] = 100
            self.tasks[task_id]["result"] = document
            
            logger.info(f"Document processing completed for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error processing document for task {task_id}: {e}")
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "system": "BUL - Business Universal Language",
            "version": "3.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "available_areas": self.agent_manager.get_available_areas(),
            "active_tasks": len(self.tasks)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_tasks": len(self.tasks),
            "components": {
                "document_processor": "operational",
                "query_analyzer": "operational",
                "agent_manager": "operational"
            }
        }

