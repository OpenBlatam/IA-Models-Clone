"""
Simple, practical FastAPI application for Export IA.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from ..core.simple_engine import get_export_engine
from ..core.models import ExportConfig, ExportFormat, DocumentType, QualityLevel

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Export IA API",
        description="Simple document export API",
        version="2.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Get engine instance
    engine = get_export_engine()
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize engine on startup."""
        await engine.initialize()
        logger.info("Export IA API started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown engine on shutdown."""
        await engine.shutdown()
        logger.info("Export IA API shutdown")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Export IA API",
            "version": "2.0.0",
            "status": "running",
            "formats": engine.list_supported_formats()
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.post("/export")
    async def export_document(request: Request):
        """Export a document."""
        try:
            data = await request.json()
            content = data.get("content")
            format_name = data.get("format", "pdf")
            document_type = data.get("document_type", "report")
            quality_level = data.get("quality_level", "professional")
            
            if not content:
                raise HTTPException(status_code=400, detail="Content is required")
            
            # Create config
            config = ExportConfig(
                format=ExportFormat(format_name),
                document_type=DocumentType(document_type),
                quality_level=QualityLevel(quality_level)
            )
            
            # Export document
            task_id = await engine.export_document(content, config)
            
            return {
                "task_id": task_id,
                "status": "pending",
                "message": "Export task created"
            }
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/export/{task_id}/status")
    async def get_export_status(task_id: str):
        """Get export task status."""
        try:
            status = await engine.get_task_status(task_id)
            if not status:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return status
            
        except Exception as e:
            logger.error(f"Status request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/export/{task_id}/download")
    async def download_export(task_id: str):
        """Download exported file."""
        try:
            status = await engine.get_task_status(task_id)
            if not status:
                raise HTTPException(status_code=404, detail="Task not found")
            
            if status.get("status") != "completed":
                raise HTTPException(status_code=400, detail="Task not completed")
            
            file_path = status.get("file_path")
            if not file_path:
                raise HTTPException(status_code=404, detail="File not found")
            
            return FileResponse(
                path=file_path,
                filename=file_path.split("/")[-1]
            )
            
        except Exception as e:
            logger.error(f"Download request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/validate")
    async def validate_content(request: Request):
        """Validate document content."""
        try:
            data = await request.json()
            content = data.get("content")
            format_name = data.get("format", "pdf")
            document_type = data.get("document_type", "report")
            quality_level = data.get("quality_level", "professional")
            
            if not content:
                raise HTTPException(status_code=400, detail="Content is required")
            
            # Create config
            config = ExportConfig(
                format=ExportFormat(format_name),
                document_type=DocumentType(document_type),
                quality_level=QualityLevel(quality_level)
            )
            
            # Validate content
            result = engine.validate_content(content, config)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/formats")
    async def get_supported_formats():
        """Get supported export formats."""
        return engine.list_supported_formats()
    
    @app.get("/statistics")
    async def get_statistics():
        """Get system statistics."""
        return await engine.get_export_statistics()
    
    @app.get("/templates/{doc_type}")
    async def get_template(doc_type: str):
        """Get document template."""
        try:
            template = engine.get_document_template(DocumentType(doc_type))
            return template
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document type")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




