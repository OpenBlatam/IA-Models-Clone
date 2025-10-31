"""
Export Service - Microservice for document export operations.
"""

import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
import uvicorn

from services.core import BaseService
from services.communication import get_message_bus, EventPublisher
from services.discovery import get_service_discovery
from src.core.models import ExportConfig, ExportFormat, DocumentType, QualityLevel
from src.exporters import ExporterFactory

logger = logging.getLogger(__name__)


class ExportService(BaseService):
    """Microservice for handling document export operations."""
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        super().__init__("export-service", "1.0.0", host, port)
        self.app = FastAPI(title="Export Service", version="1.0.0")
        self.message_bus = get_message_bus()
        self.service_discovery = get_service_discovery()
        self.event_publisher = EventPublisher(self.message_bus, "export-service")
        
        # Export engine
        self.export_engine = None
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup API routes for the export service."""
        
        @self.app.get("/health")
        async def health():
            return await self.health_check()
        
        @self.app.post("/export")
        async def export_document(request: Dict[str, Any]):
            """Export a document."""
            try:
                content = request.get("content")
                config_data = request.get("config", {})
                
                if not content:
                    raise HTTPException(status_code=400, detail="Content is required")
                
                # Create export configuration
                config = ExportConfig(
                    format=ExportFormat(config_data.get("format", "pdf")),
                    document_type=DocumentType(config_data.get("document_type", "report")),
                    quality_level=QualityLevel(config_data.get("quality_level", "professional"))
                )
                
                # Export document
                task_id = await self.export_engine.export_document(content, config)
                
                # Publish event
                await self.event_publisher.publish_event(
                    "export.created",
                    {"task_id": task_id, "format": config.format.value}
                )
                
                return {"task_id": task_id, "status": "pending"}
                
            except Exception as e:
                logger.error(f"Export failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/formats")
        async def get_supported_formats():
            """Get supported export formats."""
            return self.export_engine.list_supported_formats()
        
        @self.app.get("/statistics")
        async def get_statistics():
            """Get export statistics."""
            return self.export_engine.get_export_statistics()
    
    async def _start(self) -> None:
        """Start the export service."""
        # Initialize export engine
        from src.core.engine import ExportIAEngine
        self.export_engine = ExportIAEngine()
        await self.export_engine.initialize()
        
        # Start message bus
        await self.message_bus.start()
        
        # Start service discovery
        await self.service_discovery.start()
        
        # Register with service discovery
        await self.service_discovery.register_service(
            name=self.name,
            host=self.host,
            port=self.port,
            health_url="/health",
            api_url="/"
        )
        
        # Register message handlers
        await self.message_bus.register_request_handler("export", self._handle_export_request)
        await self.message_bus.register_request_handler("formats", self._handle_formats_request)
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        
        logger.info(f"Export Service started on {self.host}:{self.port}")
    
    async def _stop(self) -> None:
        """Stop the export service."""
        # Stop server
        if hasattr(self, '_server_task'):
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        # Stop export engine
        if self.export_engine:
            await self.export_engine.shutdown()
        
        # Stop service discovery
        await self.service_discovery.stop()
        
        # Stop message bus
        await self.message_bus.stop()
        
        logger.info("Export Service stopped")
    
    async def _handle_export_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle export request via message bus."""
        try:
            content = payload.get("content")
            config_data = payload.get("config", {})
            
            # Create export configuration
            config = ExportConfig(
                format=ExportFormat(config_data.get("format", "pdf")),
                document_type=DocumentType(config_data.get("document_type", "report")),
                quality_level=QualityLevel(config_data.get("quality_level", "professional"))
            )
            
            # Export document
            task_id = await self.export_engine.export_document(content, config)
            
            # Publish event
            await self.event_publisher.publish_event(
                "export.created",
                {"task_id": task_id, "format": config.format.value}
            )
            
            return {"task_id": task_id, "status": "pending"}
            
        except Exception as e:
            logger.error(f"Export request failed: {e}")
            return {"error": str(e)}
    
    async def _handle_formats_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle formats request via message bus."""
        try:
            return self.export_engine.list_supported_formats()
        except Exception as e:
            logger.error(f"Formats request failed: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Create and start the export service
    service = ExportService()
    
    async def main():
        await service.start()
        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await service.stop()
    
    asyncio.run(main())




