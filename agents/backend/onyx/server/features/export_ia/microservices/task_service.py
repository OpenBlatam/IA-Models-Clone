"""
Task Service - Microservice for task management and tracking.
"""

import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
import uvicorn

from services.core import BaseService
from services.communication import get_message_bus, EventPublisher, EventSubscriber
from services.discovery import get_service_discovery
from src.core.task_manager import TaskManager
from src.core.config import SystemConfig

logger = logging.getLogger(__name__)


class TaskService(BaseService):
    """Microservice for handling task management and tracking."""
    
    def __init__(self, host: str = "localhost", port: int = 8003):
        super().__init__("task-service", "1.0.0", host, port)
        self.app = FastAPI(title="Task Service", version="1.0.0")
        self.message_bus = get_message_bus()
        self.service_discovery = get_service_discovery()
        self.event_publisher = EventPublisher(self.message_bus, "task-service")
        self.event_subscriber = EventSubscriber(self.message_bus, "task-service")
        
        # Task management
        self.task_manager = None
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup API routes for the task service."""
        
        @self.app.get("/health")
        async def health():
            return await self.health_check()
        
        @self.app.get("/tasks/{task_id}/status")
        async def get_task_status(task_id: str):
            """Get task status."""
            try:
                status = await self.task_manager.get_task_status(task_id)
                if status is None:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                return status
                
            except Exception as e:
                logger.error(f"Status request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/tasks/{task_id}")
        async def cancel_task(task_id: str):
            """Cancel a task."""
            try:
                success = await self.task_manager.cancel_task(task_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
                
                # Publish event
                await self.event_publisher.publish_event(
                    "task.cancelled",
                    {"task_id": task_id}
                )
                
                return {"message": "Task cancelled successfully"}
                
            except Exception as e:
                logger.error(f"Cancel request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks/{task_id}/download")
        async def download_task_file(task_id: str):
            """Get download information for a task."""
            try:
                status = await self.task_manager.get_task_status(task_id)
                if status is None:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                if status.get("status") != "completed":
                    raise HTTPException(status_code=400, detail="Task not completed")
                
                file_path = status.get("file_path")
                if not file_path:
                    raise HTTPException(status_code=404, detail="File not found")
                
                return {
                    "file_path": file_path,
                    "filename": file_path.split("/")[-1],
                    "file_size": status.get("file_size")
                }
                
            except Exception as e:
                logger.error(f"Download request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/statistics")
        async def get_statistics():
            """Get task statistics."""
            try:
                stats = self.task_manager.get_statistics()
                return {
                    "total_tasks": stats.total_tasks,
                    "active_tasks": stats.active_tasks,
                    "completed_tasks": stats.completed_tasks,
                    "failed_tasks": stats.failed_tasks,
                    "format_distribution": stats.format_distribution,
                    "quality_distribution": stats.quality_distribution,
                    "average_quality_score": stats.average_quality_score,
                    "average_processing_time": stats.average_processing_time,
                    "total_processing_time": stats.total_processing_time
                }
                
            except Exception as e:
                logger.error(f"Statistics request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks")
        async def list_tasks():
            """List all tasks."""
            try:
                # Get active tasks
                active_tasks = list(self.task_manager.active_tasks.values())
                
                # Get completed tasks
                completed_tasks = list(self.task_manager.completed_tasks.values())
                
                return {
                    "active_tasks": [
                        {
                            "id": task.id,
                            "status": task.status.value,
                            "format": task.config.format.value,
                            "created_at": task.created_at.isoformat(),
                            "progress": task.progress
                        }
                        for task in active_tasks
                    ],
                    "completed_tasks": [
                        {
                            "id": result.task_id,
                            "status": "completed" if result.success else "failed",
                            "file_path": result.file_path,
                            "quality_score": result.quality_score,
                            "processing_time": result.processing_time
                        }
                        for result in completed_tasks
                    ]
                }
                
            except Exception as e:
                logger.error(f"List tasks request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _start(self) -> None:
        """Start the task service."""
        # Initialize task manager
        system_config = SystemConfig()
        self.task_manager = TaskManager(system_config)
        await self.task_manager.start()
        
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
        await self.message_bus.register_request_handler("status", self._handle_status_request)
        await self.message_bus.register_request_handler("cancel", self._handle_cancel_request)
        await self.message_bus.register_request_handler("download", self._handle_download_request)
        await self.message_bus.register_request_handler("statistics", self._handle_statistics_request)
        
        # Subscribe to events
        await self.event_subscriber.subscribe("export.created", self._handle_export_created)
        await self.event_subscriber.subscribe("export.completed", self._handle_export_completed)
        await self.event_subscriber.subscribe("export.failed", self._handle_export_failed)
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        
        logger.info(f"Task Service started on {self.host}:{self.port}")
    
    async def _stop(self) -> None:
        """Stop the task service."""
        # Stop server
        if hasattr(self, '_server_task'):
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        # Stop task manager
        if self.task_manager:
            await self.task_manager.stop()
        
        # Stop service discovery
        await self.service_discovery.stop()
        
        # Stop message bus
        await self.message_bus.stop()
        
        logger.info("Task Service stopped")
    
    async def _handle_status_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status request via message bus."""
        try:
            task_id = payload.get("task_id")
            if not task_id:
                return {"error": "task_id is required"}
            
            status = await self.task_manager.get_task_status(task_id)
            if status is None:
                return {"error": "Task not found"}
            
            return status
            
        except Exception as e:
            logger.error(f"Status request failed: {e}")
            return {"error": str(e)}
    
    async def _handle_cancel_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cancel request via message bus."""
        try:
            task_id = payload.get("task_id")
            if not task_id:
                return {"error": "task_id is required"}
            
            success = await self.task_manager.cancel_task(task_id)
            if not success:
                return {"error": "Task not found or cannot be cancelled"}
            
            # Publish event
            await self.event_publisher.publish_event(
                "task.cancelled",
                {"task_id": task_id}
            )
            
            return {"message": "Task cancelled successfully"}
            
        except Exception as e:
            logger.error(f"Cancel request failed: {e}")
            return {"error": str(e)}
    
    async def _handle_download_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle download request via message bus."""
        try:
            task_id = payload.get("task_id")
            if not task_id:
                return {"error": "task_id is required"}
            
            status = await self.task_manager.get_task_status(task_id)
            if status is None:
                return {"error": "Task not found"}
            
            if status.get("status") != "completed":
                return {"error": "Task not completed"}
            
            file_path = status.get("file_path")
            if not file_path:
                return {"error": "File not found"}
            
            return {
                "file_path": file_path,
                "filename": file_path.split("/")[-1],
                "file_size": status.get("file_size")
            }
            
        except Exception as e:
            logger.error(f"Download request failed: {e}")
            return {"error": str(e)}
    
    async def _handle_statistics_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistics request via message bus."""
        try:
            stats = self.task_manager.get_statistics()
            return {
                "total_tasks": stats.total_tasks,
                "active_tasks": stats.active_tasks,
                "completed_tasks": stats.completed_tasks,
                "failed_tasks": stats.failed_tasks,
                "format_distribution": stats.format_distribution,
                "quality_distribution": stats.quality_distribution,
                "average_quality_score": stats.average_quality_score,
                "average_processing_time": stats.average_processing_time,
                "total_processing_time": stats.total_processing_time
            }
            
        except Exception as e:
            logger.error(f"Statistics request failed: {e}")
            return {"error": str(e)}
    
    async def _handle_export_created(self, payload: Dict[str, Any]) -> None:
        """Handle export created event."""
        task_id = payload.get("task_id")
        logger.info(f"Export task created: {task_id}")
    
    async def _handle_export_completed(self, payload: Dict[str, Any]) -> None:
        """Handle export completed event."""
        task_id = payload.get("task_id")
        logger.info(f"Export task completed: {task_id}")
    
    async def _handle_export_failed(self, payload: Dict[str, Any]) -> None:
        """Handle export failed event."""
        task_id = payload.get("task_id")
        error = payload.get("error")
        logger.error(f"Export task failed: {task_id} - {error}")


if __name__ == "__main__":
    # Create and start the task service
    service = TaskService()
    
    async def main():
        await service.start()
        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await service.stop()
    
    asyncio.run(main())




