"""
Task management system for Export IA.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging

from .models import ExportTask, ExportResult, TaskStatus, ExportStatistics
from .config import SystemConfig

logger = logging.getLogger(__name__)


class TaskManager:
    """Manages export tasks and their lifecycle."""
    
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.active_tasks: Dict[str, ExportTask] = {}
        self.completed_tasks: Dict[str, ExportResult] = {}
        self.task_queue = asyncio.Queue(maxsize=system_config.max_concurrent_tasks)
        self.executor = ThreadPoolExecutor(max_workers=system_config.max_concurrent_tasks)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the task manager."""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Task manager started")
    
    async def stop(self):
        """Stop the task manager."""
        if self._running:
            self._running = False
            if self._worker_task:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
            self.executor.shutdown(wait=True)
            logger.info("Task manager stopped")
    
    async def submit_task(
        self,
        content: Dict[str, Any],
        config: Any,  # ExportConfig
        output_path: Optional[str] = None
    ) -> str:
        """
        Submit a new export task.
        
        Args:
            content: Document content to export
            config: Export configuration
            output_path: Optional output file path
            
        Returns:
            Task ID for tracking the export process
        """
        task_id = str(uuid.uuid4())
        
        # Create export task
        task = ExportTask(
            id=task_id,
            content=content,
            config=config,
            output_path=output_path
        )
        
        self.active_tasks[task_id] = task
        
        # Add to queue for processing
        await self.task_queue.put(task)
        
        logger.info(f"Task submitted: {task_id} - {config.format.value} format")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an export task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status.value,
                "format": task.config.format.value,
                "document_type": task.config.document_type.value,
                "quality_level": task.config.quality_level.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "progress": task.progress,
                "error": task.error
            }
        elif task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "id": task_id,
                "status": "completed",
                "success": result.success,
                "file_path": result.file_path,
                "file_size": result.file_size,
                "quality_score": result.quality_score,
                "processing_time": result.processing_time,
                "error": result.error,
                "warnings": result.warnings
            }
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                
                # Create cancelled result
                result = ExportResult(
                    task_id=task_id,
                    success=False,
                    error="Task cancelled by user"
                )
                self.completed_tasks[task_id] = result
                
                # Remove from active tasks
                del self.active_tasks[task_id]
                
                logger.info(f"Task cancelled: {task_id}")
                return True
        return False
    
    def get_statistics(self) -> ExportStatistics:
        """Get export statistics."""
        active_count = len(self.active_tasks)
        completed_count = len(self.completed_tasks)
        failed_count = sum(1 for r in self.completed_tasks.values() if not r.success)
        
        format_counts = {}
        quality_counts = {}
        total_processing_time = 0.0
        
        for result in self.completed_tasks.values():
            format_counts[result.format] = format_counts.get(result.format, 0) + 1
            total_processing_time += result.processing_time
        
        for task in self.active_tasks.values():
            quality_counts[task.config.quality_level.value] = quality_counts.get(task.config.quality_level.value, 0) + 1
        
        avg_quality_score = 0.0
        avg_processing_time = 0.0
        
        if completed_count > 0:
            avg_quality_score = sum(r.quality_score for r in self.completed_tasks.values()) / completed_count
            avg_processing_time = total_processing_time / completed_count
        
        return ExportStatistics(
            total_tasks=active_count + completed_count,
            active_tasks=active_count,
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            format_distribution=format_counts,
            quality_distribution=quality_counts,
            average_quality_score=avg_quality_score,
            average_processing_time=avg_processing_time,
            total_processing_time=total_processing_time
        )
    
    async def _worker_loop(self):
        """Main worker loop for processing tasks."""
        while self._running:
            try:
                # Wait for a task with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Process the task
                await self._process_task(task)
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_task(self, task: ExportTask):
        """Process a single export task."""
        start_time = datetime.now()
        
        try:
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.started_at = start_time
            task.progress = 0.1
            
            # Import here to avoid circular imports
            from .quality_manager import QualityManager
            from ..exporters import ExporterFactory
            
            # Initialize components
            quality_manager = QualityManager()
            exporter = ExporterFactory.create_exporter(task.config.format)
            
            # Process content for quality
            task.progress = 0.3
            processed_content = await quality_manager.process_content_for_quality(
                task.content, task.config
            )
            
            # Generate output path if not provided
            if not task.output_path:
                task.output_path = self._generate_output_path(task)
            
            # Ensure output directory exists
            import os
            os.makedirs(os.path.dirname(task.output_path), exist_ok=True)
            
            # Export document
            task.progress = 0.5
            result = await exporter.export(processed_content, task.config, task.output_path)
            
            # Calculate quality score
            task.progress = 0.8
            quality_score = await quality_manager.calculate_quality_score(result, task.config)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create export result
            export_result = ExportResult(
                task_id=task.id,
                success=True,
                file_path=task.output_path,
                file_size=os.path.getsize(task.output_path) if os.path.exists(task.output_path) else 0,
                format=task.config.format.value,
                quality_score=quality_score,
                processing_time=processing_time,
                metadata={
                    "document_type": task.config.document_type.value,
                    "quality_level": task.config.quality_level.value,
                    "exported_at": datetime.now().isoformat()
                }
            )
            
            # Store result
            self.completed_tasks[task.id] = export_result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            task.file_path = task.output_path
            task.file_size = export_result.file_size
            task.progress = 1.0
            
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            logger.info(f"Task completed: {task.id} - Quality score: {quality_score:.2f}")
            
        except Exception as e:
            logger.error(f"Task failed: {task.id} - {e}")
            
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            task.progress = 0.0
            
            # Create failed result
            export_result = ExportResult(
                task_id=task.id,
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.completed_tasks[task.id] = export_result
            
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    def _generate_output_path(self, task: ExportTask) -> str:
        """Generate output file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{task.id}_{timestamp}.{task.config.format.value}"
        return os.path.join(self.system_config.output_directory, filename)




