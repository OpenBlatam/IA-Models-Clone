"""
Continuous Processor for BUL System
===================================

Handles continuous processing of business queries and document generation.
Keeps working until manually stopped, processing queries in real-time.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import signal
import sys

from .bul_engine import BULEngine, ProcessingTask, ProcessingResult
from ..config.bul_config import BULConfig

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Metrics for continuous processing."""
    start_time: datetime
    total_queries_processed: int = 0
    total_documents_generated: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 0.0
    errors_count: int = 0
    last_activity: Optional[datetime] = None

class ContinuousProcessor:
    """
    Continuous processor that keeps working until manually stopped.
    """
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.engine = BULEngine(self.config)
        
        # Processing state
        self.is_running = False
        self.should_stop = False
        self.metrics = ProcessingMetrics(start_time=datetime.now())
        
        # Callbacks
        self.on_document_generated: Optional[Callable] = None
        self.on_task_completed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("Continuous Processor initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Start continuous processing."""
        if self.is_running:
            logger.warning("Continuous processor is already running")
            return
        
        self.is_running = True
        self.should_stop = False
        self.metrics.start_time = datetime.now()
        
        logger.info("Starting continuous processing mode...")
        logger.info(f"Enabled business areas: {', '.join(self.config.sme.enabled_areas)}")
        logger.info("Press Ctrl+C to stop processing")
        
        try:
            await self._main_processing_loop()
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
            if self.on_error:
                await self._safe_callback(self.on_error, e)
        finally:
            self.is_running = False
            logger.info("Continuous processing stopped")
            await self._print_final_metrics()
    
    async def _main_processing_loop(self):
        """Main processing loop."""
        while not self.should_stop:
            try:
                # Check for new tasks
                await self._process_pending_tasks()
                
                # Update metrics
                await self._update_metrics()
                
                # Check for idle timeout
                if self._should_enter_idle_mode():
                    await self._handle_idle_mode()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info("Processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                if self.on_error:
                    await self._safe_callback(self.on_error, e)
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_pending_tasks(self):
        """Process any pending tasks."""
        stats = self.engine.get_processing_stats()
        
        if stats["queued_tasks"] > 0 or stats["active_tasks"] > 0:
            self.metrics.last_activity = datetime.now()
            
            # The engine handles its own processing, we just monitor
            await asyncio.sleep(0.1)
    
    async def _update_metrics(self):
        """Update processing metrics."""
        stats = self.engine.get_processing_stats()
        
        # Update metrics
        self.metrics.total_queries_processed = stats["total_tasks"]
        self.metrics.average_processing_time = stats["average_processing_time"]
        
        # Calculate success rate
        if stats["total_tasks"] > 0:
            self.metrics.success_rate = (
                stats["completed_tasks"] / stats["total_tasks"]
            ) * 100
        
        # Count errors
        self.metrics.errors_count = stats["failed_tasks"]
    
    def _should_enter_idle_mode(self) -> bool:
        """Check if should enter idle mode."""
        if not self.metrics.last_activity:
            return False
        
        idle_timeout = timedelta(minutes=30)  # 30 minutes of inactivity
        return datetime.now() - self.metrics.last_activity > idle_timeout
    
    async def _handle_idle_mode(self):
        """Handle idle mode when no tasks are being processed."""
        logger.info("Entering idle mode - no active tasks")
        
        # In idle mode, we can perform maintenance tasks
        await self._perform_maintenance()
        
        # Wait longer in idle mode
        await asyncio.sleep(10)
    
    async def _perform_maintenance(self):
        """Perform maintenance tasks during idle time."""
        try:
            # Clean up old completed tasks (keep last 100)
            await self._cleanup_old_tasks()
            
            # Update document processor cache
            await self.engine.document_processor.cleanup_cache()
            
            logger.debug("Maintenance tasks completed")
            
        except Exception as e:
            logger.error(f"Error during maintenance: {e}")
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed tasks to prevent memory buildup."""
        if len(self.engine.completed_tasks) > 100:
            # Keep only the most recent 100 tasks
            sorted_tasks = sorted(
                self.engine.completed_tasks.items(),
                key=lambda x: x[1].processing_time,
                reverse=True
            )
            
            # Remove old tasks
            tasks_to_remove = sorted_tasks[100:]
            for task_id, _ in tasks_to_remove:
                del self.engine.completed_tasks[task_id]
            
            logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute a callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    async def _print_final_metrics(self):
        """Print final processing metrics."""
        runtime = datetime.now() - self.metrics.start_time
        
        logger.info("=== Final Processing Metrics ===")
        logger.info(f"Total runtime: {runtime}")
        logger.info(f"Total queries processed: {self.metrics.total_queries_processed}")
        logger.info(f"Total documents generated: {self.metrics.total_documents_generated}")
        logger.info(f"Average processing time: {self.metrics.average_processing_time:.2f}s")
        logger.info(f"Success rate: {self.metrics.success_rate:.1f}%")
        logger.info(f"Errors: {self.metrics.errors_count}")
        logger.info("================================")
    
    async def submit_query(self, query: str, priority: int = 3) -> str:
        """Submit a query for processing."""
        if not self.is_running:
            raise RuntimeError("Continuous processor is not running")
        
        task_id = await self.engine.submit_query(query, priority)
        logger.info(f"Query submitted for processing: {task_id}")
        
        return task_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the continuous processor."""
        stats = self.engine.get_processing_stats()
        
        return {
            "is_running": self.is_running,
            "should_stop": self.should_stop,
            "metrics": {
                "start_time": self.metrics.start_time.isoformat(),
                "total_queries_processed": self.metrics.total_queries_processed,
                "total_documents_generated": self.metrics.total_documents_generated,
                "average_processing_time": self.metrics.average_processing_time,
                "success_rate": self.metrics.success_rate,
                "errors_count": self.metrics.errors_count,
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            },
            "engine_stats": stats
        }
    
    def stop(self):
        """Stop continuous processing."""
        logger.info("Stop requested for continuous processor")
        self.should_stop = True
    
    def set_document_callback(self, callback: Callable):
        """Set callback for when documents are generated."""
        self.on_document_generated = callback
    
    def set_task_callback(self, callback: Callable):
        """Set callback for when tasks are completed."""
        self.on_task_completed = callback
    
    def set_error_callback(self, callback: Callable):
        """Set callback for when errors occur."""
        self.on_error = callback

# Global processor instance
_global_processor: Optional[ContinuousProcessor] = None

def get_global_processor() -> ContinuousProcessor:
    """Get the global continuous processor instance."""
    global _global_processor
    if _global_processor is None:
        _global_processor = ContinuousProcessor()
    return _global_processor

async def start_global_processor():
    """Start the global continuous processor."""
    processor = get_global_processor()
    await processor.start()

def stop_global_processor():
    """Stop the global continuous processor."""
    processor = get_global_processor()
    processor.stop()

