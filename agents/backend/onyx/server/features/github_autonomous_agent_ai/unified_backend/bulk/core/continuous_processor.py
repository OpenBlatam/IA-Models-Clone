"""
Continuous Processor for BUL System
===================================

Handles continuous processing of business queries and document generation.
Keeps working until manually stopped, processing queries in real-time.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from ..config.bul_config import BULConfig
from .truthgpt_bulk_processor import TruthGPTBulkProcessor
from .processor_config import ProcessorConfig, DEFAULT_CONFIG
from .processor_metrics import ProcessingMetrics
from .callback_manager import CallbackManager
from .maintenance_manager import MaintenanceManager
from .idle_manager import IdleManager
from .signal_handler import SignalHandler

logger = logging.getLogger(__name__)

class ContinuousProcessor:
    """
    Continuous processor that keeps working until manually stopped.
    """
    
    def __init__(
        self, 
        config: Optional[BULConfig] = None,
        processor_config: Optional[ProcessorConfig] = None
    ):
        self.config = config or BULConfig()
        self.processor_config = processor_config or DEFAULT_CONFIG
        self.processor = TruthGPTBulkProcessor(self.config)
        
        self.is_running = False
        self.should_stop = False
        self.metrics = ProcessingMetrics(start_time=datetime.now())
        self.callbacks = CallbackManager()
        self.maintenance = MaintenanceManager(self.processor_config, self.processor)
        self.idle_manager = IdleManager(self.processor_config)
        self.signal_handler = SignalHandler(shutdown_callback=self._handle_shutdown_signal)
        
        self.signal_handler.setup()
        
        self.processor.set_document_callback(self._on_document_generated)
        self.processor.set_error_callback(self._on_error)
        
        logger.info("Continuous Processor initialized")
    
    def _handle_shutdown_signal(self) -> None:
        """Handle shutdown signal from signal handler."""
        self.should_stop = True
    
    async def _on_document_generated(self, task: Any, processed_doc: Any) -> None:
        """Internal callback wrapper for document generation."""
        await self.callbacks.execute_document_callback(task, processed_doc)
    
    async def _on_error(self, *args: Any) -> None:
        """Internal callback wrapper for errors."""
        await self.callbacks.execute_error_callback(*args)
    
    async def start(self) -> None:
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
        
        processor_task = asyncio.create_task(self.processor.start_continuous_processing())
        
        try:
            await self._main_processing_loop()
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}", exc_info=True)
            await self.callbacks.execute_error_callback(e)
        finally:
            self.is_running = False
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
            self.signal_handler.restore()
            logger.info("Continuous processing stopped")
            await self._print_final_metrics()
    
    async def _main_processing_loop(self) -> None:
        """Main processing loop."""
        while not self.should_stop:
            try:
                await self._process_pending_tasks()
                await self._update_metrics()
                
                if self._should_enter_idle_mode():
                    await self._handle_idle_mode()
                
                await asyncio.sleep(self.processor_config.loop_sleep_seconds)
                
            except asyncio.CancelledError:
                logger.info("Processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await self.callbacks.execute_error_callback(e)
                await asyncio.sleep(self.processor_config.retry_sleep_seconds)
    
    async def _process_pending_tasks(self) -> None:
        """Process any pending tasks."""
        stats = self.processor.get_processing_stats()
        
        if stats.get("queued_tasks", 0) > 0 or stats.get("active_tasks", 0) > 0:
            self.idle_manager.record_activity(self.metrics)
            await asyncio.sleep(self.processor_config.task_monitor_sleep_seconds)
    
    async def _update_metrics(self) -> None:
        """Update processing metrics."""
        stats = self.processor.get_processing_stats()
        self.metrics.update_from_stats(stats)
    
    def _should_enter_idle_mode(self) -> bool:
        """Check if should enter idle mode."""
        return self.idle_manager.should_enter_idle_mode(self.metrics.last_activity)
    
    async def _handle_idle_mode(self) -> None:
        """Handle idle mode when no tasks are being processed."""
        logger.info("Entering idle mode - no active tasks")
        await self.maintenance.perform_maintenance()
        await asyncio.sleep(self.processor_config.maintenance_sleep_seconds)
    
    async def _print_final_metrics(self) -> None:
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
    
    async def submit_query(
        self, 
        query: str, 
        document_types: Optional[List[str]] = None,
        business_areas: Optional[List[str]] = None,
        priority: int = 3
    ) -> str:
        """Submit a query for processing."""
        if not self.is_running:
            raise RuntimeError("Continuous processor is not running")
        
        if document_types is None:
            document_types = ["strategy", "plan", "analysis"]
        if business_areas is None:
            business_areas = self.config.sme.enabled_areas[:3]
        
        request_id = await self.processor.submit_bulk_request(
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=10,
            continuous_mode=True,
            priority=priority
        )
        logger.info(f"Query submitted for processing: {request_id}")
        
        return request_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the continuous processor."""
        stats = self.processor.get_processing_stats()
        
        return {
            "is_running": self.is_running,
            "should_stop": self.should_stop,
            "metrics": self.metrics.to_dict(),
            "processor_stats": stats
        }
    
    def stop(self) -> None:
        """Stop continuous processing."""
        logger.info("Stop requested for continuous processor")
        self.should_stop = True
        self.processor.stop_processing()
    
    def set_document_callback(self, callback: Callable[[Any, Any], None]) -> None:
        """Set callback for when documents are generated."""
        self.callbacks.set_document_callback(callback)
    
    def set_task_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for when tasks are completed."""
        self.callbacks.set_task_callback(callback)
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for when errors occur."""
        self.callbacks.set_error_callback(callback)

