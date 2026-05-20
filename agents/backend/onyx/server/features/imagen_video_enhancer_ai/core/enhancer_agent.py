"""
Enhancer Agent - Main class for image/video enhancement with SAM3 architecture
=============================================================================
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Consolidated imports
from .imports import (
    # Infrastructure
    OpenRouterClient,
    TruthGPTClient,
    # Configuration
    EnhancerConfig,
    # Core components
    SystemPromptsBuilder,
    TaskManager,
    FileTaskRepository,
    TaskEvent,
    ParallelExecutor,
    ServiceHandler,
    ServiceType,
    TaskCreator,
    BatchProcessor,
    BatchItem,
    BatchResult,
    CacheManager,
    WebhookManager,
    WebhookEvent,
    RetryManager,
    RetryConfig,
    MetricsCollector,
    EventBus,
    Event,
    EventType,
    VideoProcessor,
    # Utilities
    create_output_directories,
    PerformanceMonitor,
    # Constants
    DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_MAX_RETRIES,
    OUTPUT_DIRECTORIES,
    DEFAULT_MAX_PARALLEL_TASKS,
    DEFAULT_OUTPUT_DIR
)

logger = logging.getLogger(__name__)


class EnhancerAgent:
    """
    Autonomous agent for image/video enhancement based on SAM3 architecture.
    
    Features:
    - Continuous 24/7 operation
    - Parallel task execution
    - OpenRouter LLM integration
    - TruthGPT optimization
    - Automatic task management
    - Image and video enhancement services
    """
    
    def __init__(
        self,
        config: Optional[EnhancerConfig] = None,
        max_parallel_tasks: int = DEFAULT_MAX_PARALLEL_TASKS,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        debug: bool = False,
    ):
        """Initialize EnhancerAgent with all components."""
        # Configuration
        self.config = config or EnhancerConfig()
        self.config.validate()
        self.debug = debug
        self.running = False
        
        # Setup output directories
        self._setup_output_directories(output_dir)
        
        # Initialize clients
        self._initialize_clients()
        
        # Initialize managers
        self._initialize_managers(max_parallel_tasks)
        
        # Initialize processors
        self._initialize_processors()
        
        # Initialize service handler
        self._initialize_service_handler()
        
        # Setup event subscriptions
        self._setup_webhook_subscriptions()
        
        logger.info(f"Initialized EnhancerAgent with {max_parallel_tasks} max parallel tasks")
    
    def _setup_output_directories(self, output_dir: str):
        """Setup output directories."""
        self.output_dir = Path(output_dir)
        self.output_dirs = create_output_directories(
            self.output_dir,
            OUTPUT_DIRECTORIES
        )
    
    def _initialize_clients(self):
        """Initialize external API clients."""
        self.openrouter_client = OpenRouterClient(
            api_key=self.config.openrouter.api_key
        )
        self.truthgpt_client = TruthGPTClient(
            config=self.config.truthgpt.to_dict() if self.config.truthgpt else {}
        )
    
    def _initialize_managers(self, max_parallel_tasks: int):
        """Initialize all manager components."""
        # Task manager
        self.task_manager = TaskManager(
            repository=FileTaskRepository(str(self.output_dirs["storage"]))
        )
        
        # Batch processor
        self.batch_processor = BatchProcessor(max_concurrent=max_parallel_tasks)
        
        # Cache manager
        self.cache_manager = CacheManager(
            cache_dir=self.output_dirs["cache"],
            default_ttl_hours=DEFAULT_CACHE_TTL_HOURS
        )
        
        # Webhook manager
        self.webhook_manager = WebhookManager()
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Retry manager
        self.retry_manager = RetryManager(
            config=RetryConfig(
                max_retries=DEFAULT_MAX_RETRIES,
                strategy=RetryConfig.RetryStrategy.EXPONENTIAL_BACKOFF
            )
        )
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Event bus
        self.event_bus = EventBus()
        
        # Parallel executor
        self.parallel_executor = ParallelExecutor(max_workers=max_parallel_tasks)
        
        # Initialize new systems
        self._initialize_security_systems()
    
    def _initialize_processors(self):
        """Initialize processing components."""
        self.video_processor = VideoProcessor()
        self.system_prompts = SystemPromptsBuilder.build_all_prompts()
    
    def _initialize_service_handler(self):
        """Initialize service handler with all dependencies."""
        self.service_handler = ServiceHandler(
            openrouter_client=self.openrouter_client,
            truthgpt_client=self.truthgpt_client,
            config=self.config,
            system_prompts=self.system_prompts,
            video_processor=self.video_processor
        )
    
    def _setup_webhook_subscriptions(self):
        """Setup webhook subscriptions to task events."""
        # Create event handlers
        handlers = {
            TaskEvent.CREATED: self._on_task_created_webhook,
            TaskEvent.STARTED: self._on_task_started_webhook,
            TaskEvent.COMPLETED: self._on_task_completed_webhook,
            TaskEvent.FAILED: self._on_task_failed_webhook,
        }
        
        # Subscribe all handlers
        for event_type, handler in handlers.items():
            self.task_manager.events.subscribe(event_type, handler)
    
    async def _on_task_created_webhook(self, task):
        """Handle task created event for webhooks."""
        await self.webhook_manager.send(
            WebhookEvent.TASK_CREATED,
            {"task_id": task.id, "service_type": task.service_type},
            task_id=task.id
        )
    
    async def _on_task_started_webhook(self, task):
        """Handle task started event for webhooks."""
        await self.webhook_manager.send(
            WebhookEvent.TASK_STARTED,
            {"task_id": task.id, "service_type": task.service_type},
            task_id=task.id
        )
    
    async def _on_task_completed_webhook(self, task):
        """Handle task completed event for webhooks."""
        await self.webhook_manager.send(
            WebhookEvent.TASK_COMPLETED,
            {"task_id": task.id, "result": task.result},
            task_id=task.id
        )
    
    async def _on_task_failed_webhook(self, task):
        """Handle task failed event for webhooks."""
        await self.webhook_manager.send(
            WebhookEvent.TASK_FAILED,
            {"task_id": task.id, "error": task.error},
            task_id=task.id
        )
    
    async def start(self):
        """Start the autonomous agent in continuous operation mode."""
        if self.running:
            logger.warning("Agent is already running")
            return
        
        self.running = True
        logger.info("Starting Enhancer Agent (24/7 mode)")
        
        try:
            # Initialize task manager (load existing tasks)
            await self.task_manager.initialize()
            
            # Start parallel executor
            await self.parallel_executor.start()
            
            # Main event loop for continuous operation
            while self.running:
                try:
                    # Get pending tasks from task manager
                    tasks = await self.task_manager.get_pending_tasks(limit=10)
                    
                    if tasks:
                        # Submit tasks to parallel executor
                        for task in tasks:
                            await self.parallel_executor.submit_task(
                                self._process_task,
                                task
                            )
                    
                    # Wait before next iteration
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(5.0)
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the autonomous agent."""
        logger.info("Stopping Enhancer Agent")
        self.running = False
        await self.parallel_executor.stop()
        await self.openrouter_client.close()
        await self.truthgpt_client.close()
        await self.webhook_manager.close()
    
    async def close(self):
        """Close agent and cleanup resources."""
        await self.stop()
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task."""
        import time
        task_id = task.get("id", "unknown")
        service_type_str = task.get("service_type")
        parameters = task.get("parameters", {})
        
        logger.info(f"Processing task {task_id}: {service_type_str}")
        
        start_time = time.time()
        try:
            # Update task status (webhook will be sent automatically via event subscription)
            await self.task_manager.update_task_status(task_id, "processing")
            
            # Publish event
            await self.event_bus.publish(Event(
                event_type=EventType.TASK_STARTED,
                data={"task_id": task_id, "service_type": service_type_str},
                source="enhancer_agent"
            ))
            
            # Record metric
            self.metrics_collector.increment("tasks.started", tags={"service_type": service_type_str})
            
            # Convert string service type to Enum
            try:
                service_type = ServiceType(service_type_str)
            except ValueError:
                raise ValueError(f"Unknown service type: {service_type_str}")
            
            # Execute service using handler
            result = await self.service_handler.handle(service_type, parameters)
            
            if result.success:
                # Save result (webhook will be sent automatically via event subscription)
                await self.task_manager.complete_task(task_id, result.to_dict())
                
                # Record performance
                duration = time.time() - start_time
                self.performance_monitor.record(service_type_str, duration)
                
                # Record metrics
                self.metrics_collector.record_histogram("task.duration", duration, tags={"service_type": service_type_str})
                self.metrics_collector.increment("tasks.completed", tags={"service_type": service_type_str})
                
                # Publish event
                await self.event_bus.publish(Event(
                    event_type=EventType.TASK_COMPLETED,
                    data={"task_id": task_id, "duration": duration, "service_type": service_type_str},
                    source="enhancer_agent"
                ))
                
                logger.info(f"Task {task_id} completed successfully in {duration:.3f}s")
                return result.to_dict()
            else:
                # Handle service failure (webhook will be sent automatically via event subscription)
                error_msg = result.error or "Unknown error"
                error = Exception(error_msg)
                
                # Try to retry if applicable
                if self.retry_manager.should_retry(task_id, error, 0):
                    try:
                        async def retry_func():
                            return await self.service_handler.handle(service_type, parameters)
                        
                        retry_result = await self.retry_manager.retry_task(
                            task_id=task_id,
                            task_func=retry_func,
                            error=error,
                            attempt_number=0
                        )
                        
                        if retry_result and hasattr(retry_result, 'success') and retry_result.success:
                            await self.task_manager.complete_task(task_id, retry_result.to_dict())
                            duration = time.time() - start_time
                            self.performance_monitor.record(service_type_str, duration)
                            logger.info(f"Task {task_id} completed after retry")
                            return retry_result.to_dict()
                    except Exception as retry_error:
                        logger.warning(f"Retry failed for task {task_id}: {retry_error}")
                
                await self.task_manager.fail_task(task_id, error_msg)
                
                # Record performance even for failures
                duration = time.time() - start_time
                self.performance_monitor.record(f"{service_type_str}_failed", duration)
                
                # Record metrics
                self.metrics_collector.increment("tasks.failed", tags={"service_type": service_type_str})
                
                # Publish event
                await self.event_bus.publish(Event(
                    event_type=EventType.TASK_FAILED,
                    data={"task_id": task_id, "error": error_msg, "service_type": service_type_str},
                    source="enhancer_agent"
                ))
                
                logger.error(f"Task {task_id} failed: {error_msg}")
                raise error
            
        except Exception as e:
            # Try to retry if applicable
            if self.retry_manager.should_retry(task_id, e, 0):
                try:
                    async def retry_func():
                        return await self.service_handler.handle(service_type, parameters)
                    
                    retry_result = await self.retry_manager.retry_task(
                        task_id=task_id,
                        task_func=retry_func,
                        error=e,
                        attempt_number=0
                    )
                    
                    if retry_result and hasattr(retry_result, 'success') and retry_result.success:
                        await self.task_manager.complete_task(task_id, retry_result.to_dict())
                        duration = time.time() - start_time
                        self.performance_monitor.record(service_type_str, duration)
                        logger.info(f"Task {task_id} completed after retry")
                        return retry_result.to_dict()
                except Exception as retry_error:
                    logger.warning(f"Retry failed for task {task_id}: {retry_error}")
            
            # Record performance for exceptions
            duration = time.time() - start_time
            self.performance_monitor.record(f"{service_type_str}_error", duration)
            
            # Record metrics
            self.metrics_collector.increment("tasks.error", tags={"service_type": service_type_str})
            
            # Publish event
            await self.event_bus.publish(Event(
                event_type=EventType.ERROR,
                data={"task_id": task_id, "error": str(e), "service_type": service_type_str},
                source="enhancer_agent"
            ))
            
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            await self.task_manager.fail_task(task_id, str(e))
            raise
    
    # Public API methods
    async def enhance_image(
        self,
        file_path: str,
        enhancement_type: str = "general",
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit image enhancement task."""
        return await TaskCreator.create_enhance_image_task(
            self.task_manager, file_path, enhancement_type, options, priority
        )
    
    async def enhance_video(
        self,
        file_path: str,
        enhancement_type: str = "general",
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit video enhancement task."""
        return await TaskCreator.create_enhance_video_task(
            self.task_manager, file_path, enhancement_type, options, priority
        )
    
    async def upscale(
        self,
        file_path: str,
        scale_factor: int = 2,
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit upscale task."""
        return await TaskCreator.create_upscale_task(
            self.task_manager, file_path, scale_factor, options, priority
        )
    
    async def denoise(
        self,
        file_path: str,
        noise_level: str = "medium",
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit denoise task."""
        return await TaskCreator.create_denoise_task(
            self.task_manager, file_path, noise_level, options, priority
        )
    
    async def restore(
        self,
        file_path: str,
        damage_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit restoration task."""
        return await TaskCreator.create_restore_task(
            self.task_manager, file_path, damage_type, options, priority
        )
    
    async def color_correction(
        self,
        file_path: str,
        correction_type: str = "auto",
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit color correction task."""
        return await TaskCreator.create_color_correction_task(
            self.task_manager, file_path, correction_type, options, priority
        )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        return await self.task_manager.get_task_status(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed task."""
        return await self.task_manager.get_task_result(task_id)
    
    async def process_batch(
        self,
        items: List[BatchItem],
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process multiple files in batch.
        
        Args:
            items: List of BatchItem to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with aggregated results
        """
        async def process_item(file_path: str, service_type: str = "enhance_image", **kwargs):
            # Check cache first
            cached = await self.cache_manager.get(
                file_path=file_path,
                service_type=service_type,
                **kwargs
            )
            
            if cached:
                return cached.get("task_id") if isinstance(cached, dict) else cached
            
            # Process based on service type
            if service_type == "enhance_image":
                task_id = await self.enhance_image(
                    file_path=file_path,
                    enhancement_type=kwargs.get("enhancement_type", "general"),
                    options=kwargs.get("options"),
                    priority=kwargs.get("priority", 0)
                )
            elif service_type == "enhance_video":
                task_id = await self.enhance_video(
                    file_path=file_path,
                    enhancement_type=kwargs.get("enhancement_type", "general"),
                    options=kwargs.get("options"),
                    priority=kwargs.get("priority", 0)
                )
            elif service_type == "upscale":
                task_id = await self.upscale(
                    file_path=file_path,
                    scale_factor=kwargs.get("scale_factor", 2),
                    options=kwargs.get("options"),
                    priority=kwargs.get("priority", 0)
                )
            elif service_type == "denoise":
                task_id = await self.denoise(
                    file_path=file_path,
                    noise_level=kwargs.get("noise_level", "medium"),
                    options=kwargs.get("options"),
                    priority=kwargs.get("priority", 0)
                )
            else:
                task_id = await self.enhance_image(
                    file_path=file_path,
                    enhancement_type=kwargs.get("enhancement_type", "general"),
                    options=kwargs.get("options"),
                    priority=kwargs.get("priority", 0)
                )
            
            return task_id
        
        # Process batch
        async def process_wrapper(item: BatchItem):
            return await process_item(
                file_path=item.file_path,
                service_type=item.service_type,
                enhancement_type=item.enhancement_type,
                options=item.options,
                priority=item.priority
            )
        
        result = await self.batch_processor.process_batch(
            items=items,
            process_func=process_wrapper,
            progress_callback=progress_callback
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "executor_stats": self.parallel_executor.get_stats(),
            "cache_stats": self.cache_manager.get_stats(),
            "webhook_stats": self.webhook_manager.get_stats(),
            "retry_stats": self.retry_manager.get_stats(),
            "performance_stats": self.performance_monitor.get_all_stats(),
            "metrics": {
                "counters": self.metrics_collector.get_counters(),
                "gauges": self.metrics_collector.get_gauges(),
                "available_metrics": self.metrics_collector.get_all_metrics()
            },
            "events": {
                "subscriber_count": self.event_bus.get_subscriber_count(),
                "history_size": len(self.event_bus.get_history())
            },
            "running": self.running,
            "max_parallel_tasks": self.parallel_executor.max_workers,
        }
    
    async def export_results(
        self,
        task_ids: Optional[List[str]] = None,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export task results to file.
        
        Args:
            task_ids: List of task IDs (None for all completed tasks)
            format: Export format (json, markdown, csv, html)
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        from ..utils.export import ResultExporter
        
        # Collect results
        results = []
        
        if task_ids:
            for task_id in task_ids:
                result = await self.get_task_result(task_id)
                if result:
                    status = await self.get_task_status(task_id)
                    results.append({
                        "task_id": task_id,
                        **status,
                        **result
                    })
        else:
            # Get all completed tasks
            all_tasks = await self.task_manager.repository.get_all()
            for task in all_tasks:
                if task.status.value == "completed" and task.result:
                    results.append({
                        "task_id": task.id,
                        "service_type": task.service_type,
                        "file_path": task.parameters.get("file_path", ""),
                        "status": task.status.value,
                        **task.result
                    })
        
        # Determine output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dirs["results"] / f"export_{timestamp}.{format}")
        
        # Export based on format
        if format == "json":
            return ResultExporter.export_json(results, output_path)
        elif format == "markdown":
            return ResultExporter.export_markdown(results, output_path)
        elif format == "csv":
            return ResultExporter.export_csv(results, output_path)
        elif format == "html":
            return ResultExporter.export_html(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

