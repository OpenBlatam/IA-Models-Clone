"""
Piel Mejorador Agent - Main class for skin enhancement with SAM3 architecture
============================================================================
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .alert_manager import AlertLevel
from pathlib import Path

from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient
from ..config.piel_mejorador_config import PielMejoradorConfig
from .system_prompts_builder import SystemPromptsBuilder
from .task_manager import TaskManager, FileTaskRepository, ServiceType
from .service_handler import ServiceHandler
from .parallel_executor import ParallelExecutor
from .validators import ParameterValidator, ValidationError
from .cache_manager import CacheManager
from .batch_processor import BatchProcessor, BatchItem, BatchResult
from .webhook_manager import WebhookManager, WebhookEvent
from .memory_optimizer import MemoryOptimizer
from .alert_manager import AlertManager, AlertType, AlertLevel
from .config_validator import ConfigValidator
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .performance_optimizer import PerformanceOptimizer
from .backup_manager import BackupManager
from .helpers import create_output_directories
from .service_factory import ServiceFactory

logger = logging.getLogger(__name__)


class PielMejoradorAgent:
    """
    Autonomous agent for skin enhancement based on SAM3 architecture.
    
    Features:
    - Continuous 24/7 operation
    - Parallel task execution
    - OpenRouter LLM integration
    - TruthGPT optimization
    - Automatic task management
    - Image and video skin enhancement
    """
    
    def __init__(
        self,
        config: Optional[PielMejoradorConfig] = None,
        max_parallel_tasks: int = 5,
        output_dir: str = "piel_mejorador_output",
        debug: bool = False,
        factory: Optional[ServiceFactory] = None,
    ):
        """
        Initialize agent.
        
        Args:
            config: Configuration
            max_parallel_tasks: Maximum parallel tasks
            output_dir: Output directory
            debug: Debug mode
            factory: Optional service factory (uses default if None)
        """
        self.config = config or PielMejoradorConfig()
        self.config.validate()
        
        self.factory = factory or ServiceFactory()
        self.max_parallel_tasks = max_parallel_tasks
        self.debug = debug
        self.running = False
        
        # Create output directories
        self.output_dir = Path(output_dir)
        self.output_dirs = self.factory.create_output_directories(self.output_dir)
        
        # Initialize clients
        self.openrouter_client = self.factory.create_openrouter_client(self.config)
        self.truthgpt_client = self.factory.create_truthgpt_client(self.config)
        
        # Initialize managers
        self.task_manager = self.factory.create_task_manager(self.output_dirs)
        self.cache_manager = self.factory.create_cache_manager(self.output_dirs)
        self.webhook_manager = self.factory.create_webhook_manager()
        self.memory_optimizer = self.factory.create_memory_optimizer()
        self.alert_manager = self.factory.create_alert_manager()
        
        # Initialize processors
        self.parallel_executor = self.factory.create_parallel_executor(max_parallel_tasks)
        self.batch_processor = self.factory.create_batch_processor(max_parallel_tasks)
        
        # Initialize optimizers
        self.performance_optimizer = self.factory.create_performance_optimizer(max_parallel_tasks)
        self.openrouter_circuit = self.factory.create_circuit_breaker(
            "openrouter",
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60.0)
        )
        
        # Initialize backup manager
        self.backup_manager = self.factory.create_backup_manager(self.output_dirs)
        
        # Setup alert handlers
        self._setup_alert_handlers()
        
        # Initialize system prompts and service handler
        self._initialize_service_handler()
        
        logger.info(f"Initialized PielMejoradorAgent with {max_parallel_tasks} max parallel tasks")
    
    def _setup_alert_handlers(self):
        """Setup alert handlers."""
        def memory_alert_handler(alert):
            logger.critical(f"MEMORY ALERT: {alert.message}")
            # Could trigger webhook, email, etc.
        
        def failure_rate_handler(alert):
            logger.error(f"FAILURE RATE ALERT: {alert.message}")
        
        self.alert_manager.register_handler(AlertType.MEMORY_HIGH, memory_alert_handler)
        self.alert_manager.register_handler(AlertType.TASK_FAILURE_RATE, failure_rate_handler)
    
    def _initialize_service_handler(self):
        """Initialize system prompts and service handler."""
        self.system_prompts = SystemPromptsBuilder.build_all_prompts()
        self.service_handler = ServiceHandler(
            openrouter_client=self.openrouter_client,
            truthgpt_client=self.truthgpt_client,
            config=self.config,
            system_prompts=self.system_prompts
        )
    
    async def start(self):
        """Start the autonomous agent in continuous operation mode."""
        if self.running:
            logger.warning("Agent is already running")
            return
        
        self.running = True
        logger.info("Starting Piel Mejorador Agent (24/7 mode)")
        
        try:
            # Initialize task manager (load existing tasks)
            await self.task_manager.initialize()
            
            # Start parallel executor
            await self.parallel_executor.start()
            
            # Start performance optimizer
            asyncio.create_task(self.performance_optimizer.optimize_periodically())
            
            # Create initial backup
            await self.backup_manager.create_backup(
                source_dir=self.output_dirs["storage"],
                backup_name="initial_backup"
            )
            
            # Main event loop for continuous operation
            while self.running:
                try:
                    # Get pending tasks from task manager
                    tasks = await self.task_manager.get_pending_tasks(limit=self.max_parallel_tasks)
                    
                    if tasks:
                        # Submit tasks to parallel executor
                        for task in tasks:
                            await self.parallel_executor.submit_task(
                                self._process_task,
                                task.to_dict()
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
        logger.info("Stopping Piel Mejorador Agent")
        self.running = False
        await self.parallel_executor.stop()
        await self.webhook_manager.close()
        await self.openrouter_client.close()
        await self.truthgpt_client.close()
    
    async def close(self):
        """Close agent and cleanup resources."""
        await self.stop()
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task."""
        task_id = task.get("id", "unknown")
        service_type_str = task.get("service_type")
        parameters = task.get("parameters", {})
        
        logger.info(f"Processing task {task_id}: {service_type_str}")
        
        try:
            # Update task status
            await self.task_manager.update_task_status(task_id, "processing")
            
            # Send webhook notification
            await self.webhook_manager.send(
                WebhookEvent.TASK_STARTED,
                {"service_type": service_type_str},
                task_id=task_id
            )
            
            # Convert string service type to Enum
            try:
                service_type = ServiceType(service_type_str)
            except ValueError:
                raise ValueError(f"Unknown service type: {service_type_str}")
            
            # Execute service using handler
            result = await self.service_handler.handle(service_type, parameters)
            
            if result.success:
                # Save result
                await self.task_manager.complete_task(task_id, result.to_dict())
                logger.info(f"Task {task_id} completed successfully")
                
                # Send webhook notification
                await self.webhook_manager.send(
                    WebhookEvent.TASK_COMPLETED,
                    {"result": result.to_dict()},
                    task_id=task_id
                )
                
                # Optimize memory periodically
                if self.memory_optimizer.is_memory_pressure():
                    await self.memory_optimizer.optimize()
                    # Check for memory alert
                    memory_usage = self.memory_optimizer.get_memory_usage()
                    self.alert_manager.check_memory_alert(
                        memory_usage["system_memory_percent"]
                    )
                
                return result.to_dict()
            else:
                # Handle service failure
                error_msg = result.error or "Unknown error"
                await self.task_manager.fail_task(task_id, error_msg)
                logger.error(f"Task {task_id} failed: {error_msg}")
                
                # Send webhook notification
                await self.webhook_manager.send(
                    WebhookEvent.TASK_FAILED,
                    {"error": error_msg},
                    task_id=task_id
                )
                
                raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            await self.task_manager.fail_task(task_id, str(e))
            raise
    
    # Public API methods
    async def mejorar_imagen(
        self,
        file_path: str,
        enhancement_level: str = "medium",
        realism_level: Optional[float] = None,
        custom_instructions: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """Submit image enhancement task."""
        # Validate parameters
        ParameterValidator.validate_enhancement_level(enhancement_level)
        ParameterValidator.validate_realism_level(realism_level)
        ParameterValidator.validate_file_path(file_path)
        ParameterValidator.validate_priority(priority)
        ParameterValidator.validate_custom_instructions(custom_instructions)
        
        return await self.task_manager.create_task(
            service_type=ServiceType.MEJORAR_IMAGEN.value,
            parameters={
                "file_path": file_path,
                "enhancement_level": enhancement_level,
                "realism_level": realism_level,
                "custom_instructions": custom_instructions,
            },
            priority=priority
        )
    
    async def mejorar_video(
        self,
        file_path: str,
        enhancement_level: str = "medium",
        realism_level: Optional[float] = None,
        custom_instructions: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """Submit video enhancement task."""
        # Validate parameters
        ParameterValidator.validate_enhancement_level(enhancement_level)
        ParameterValidator.validate_realism_level(realism_level)
        ParameterValidator.validate_file_path(file_path)
        ParameterValidator.validate_priority(priority)
        ParameterValidator.validate_custom_instructions(custom_instructions)
        
        return await self.task_manager.create_task(
            service_type=ServiceType.MEJORAR_VIDEO.value,
            parameters={
                "file_path": file_path,
                "enhancement_level": enhancement_level,
                "realism_level": realism_level,
                "custom_instructions": custom_instructions,
            },
            priority=priority
        )
    
    async def analizar_piel(
        self,
        file_path: str,
        file_type: str = "image",
        priority: int = 0,
    ) -> str:
        """Submit skin analysis task."""
        # Validate parameters
        ParameterValidator.validate_file_type(file_type)
        ParameterValidator.validate_file_path(file_path)
        ParameterValidator.validate_priority(priority)
        
        return await self.task_manager.create_task(
            service_type=ServiceType.ANALISIS_PIEL.value,
            parameters={
                "file_path": file_path,
                "file_type": file_type,
            },
            priority=priority
        )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        return await self.task_manager.get_task_status(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed task."""
        return await self.task_manager.get_task_result(task_id)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "executor_stats": self.parallel_executor.get_stats(),
            "cache_stats": self.cache_manager.get_stats(),
            "webhook_stats": self.webhook_manager.get_stats(),
            "memory_usage": self.memory_optimizer.get_memory_usage(),
            "alert_stats": self.alert_manager.get_stats(),
            "running": self.running,
            "max_parallel_tasks": self.max_parallel_tasks,
        }
        
        # Check for alerts based on stats
        executor_stats = stats["executor_stats"]
        if executor_stats.get("total_tasks", 0) > 0:
            failure_rate = executor_stats.get("failed_tasks", 0) / executor_stats["total_tasks"]
            self.alert_manager.check_task_failure_alert(failure_rate)
        
        cache_stats = stats["cache_stats"]
        if cache_stats.get("hit_rate", 0) < 0.5:
            miss_rate = 1 - cache_stats["hit_rate"]
            self.alert_manager.check_and_alert(
                AlertType.CACHE_MISS_RATE,
                miss_rate,
                f"Cache miss rate is high: {miss_rate:.1%}",
                cache_stats
            )
        
        return stats
    
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
        async def process_item(file_path: str, enhancement_level: str, **kwargs):
            # Check cache first
            cached = await self.cache_manager.get(
                file_path=file_path,
                enhancement_level=enhancement_level,
                realism_level=kwargs.get("realism_level"),
                custom_instructions=kwargs.get("custom_instructions")
            )
            
            if cached:
                return cached
            
            # Process and cache result
            task_id = await self.mejorar_imagen(
                file_path=file_path,
                enhancement_level=enhancement_level,
                **{k: v for k, v in kwargs.items() if k != "priority"}
            )
            
            # Wait for completion
            while True:
                status = await self.get_task_status(task_id)
                if status["status"] == "completed":
                    result = await self.get_task_result(task_id)
                    # Cache result
                    await self.cache_manager.set(
                        file_path=file_path,
                        enhancement_level=enhancement_level,
                        result=result,
                        realism_level=kwargs.get("realism_level"),
                        custom_instructions=kwargs.get("custom_instructions")
                    )
                    return result
                elif status["status"] == "failed":
                    raise Exception(status.get("error", "Task failed"))
                await asyncio.sleep(0.5)
        
        return await self.batch_processor.process_batch(
            items=items,
            process_func=process_item,
            progress_callback=progress_callback
        )
    
    async def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        return await self.cache_manager.cleanup_expired()
    
    def register_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None
    ):
        """Register a webhook for notifications."""
        from .webhook_manager import Webhook
        webhook = Webhook(url=url, events=events, secret=secret)
        self.webhook_manager.register(webhook)
    
    async def optimize_memory(self, force: bool = False):
        """Optimize memory usage."""
        await self.memory_optimizer.optimize(force=force)
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        return self.memory_optimizer.get_recommendations()
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List:
        """Get active alerts."""
        from .alert_manager import AlertLevel as AL
        return self.alert_manager.get_active_alerts(level=level)
    
    def get_alert_history(self, limit: int = 100) -> List:
        """Get alert history."""
        return self.alert_manager.get_alert_history(limit=limit)

