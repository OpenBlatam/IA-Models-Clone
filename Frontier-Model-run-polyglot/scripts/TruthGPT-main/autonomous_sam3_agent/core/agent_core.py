"""
Autonomous SAM3 Agent Core
==========================

Core agent implementation based on SAM3 architecture with OpenRouter integration
for continuous 24/7 parallel execution.

Refactored with:
- SAM3ServiceHandler for service logic encapsulation
- TaskExecutor integration
- Improved separation of concerns
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path

from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.sam3_client import SAM3Client
from .task_manager import TaskManager
from .parallel_executor import ParallelExecutor
from .sam3_service_handler import SAM3ServiceHandler
from .helpers import (
    create_message,
    create_text_content,
    create_image_content,
    create_user_message_with_image,
    create_tool_message,
    load_json_file,
    save_json_file,
    create_output_structure,
    filter_outputs_by_indices,
    extract_tool_call_from_text,
)
from .validators import (
    TaskValidator,
    ValidationError,
)
from .metrics import MetricsCollector
from .rate_limiter import RateLimiter, RateLimitConfig
from .cache import ResultCache
from .notifications import (
    NotificationManager,
    NotificationType,
    LoggingNotificationHandler,
)

logger = logging.getLogger(__name__)


class AutonomousSAM3Agent:
    """
    Autonomous agent based on SAM3 architecture with OpenRouter integration.
    
    Features:
    - Continuous 24/7 operation
    - Parallel task execution
    - OpenRouter LLM integration
    - SAM3 segmentation capabilities
    - Automatic task management
    """
    
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        sam3_model_path: Optional[str] = None,
        max_parallel_tasks: int = 10,
        output_dir: str = "autonomous_agent_output",
        model: str = "anthropic/claude-3.5-sonnet",
        debug: bool = False,
    ):
        """
        Initialize autonomous SAM3 agent.
        """
        self.openrouter_client = OpenRouterClient(api_key=openrouter_api_key)
        self.sam3_client = SAM3Client(model_path=sam3_model_path)
        self.task_manager = TaskManager()
        self.parallel_executor = ParallelExecutor(max_workers=max_parallel_tasks)
        self.output_dir = Path(output_dir)
        self.model = model
        self.debug = debug
        self.running = False
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sam_out").mkdir(exist_ok=True)
        (self.output_dir / "agent_debug_out").mkdir(exist_ok=True)
        (self.output_dir / "none_out").mkdir(exist_ok=True)
        
        # Initialize Service Handler
        self.service_handler = SAM3ServiceHandler(
            openrouter_client=self.openrouter_client,
            sam3_client=self.sam3_client,
            model=self.model,
            output_dir=self.output_dir,
        )
        
        # Advanced metrics
        self.metrics = MetricsCollector(max_history=1000)
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            RateLimitConfig(
                max_requests=100,
                window_seconds=60,
                max_concurrent=max_parallel_tasks,
            )
        )
        
        # Result cache
        self.cache = ResultCache(
            cache_dir=str(self.output_dir / "cache"),
            max_size=1000,
            ttl_hours=24,
        )
        
        # Notification system
        self.notifications = NotificationManager()
        self.notifications.add_handler(LoggingNotificationHandler())
        
        # Legacy statistics
        self.stats = {
            "tasks_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_generations": 0,
            "errors": [],
        }
        
        logger.info(f"Initialized AutonomousSAM3Agent with {max_parallel_tasks} max parallel tasks")
    
    async def start(self):
        """Start the autonomous agent in continuous operation mode."""
        if self.running:
            logger.warning("Agent is already running")
            return
        
        self.running = True
        logger.info("Starting Autonomous SAM3 Agent (24/7 mode)")
        
        await self.notifications.notify(
            NotificationType.AGENT_STARTED,
            "Autonomous SAM3 Agent started",
            {"max_parallel_tasks": self.parallel_executor.max_workers},
        )
        
        try:
            await self.parallel_executor.start()
            
            while self.running:
                try:
                    tasks = await self.task_manager.get_pending_tasks(limit=10)
                    
                    if tasks:
                        for task in tasks:
                            await self.parallel_executor.submit_task(
                                self._process_task,
                                task
                            )
                    
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
        logger.info("Stopping Autonomous SAM3 Agent")
        self.running = False
        
        await self.notifications.notify(
            NotificationType.AGENT_STOPPED,
            "Autonomous SAM3 Agent stopping",
            {"stats": self.get_stats()},
        )
        
        await self.parallel_executor.stop()
        await self.openrouter_client.close()
        await self.sam3_client.close()
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task using SAM3 agent inference."""
        task_id = task.get("id", "unknown")
        image_path = task.get("image_path")
        text_prompt = task.get("text_prompt")
        
        logger.info(f"Processing task {task_id}: {text_prompt}")
        
        self.stats["tasks_processed"] += 1
        self.metrics.start_task(task_id)
        
        try:
            # Check cache first
            cached_result = self.cache.get(image_path, text_prompt)
            if cached_result:
                logger.info(f"Task {task_id} served from cache")
                await self.task_manager.complete_task(task_id, cached_result)
                self.stats["tasks_completed"] += 1
                self.metrics.complete_task(
                    task_id,
                    generations=0,
                    masks_found=len(cached_result.get("pred_masks", []))
                )
                return cached_result
            
            # Update task status
            await self.task_manager.update_task_status(task_id, "processing")
            
            await self.notifications.notify(
                NotificationType.TASK_STARTED,
                f"Task {task_id} started processing",
                {"task_id": task_id, "text_prompt": text_prompt},
            )
            
            # Run agent inference with rate limiting
            try:
                await self.rate_limiter.acquire()
                try:
                    # Delegate to Service Handler
                    result = await self.service_handler.run_inference(
                        image_path=image_path,
                        initial_text_prompt=text_prompt,
                        task_id=task_id,
                    )
                finally:
                    await self.rate_limiter.release()
            except Exception as e:
                await self.rate_limiter.release()
                raise
            
            # Cache result
            self.cache.set(image_path, text_prompt, result)
            
            # Save result
            await self.task_manager.complete_task(task_id, result)
            
            # Update metrics
            num_masks = len(result.get("pred_masks", []))
            generations = result.get("generations", 0)
            self.metrics.complete_task(task_id, generations, num_masks)
            
            self.stats["tasks_completed"] += 1
            self.stats["total_generations"] += generations
            logger.info(f"Task {task_id} completed successfully")
            
            await self.notifications.notify(
                NotificationType.TASK_COMPLETED,
                f"Task {task_id} completed successfully",
                {
                    "task_id": task_id,
                    "masks_found": num_masks,
                    "generations": generations,
                },
            )
            
            return result
            
        except ValidationError as e:
            self._handle_error(task_id, e, "ValidationError")
            raise
        except Exception as e:
            self._handle_error(task_id, e, type(e).__name__)
            raise
    
    async def _handle_error(self, task_id: str, error: Exception, error_type: str):
        """Handle task error."""
        self.stats["tasks_failed"] += 1
        self.metrics.fail_task(task_id, str(error))
        self.stats["errors"].append({
            "task_id": task_id,
            "error_type": error_type,
            "message": str(error),
        })
        logger.error(f"Error processing task {task_id}: {error}", exc_info=True)
        await self.task_manager.fail_task(task_id, str(error))
        
        await self.notifications.notify(
            NotificationType.TASK_FAILED,
            f"Task {task_id} failed: {str(error)}",
            {"task_id": task_id, "error": str(error), "error_type": error_type},
            severity="error",
        )
    
    async def submit_task(
        self,
        image_path: str,
        text_prompt: str,
        priority: int = 0,
    ) -> str:
        """Submit a new task to the agent."""
        try:
            TaskValidator.validate_task_inputs_raise(image_path, text_prompt, priority)
        except ValidationError as e:
            logger.error(f"Task validation failed: {e}")
            raise
        
        task_id = await self.task_manager.create_task(
            image_path=image_path,
            text_prompt=text_prompt,
            priority=priority,
        )
        
        logger.info(f"Submitted task {task_id}: {text_prompt}")
        
        await self.notifications.notify(
            NotificationType.TASK_CREATED,
            f"Task {task_id} created",
            {"task_id": task_id, "text_prompt": text_prompt, "priority": priority},
        )
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        return await self.task_manager.get_task_status(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed task."""
        return await self.task_manager.get_task_result(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        metrics_summary = self.metrics.get_summary()
        performance_stats = self.metrics.get_performance_stats()
        
        return {
            **self.stats,
            **metrics_summary,
            "performance": performance_stats,
            "cache": self.cache.get_stats(),
            "rate_limiter": self.rate_limiter.get_stats(),
            "success_rate": (
                self.stats["tasks_completed"] / self.stats["tasks_processed"]
                if self.stats["tasks_processed"] > 0
                else 0.0
            ),
            "avg_generations_per_task": (
                self.stats["total_generations"] / self.stats["tasks_completed"]
                if self.stats["tasks_completed"] > 0
                else 0.0
            ),
            "running": self.running,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on agent components."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Check components
        components = {
            "openrouter": self.openrouter_client,
            "sam3": self.sam3_client,
            "task_manager": self.task_manager,
            "parallel_executor": self.parallel_executor,
        }
        
        for name, comp in components.items():
            try:
                health["components"][name] = {
                    "status": "healthy" if comp else "unhealthy",
                    "message": "Initialized" if comp else "Not initialized",
                }
            except Exception as e:
                health["components"][name] = {
                    "status": "unhealthy",
                    "message": str(e),
                }
        
        # Check output directory
        try:
            output_dir_exists = self.output_dir.exists() and self.output_dir.is_dir()
            health["components"]["output_dir"] = {
                "status": "healthy" if output_dir_exists else "unhealthy",
                "message": f"Directory exists: {self.output_dir}" if output_dir_exists else "Directory not found",
            }
        except Exception as e:
            health["components"]["output_dir"] = {
                "status": "unhealthy",
                "message": str(e),
            }
        
        # Overall status
        unhealthy_components = [
            name for name, comp in health["components"].items()
            if comp["status"] == "unhealthy"
        ]
        if unhealthy_components:
            health["status"] = "degraded"
            health["unhealthy_components"] = unhealthy_components
        
        return health
    
    def clear_cache(self) -> None:
        """Clear result cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        return self.cache.cleanup_expired()
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self.stats["errors"][-limit:]
