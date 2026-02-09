"""
Service Factory for Piel Mejorador AI SAM3
==========================================

Factory for creating and configuring services.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..config.piel_mejorador_config import PielMejoradorConfig
from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient
from .task_manager import TaskManager, FileTaskRepository
from .cache_manager import CacheManager
from .parallel_executor import ParallelExecutor
from .batch_processor import BatchProcessor
from .webhook_manager import WebhookManager
from .memory_optimizer import MemoryOptimizer
from .alert_manager import AlertManager
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .performance_optimizer import PerformanceOptimizer
from .backup_manager import BackupManager
from .helpers import create_output_directories

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating services.
    
    Centralizes service creation and configuration.
    """
    
    @staticmethod
    def create_openrouter_client(config: PielMejoradorConfig) -> OpenRouterClient:
        """Create OpenRouter client."""
        return OpenRouterClient(api_key=config.openrouter.api_key)
    
    @staticmethod
    def create_truthgpt_client(config: PielMejoradorConfig) -> TruthGPTClient:
        """Create TruthGPT client."""
        truthgpt_config = config.truthgpt.to_dict() if config.truthgpt else {}
        return TruthGPTClient(config=truthgpt_config)
    
    @staticmethod
    def create_task_manager(output_dirs: Dict[str, Path]) -> TaskManager:
        """Create task manager."""
        return TaskManager(
            repository=FileTaskRepository(str(output_dirs["storage"]))
        )
    
    @staticmethod
    def create_cache_manager(output_dirs: Dict[str, Path]) -> CacheManager:
        """Create cache manager."""
        return CacheManager(
            cache_dir=output_dirs.get("cache", output_dirs["results"] / "cache")
        )
    
    @staticmethod
    def create_parallel_executor(max_workers: int) -> ParallelExecutor:
        """Create parallel executor."""
        return ParallelExecutor(max_workers=max_workers)
    
    @staticmethod
    def create_batch_processor(max_concurrent: int) -> BatchProcessor:
        """Create batch processor."""
        return BatchProcessor(max_concurrent=max_concurrent)
    
    @staticmethod
    def create_webhook_manager() -> WebhookManager:
        """Create webhook manager."""
        return WebhookManager()
    
    @staticmethod
    def create_memory_optimizer() -> MemoryOptimizer:
        """Create memory optimizer."""
        return MemoryOptimizer()
    
    @staticmethod
    def create_alert_manager() -> AlertManager:
        """Create alert manager."""
        return AlertManager()
    
    @staticmethod
    def create_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create circuit breaker."""
        return CircuitBreaker(
            name=name,
            config=config or CircuitBreakerConfig()
        )
    
    @staticmethod
    def create_performance_optimizer(initial_concurrency: int) -> PerformanceOptimizer:
        """Create performance optimizer."""
        return PerformanceOptimizer(initial_concurrency=initial_concurrency)
    
    @staticmethod
    def create_backup_manager(output_dirs: Dict[str, Path], retention_days: int = 7) -> BackupManager:
        """Create backup manager."""
        backup_dir = output_dirs.get("backups", output_dirs["results"] / "backups")
        return BackupManager(backup_dir=backup_dir, retention_days=retention_days)
    
    @staticmethod
    def create_output_directories(output_dir: Path) -> Dict[str, Path]:
        """Create output directories."""
        return create_output_directories(
            output_dir,
            ["results", "tasks", "storage", "uploads", "cache", "temp", "backups"]
        )




