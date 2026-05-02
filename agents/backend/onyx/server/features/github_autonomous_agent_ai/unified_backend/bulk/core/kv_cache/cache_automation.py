"""
Cache automation utilities.

Provides automation for cache management tasks.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from threading import Thread
import signal

logger = logging.getLogger(__name__)


class CacheAutomation:
    """
    Cache automation manager.
    
    Provides automated tasks for cache management.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache automation.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.automated_tasks: List[Dict[str, Any]] = []
        self.running = False
        self.thread: Optional[Thread] = None
    
    def schedule_task(
        self,
        task_name: str,
        task_fn: Callable,
        interval: float,
        **kwargs
    ) -> None:
        """
        Schedule automated task.
        
        Args:
            task_name: Name of task
            task_fn: Task function
            interval: Execution interval (seconds)
            **kwargs: Arguments for task function
        """
        self.automated_tasks.append({
            "name": task_name,
            "function": task_fn,
            "interval": interval,
            "kwargs": kwargs,
            "last_run": 0.0,
            "enabled": True
        })
    
    def start_automation(self) -> None:
        """Start automation thread."""
        if self.running:
            logger.warning("Automation already running")
            return
        
        self.running = True
        self.thread = Thread(target=self._automation_loop, daemon=True)
        self.thread.start()
        logger.info("Cache automation started")
    
    def stop_automation(self) -> None:
        """Stop automation thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Cache automation stopped")
    
    def _automation_loop(self) -> None:
        """Main automation loop."""
        while self.running:
            current_time = time.time()
            
            for task in self.automated_tasks:
                if not task["enabled"]:
                    continue
                
                if current_time - task["last_run"] >= task["interval"]:
                    try:
                        task["function"](**task["kwargs"])
                        task["last_run"] = current_time
                    except Exception as e:
                        logger.error(f"Automated task {task['name']} failed: {e}")
            
            time.sleep(1.0)  # Check every second
    
    def enable_task(self, task_name: str) -> bool:
        """Enable a task."""
        for task in self.automated_tasks:
            if task["name"] == task_name:
                task["enabled"] = True
                return True
        return False
    
    def disable_task(self, task_name: str) -> bool:
        """Disable a task."""
        for task in self.automated_tasks:
            if task["name"] == task_name:
                task["enabled"] = False
                return True
        return False


class CacheAutoBackup:
    """
    Automated backup manager.
    
    Provides automated backup scheduling.
    """
    
    def __init__(
        self,
        cache: Any,
        backup_interval: float = 3600.0,  # 1 hour
        backup_dir: str = "auto_backups"
    ):
        """
        Initialize auto backup.
        
        Args:
            cache: Cache instance
            backup_interval: Backup interval (seconds)
            backup_dir: Backup directory
        """
        self.cache = cache
        self.backup_interval = backup_interval
        self.backup_dir = backup_dir
        
        from kv_cache import CacheBackupManager
        self.backup_manager = CacheBackupManager(cache, backup_dir)
    
    def create_backup(self) -> str:
        """
        Create automated backup.
        
        Returns:
            Path to backup file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"auto_backup_{timestamp}"
        return self.backup_manager.create_backup(name)
    
    def setup_auto_backup(self, automation: CacheAutomation) -> None:
        """
        Setup automated backup.
        
        Args:
            automation: Automation manager
        """
        automation.schedule_task(
            "auto_backup",
            self.create_backup,
            self.backup_interval
        )


class CacheAutoOptimization:
    """
    Automated optimization manager.
    
    Provides automated optimization scheduling.
    """
    
    def __init__(self, cache: Any, optimization_interval: float = 1800.0):
        """
        Initialize auto optimization.
        
        Args:
            cache: Cache instance
            optimization_interval: Optimization interval (seconds)
        """
        self.cache = cache
        self.optimization_interval = optimization_interval
        
        from kv_cache import CacheOptimizer
        self.optimizer = CacheOptimizer(cache)
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run automated optimization.
        
        Returns:
            Optimization results
        """
        return self.optimizer.optimize()
    
    def setup_auto_optimization(self, automation: CacheAutomation) -> None:
        """
        Setup automated optimization.
        
        Args:
            automation: Automation manager
        """
        automation.schedule_task(
            "auto_optimization",
            self.run_optimization,
            self.optimization_interval
        )

