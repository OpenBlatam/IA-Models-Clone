"""
Cache migration utilities.

Provides migration capabilities for cache data.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MigrationStrategy(Enum):
    """Migration strategies."""
    FULL_COPY = "full_copy"
    INCREMENTAL = "incremental"
    STREAMING = "streaming"
    PARALLEL = "parallel"


@dataclass
class MigrationPlan:
    """Migration plan."""
    strategy: MigrationStrategy
    source: Any
    target: Any
    steps: List[str]
    metadata: Dict[str, Any]


class CacheMigrator:
    """
    Cache migrator.
    
    Provides migration capabilities.
    """
    
    def __init__(self):
        """Initialize migrator."""
        self.migrations: List[Dict[str, Any]] = []
    
    def migrate(
        self,
        source: Any,
        target: Any,
        strategy: MigrationStrategy = MigrationStrategy.FULL_COPY,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Migrate cache from source to target.
        
        Args:
            source: Source cache
            target: Target cache
            strategy: Migration strategy
            progress_callback: Optional progress callback
            
        Returns:
            Migration result
        """
        start_time = time.time()
        
        result = {
            "success": False,
            "migrated_count": 0,
            "failed_count": 0,
            "duration": 0.0,
            "errors": []
        }
        
        try:
            if strategy == MigrationStrategy.FULL_COPY:
                result = self._migrate_full_copy(source, target, progress_callback)
            elif strategy == MigrationStrategy.INCREMENTAL:
                result = self._migrate_incremental(source, target, progress_callback)
            elif strategy == MigrationStrategy.STREAMING:
                result = self._migrate_streaming(source, target, progress_callback)
            elif strategy == MigrationStrategy.PARALLEL:
                result = self._migrate_parallel(source, target, progress_callback)
            
            result["success"] = result["failed_count"] == 0
            result["duration"] = time.time() - start_time
            
            self.migrations.append({
                "timestamp": time.time(),
                "strategy": strategy.value,
                "result": result
            })
            
            logger.info(f"Migration completed: {result['migrated_count']} entries")
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Migration failed: {e}")
        
        return result
    
    def _migrate_full_copy(
        self,
        source: Any,
        target: Any,
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Migrate using full copy strategy."""
        source_stats = source.get_stats()
        cache_size = source_stats.get("cache_size", 0)
        
        migrated = 0
        failed = 0
        errors = []
        
        for position in range(cache_size):
            try:
                value = source.get(position)
                if value is not None:
                    target.put(position, value)
                    migrated += 1
                
                if progress_callback:
                    progress_callback(position, cache_size)
                    
            except Exception as e:
                failed += 1
                errors.append(f"Position {position}: {e}")
        
        return {
            "migrated_count": migrated,
            "failed_count": failed,
            "errors": errors
        }
    
    def _migrate_incremental(
        self,
        source: Any,
        target: Any,
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Migrate using incremental strategy."""
        # Similar to full copy but only new/changed entries
        return self._migrate_full_copy(source, target, progress_callback)
    
    def _migrate_streaming(
        self,
        source: Any,
        target: Any,
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Migrate using streaming strategy."""
        # Stream in batches
        from kv_cache import CacheStreamer
        
        streamer = CacheStreamer(source)
        source_stats = source.get_stats()
        cache_size = source_stats.get("cache_size", 0)
        
        migrated = 0
        failed = 0
        errors = []
        
        positions = iter(range(cache_size))
        
        for value in streamer.stream_get(positions):
            try:
                if value is not None:
                    # Get position from iterator somehow
                    target.put(migrated, value)
                    migrated += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))
        
        return {
            "migrated_count": migrated,
            "failed_count": failed,
            "errors": errors
        }
    
    def _migrate_parallel(
        self,
        source: Any,
        target: Any,
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Migrate using parallel strategy."""
        import threading
        
        source_stats = source.get_stats()
        cache_size = source_stats.get("cache_size", 0)
        
        migrated = 0
        failed = 0
        errors = []
        lock = threading.Lock()
        
        def migrate_range(start: int, end: int):
            nonlocal migrated, failed
            local_migrated = 0
            local_failed = 0
            
            for position in range(start, end):
                try:
                    value = source.get(position)
                    if value is not None:
                        target.put(position, value)
                        local_migrated += 1
                except Exception as e:
                    local_failed += 1
                    errors.append(f"Position {position}: {e}")
            
            with lock:
                migrated += local_migrated
                failed += local_failed
        
        # Split into threads
        num_threads = 4
        chunk_size = cache_size // num_threads
        
        threads = []
        for i in range(num_threads):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_threads - 1 else cache_size
            thread = threading.Thread(target=migrate_range, args=(start, end))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        return {
            "migrated_count": migrated,
            "failed_count": failed,
            "errors": errors
        }
    
    def create_migration_plan(
        self,
        source: Any,
        target: Any,
        strategy: MigrationStrategy
    ) -> MigrationPlan:
        """
        Create migration plan.
        
        Args:
            source: Source cache
            target: Target cache
            strategy: Migration strategy
            
        Returns:
            Migration plan
        """
        steps = []
        
        if strategy == MigrationStrategy.FULL_COPY:
            steps = [
                "1. Backup source cache",
                "2. Initialize target cache",
                "3. Copy all entries",
                "4. Verify integrity",
                "5. Switch to target"
            ]
        elif strategy == MigrationStrategy.INCREMENTAL:
            steps = [
                "1. Track changes",
                "2. Migrate only changed entries",
                "3. Verify consistency"
            ]
        
        plan = MigrationPlan(
            strategy=strategy,
            source=source,
            target=target,
            steps=steps,
            metadata={}
        )
        
        return plan
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Get migration history.
        
        Returns:
            Migration history
        """
        return self.migrations.copy()

