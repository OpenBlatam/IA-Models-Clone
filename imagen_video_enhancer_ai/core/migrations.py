"""
Migrations System
=================

System for database and data migrations.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Migration definition."""
    id: str
    name: str
    version: str
    up: Callable[[], Any]  # Migration function
    down: Optional[Callable[[], Any]] = None  # Rollback function
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "dependencies": self.dependencies
        }


class MigrationRunner:
    """Migration runner and manager."""
    
    def __init__(self, migrations_dir: Path):
        """
        Initialize migration runner.
        
        Args:
            migrations_dir: Directory containing migration files
        """
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.migrations: Dict[str, Migration] = {}
        self.history_file = self.migrations_dir / "migration_history.json"
        self._history: List[Dict[str, Any]] = []
        self._load_history()
    
    def _load_history(self):
        """Load migration history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading migration history: {e}")
                self._history = []
    
    def _save_history(self):
        """Save migration history."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving migration history: {e}")
    
    def register(self, migration: Migration):
        """
        Register a migration.
        
        Args:
            migration: Migration to register
        """
        self.migrations[migration.id] = migration
        logger.info(f"Registered migration: {migration.id} ({migration.name})")
    
    def get_applied_migrations(self) -> List[str]:
        """
        Get list of applied migration IDs.
        
        Returns:
            List of migration IDs
        """
        return [
            entry["migration_id"]
            for entry in self._history
            if entry.get("status") == MigrationStatus.COMPLETED.value
        ]
    
    def get_pending_migrations(self) -> List[Migration]:
        """
        Get list of pending migrations.
        
        Returns:
            List of pending migrations
        """
        applied = set(self.get_applied_migrations())
        pending = [
            migration
            for migration in self.migrations.values()
            if migration.id not in applied
        ]
        
        # Sort by dependencies
        return self._sort_by_dependencies(pending)
    
    def _sort_by_dependencies(self, migrations: List[Migration]) -> List[Migration]:
        """Sort migrations by dependencies."""
        sorted_migrations = []
        remaining = migrations.copy()
        applied = set(self.get_applied_migrations())
        
        while remaining:
            progress = False
            for migration in remaining[:]:
                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    dep in applied or any(m.id == dep for m in sorted_migrations)
                    for dep in migration.dependencies
                )
                
                if deps_satisfied:
                    sorted_migrations.append(migration)
                    remaining.remove(migration)
                    progress = True
            
            if not progress:
                # Circular dependency or missing dependency
                logger.warning("Could not resolve all dependencies")
                sorted_migrations.extend(remaining)
                break
        
        return sorted_migrations
    
    async def run_migration(self, migration_id: str) -> bool:
        """
        Run a specific migration.
        
        Args:
            migration_id: Migration ID
            
        Returns:
            True if successful
        """
        if migration_id not in self.migrations:
            logger.error(f"Migration not found: {migration_id}")
            return False
        
        migration = self.migrations[migration_id]
        
        # Check if already applied
        if migration_id in self.get_applied_migrations():
            logger.info(f"Migration already applied: {migration_id}")
            return True
        
        # Record start
        history_entry = {
            "migration_id": migration_id,
            "status": MigrationStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None
        }
        self._history.append(history_entry)
        self._save_history()
        
        try:
            # Run migration
            if asyncio.iscoroutinefunction(migration.up):
                await migration.up()
            else:
                migration.up()
            
            # Record success
            history_entry["status"] = MigrationStatus.COMPLETED.value
            history_entry["completed_at"] = datetime.now().isoformat()
            self._save_history()
            
            logger.info(f"Migration completed: {migration_id}")
            return True
            
        except Exception as e:
            # Record failure
            history_entry["status"] = MigrationStatus.FAILED.value
            history_entry["error"] = str(e)
            history_entry["completed_at"] = datetime.now().isoformat()
            self._save_history()
            
            logger.error(f"Migration failed: {migration_id} - {e}")
            return False
    
    async def run_all_pending(self) -> Dict[str, bool]:
        """
        Run all pending migrations.
        
        Returns:
            Dictionary of migration_id -> success
        """
        pending = self.get_pending_migrations()
        results = {}
        
        for migration in pending:
            success = await self.run_migration(migration.id)
            results[migration.id] = success
            
            if not success:
                # Stop on first failure
                logger.error(f"Stopping migrations due to failure: {migration.id}")
                break
        
        return results
    
    async def rollback(self, migration_id: str) -> bool:
        """
        Rollback a migration.
        
        Args:
            migration_id: Migration ID
            
        Returns:
            True if successful
        """
        if migration_id not in self.migrations:
            logger.error(f"Migration not found: {migration_id}")
            return False
        
        migration = self.migrations[migration_id]
        
        # Check if applied
        if migration_id not in self.get_applied_migrations():
            logger.warning(f"Migration not applied: {migration_id}")
            return False
        
        if not migration.down:
            logger.error(f"Migration has no rollback function: {migration_id}")
            return False
        
        try:
            # Run rollback
            if asyncio.iscoroutinefunction(migration.down):
                await migration.down()
            else:
                migration.down()
            
            # Remove from history
            self._history = [
                entry for entry in self._history
                if entry.get("migration_id") != migration_id
            ]
            self._save_history()
            
            logger.info(f"Migration rolled back: {migration_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {migration_id} - {e}")
            return False



