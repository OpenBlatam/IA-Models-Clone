"""
Migration Manager for Piel Mejorador AI SAM3
===========================================

Data migration system.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
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
    version: str
    name: str
    up: Callable
    down: Optional[Callable] = None
    description: str = ""
    dependencies: List[str] = field(default_factory=list)


class MigrationManager:
    """
    Manages data migrations.
    
    Features:
    - Version-based migrations
    - Up/down migrations
    - Dependency tracking
    - Rollback support
    """
    
    def __init__(self):
        """Initialize migration manager."""
        self._migrations: Dict[str, Migration] = {}
        self._executed_migrations: List[str] = []
        self._migration_history: List[Dict[str, Any]] = []
    
    def register_migration(self, migration: Migration):
        """
        Register a migration.
        
        Args:
            migration: Migration definition
        """
        self._migrations[migration.version] = migration
        logger.info(f"Registered migration: {migration.version} - {migration.name}")
    
    async def migrate(self, target_version: Optional[str] = None) -> List[str]:
        """
        Run migrations.
        
        Args:
            target_version: Optional target version (all if None)
            
        Returns:
            List of executed migration versions
        """
        # Sort migrations by version
        sorted_versions = sorted(self._migrations.keys())
        
        executed = []
        for version in sorted_versions:
            if version in self._executed_migrations:
                continue
            
            if target_version and version > target_version:
                break
            
            migration = self._migrations[version]
            
            # Check dependencies
            for dep in migration.dependencies:
                if dep not in self._executed_migrations:
                    logger.warning(f"Migration {version} depends on {dep} which hasn't been executed")
                    continue
            
            # Execute migration
            try:
                logger.info(f"Running migration: {version} - {migration.name}")
                await migration.up()
                
                self._executed_migrations.append(version)
                self._migration_history.append({
                    "version": version,
                    "name": migration.name,
                    "status": MigrationStatus.COMPLETED.value,
                    "timestamp": datetime.now().isoformat(),
                })
                
                executed.append(version)
                
            except Exception as e:
                logger.error(f"Migration {version} failed: {e}")
                self._migration_history.append({
                    "version": version,
                    "name": migration.name,
                    "status": MigrationStatus.FAILED.value,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                raise
        
        return executed
    
    async def rollback(self, target_version: Optional[str] = None) -> List[str]:
        """
        Rollback migrations.
        
        Args:
            target_version: Optional target version (all if None)
            
        Returns:
            List of rolled back migration versions
        """
        # Sort in reverse order
        sorted_versions = sorted(self._executed_migrations, reverse=True)
        
        rolled_back = []
        for version in sorted_versions:
            if target_version and version <= target_version:
                break
            
            if version not in self._migrations:
                continue
            
            migration = self._migrations[version]
            
            if not migration.down:
                logger.warning(f"Migration {version} has no rollback function")
                continue
            
            try:
                logger.info(f"Rolling back migration: {version} - {migration.name}")
                await migration.down()
                
                self._executed_migrations.remove(version)
                self._migration_history.append({
                    "version": version,
                    "name": migration.name,
                    "status": MigrationStatus.ROLLED_BACK.value,
                    "timestamp": datetime.now().isoformat(),
                })
                
                rolled_back.append(version)
                
            except Exception as e:
                logger.error(f"Rollback {version} failed: {e}")
                raise
        
        return rolled_back
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status."""
        return {
            "total_migrations": len(self._migrations),
            "executed_migrations": len(self._executed_migrations),
            "pending_migrations": len(self._migrations) - len(self._executed_migrations),
            "executed_versions": self._executed_migrations,
            "history": self._migration_history[-10:],  # Last 10
        }




