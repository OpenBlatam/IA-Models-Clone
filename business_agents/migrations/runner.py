"""
Migration Runner
================

Executes database migrations with proper error handling and rollback support.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .types import Migration, MigrationResult, MigrationStatus, MigrationPlan, BaseMigration
from .version import VersionManager

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Executes database migrations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.version_manager = VersionManager(session)
    
    async def initialize(self):
        """Initialize the migration runner."""
        await self.version_manager.initialize()
        logger.info("Migration runner initialized")
    
    async def run_migrations(
        self, 
        migrations: List[Migration], 
        target_version: Optional[str] = None,
        dry_run: bool = False
    ) -> List[MigrationResult]:
        """Run migrations up to a target version."""
        try:
            # Create migration plan
            plan = MigrationPlan()
            for migration in migrations:
                plan.add_migration(migration)
            
            # Calculate execution order
            execution_order = plan.calculate_execution_order()
            
            # Filter migrations to run
            migrations_to_run = []
            current_version = await self.version_manager.get_current_version()
            
            for version in execution_order:
                migration = plan.get_migration_by_version(version)
                if not migration:
                    continue
                
                # Skip if already applied
                if await self.version_manager.is_version_applied(version):
                    continue
                
                # Stop at target version
                if target_version and version > target_version:
                    break
                
                migrations_to_run.append(migration)
            
            if not migrations_to_run:
                logger.info("No migrations to run")
                return []
            
            logger.info(f"Running {len(migrations_to_run)} migrations")
            
            if dry_run:
                return await self._dry_run_migrations(migrations_to_run)
            
            # Execute migrations
            results = []
            for migration in migrations_to_run:
                result = await self._run_migration(migration)
                results.append(result)
                
                if not result.success:
                    logger.error(f"Migration {migration.version} failed, stopping execution")
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to run migrations: {str(e)}")
            raise
    
    async def rollback_migrations(
        self, 
        migrations: List[Migration], 
        target_version: Optional[str] = None,
        dry_run: bool = False
    ) -> List[MigrationResult]:
        """Rollback migrations to a target version."""
        try:
            # Get applied versions in reverse order
            applied_versions = await self.version_manager.get_all_versions()
            applied_versions.reverse()
            
            # Filter versions to rollback
            versions_to_rollback = []
            for version_info in applied_versions:
                if target_version and version_info.version <= target_version:
                    break
                versions_to_rollback.append(version_info.version)
            
            if not versions_to_rollback:
                logger.info("No migrations to rollback")
                return []
            
            logger.info(f"Rolling back {len(versions_to_rollback)} migrations")
            
            if dry_run:
                return await self._dry_run_rollback(versions_to_rollback, migrations)
            
            # Execute rollbacks
            results = []
            for version in versions_to_rollback:
                # Find migration definition
                migration = next(
                    (m for m in migrations if m.version == version), 
                    None
                )
                
                if not migration:
                    logger.warning(f"Migration definition not found for version {version}")
                    continue
                
                result = await self._rollback_migration(migration)
                results.append(result)
                
                if not result.success:
                    logger.error(f"Rollback of {version} failed, stopping execution")
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to rollback migrations: {str(e)}")
            raise
    
    async def _run_migration(self, migration: Migration) -> MigrationResult:
        """Run a single migration."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Running migration: {migration.version} - {migration.name}")
            
            # Validate checksum if migration already exists
            if await self.version_manager.is_version_applied(migration.version):
                if not await self.version_manager.validate_checksum(migration):
                    raise ValueError(f"Migration {migration.version} checksum mismatch")
            
            # Execute migration
            if migration.migration_type.value == "custom":
                # Handle custom migrations
                result = await self._run_custom_migration(migration)
            else:
                # Handle SQL migrations
                result = await self._run_sql_migration(migration)
            
            # Record version if successful
            if result.success:
                checksum = self.version_manager.calculate_checksum(migration)
                await self.version_manager.record_version(migration, checksum)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return MigrationResult(
                version=migration.version,
                success=result.success,
                status=MigrationStatus.COMPLETED if result.success else MigrationStatus.FAILED,
                execution_time=execution_time,
                error_message=result.error_message,
                affected_rows=result.affected_rows,
                metadata=result.metadata
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Migration {migration.version} failed: {str(e)}")
            
            return MigrationResult(
                version=migration.version,
                success=False,
                status=MigrationStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _rollback_migration(self, migration: Migration) -> MigrationResult:
        """Rollback a single migration."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Rolling back migration: {migration.version} - {migration.name}")
            
            # Execute rollback
            if migration.migration_type.value == "custom":
                # Handle custom migrations
                result = await self._rollback_custom_migration(migration)
            else:
                # Handle SQL migrations
                result = await self._rollback_sql_migration(migration)
            
            # Remove version record if successful
            if result.success:
                await self.version_manager.remove_version(migration.version)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return MigrationResult(
                version=migration.version,
                success=result.success,
                status=MigrationStatus.ROLLED_BACK if result.success else MigrationStatus.FAILED,
                execution_time=execution_time,
                error_message=result.error_message,
                affected_rows=result.affected_rows,
                metadata=result.metadata
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Rollback of {migration.version} failed: {str(e)}")
            
            return MigrationResult(
                version=migration.version,
                success=False,
                status=MigrationStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _run_sql_migration(self, migration: Migration) -> MigrationResult:
        """Run a SQL migration."""
        try:
            if not migration.up_sql.strip():
                return MigrationResult(
                    version=migration.version,
                    success=True,
                    status=MigrationStatus.COMPLETED,
                    execution_time=0.0
                )
            
            # Execute SQL
            result = await self.session.execute(text(migration.up_sql))
            await self.session.commit()
            
            return MigrationResult(
                version=migration.version,
                success=True,
                status=MigrationStatus.COMPLETED,
                execution_time=0.0,
                affected_rows=result.rowcount if hasattr(result, 'rowcount') else None
            )
            
        except Exception as e:
            await self.session.rollback()
            raise
    
    async def _rollback_sql_migration(self, migration: Migration) -> MigrationResult:
        """Rollback a SQL migration."""
        try:
            if not migration.down_sql.strip():
                return MigrationResult(
                    version=migration.version,
                    success=True,
                    status=MigrationStatus.ROLLED_BACK,
                    execution_time=0.0
                )
            
            # Execute rollback SQL
            result = await self.session.execute(text(migration.down_sql))
            await self.session.commit()
            
            return MigrationResult(
                version=migration.version,
                success=True,
                status=MigrationStatus.ROLLED_BACK,
                execution_time=0.0,
                affected_rows=result.rowcount if hasattr(result, 'rowcount') else None
            )
            
        except Exception as e:
            await self.session.rollback()
            raise
    
    async def _run_custom_migration(self, migration: Migration) -> MigrationResult:
        """Run a custom migration."""
        try:
            # This would need to be implemented based on your custom migration system
            # For now, we'll assume custom migrations are handled elsewhere
            return MigrationResult(
                version=migration.version,
                success=True,
                status=MigrationStatus.COMPLETED,
                execution_time=0.0
            )
            
        except Exception as e:
            raise
    
    async def _rollback_custom_migration(self, migration: Migration) -> MigrationResult:
        """Rollback a custom migration."""
        try:
            # This would need to be implemented based on your custom migration system
            return MigrationResult(
                version=migration.version,
                success=True,
                status=MigrationStatus.ROLLED_BACK,
                execution_time=0.0
            )
            
        except Exception as e:
            raise
    
    async def _dry_run_migrations(self, migrations: List[Migration]) -> List[MigrationResult]:
        """Simulate running migrations without executing them."""
        results = []
        
        for migration in migrations:
            result = MigrationResult(
                version=migration.version,
                success=True,
                status=MigrationStatus.PENDING,
                execution_time=0.0,
                metadata={"dry_run": True, "action": "would_run"}
            )
            results.append(result)
        
        return results
    
    async def _dry_run_rollback(self, versions: List[str], migrations: List[Migration]) -> List[MigrationResult]:
        """Simulate rolling back migrations without executing them."""
        results = []
        
        for version in versions:
            result = MigrationResult(
                version=version,
                success=True,
                status=MigrationStatus.PENDING,
                execution_time=0.0,
                metadata={"dry_run": True, "action": "would_rollback"}
            )
            results.append(result)
        
        return results
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        try:
            current_version = await self.version_manager.get_current_version()
            all_versions = await self.version_manager.get_all_versions()
            
            return {
                "current_version": current_version,
                "total_migrations": len(all_versions),
                "latest_migration": all_versions[-1].migration_name if all_versions else None,
                "last_applied": all_versions[-1].applied_at.isoformat() if all_versions else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {str(e)}")
            return {}
