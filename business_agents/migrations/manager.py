"""
Migration Manager
=================

Central migration management system.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import importlib.util
import json

from .types import Migration, MigrationType, MigrationPlan
from .runner import MigrationRunner
from .version import VersionManager

logger = logging.getLogger(__name__)

class MigrationManager:
    """Central migration management system."""
    
    def __init__(self, migrations_dir: str = "migrations", session=None):
        self.migrations_dir = Path(migrations_dir)
        self.session = session
        self.runner: Optional[MigrationRunner] = None
        self.version_manager: Optional[VersionManager] = None
        self.migrations: List[Migration] = []
    
    async def initialize(self):
        """Initialize the migration manager."""
        try:
            if self.session:
                self.runner = MigrationRunner(self.session)
                self.version_manager = VersionManager(self.session)
                await self.runner.initialize()
            
            # Load migrations
            await self._load_migrations()
            
            logger.info("Migration manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize migration manager: {str(e)}")
            raise
    
    async def _load_migrations(self):
        """Load all migration files."""
        try:
            self.migrations_dir.mkdir(exist_ok=True)
            
            # Load JSON migration files
            json_files = list(self.migrations_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    migration = await self._load_json_migration(json_file)
                    if migration:
                        self.migrations.append(migration)
                except Exception as e:
                    logger.error(f"Failed to load migration {json_file}: {str(e)}")
            
            # Load Python migration files
            py_files = list(self.migrations_dir.glob("*.py"))
            for py_file in py_files:
                try:
                    migration = await self._load_python_migration(py_file)
                    if migration:
                        self.migrations.append(migration)
                except Exception as e:
                    logger.error(f"Failed to load migration {py_file}: {str(e)}")
            
            # Sort migrations by version
            self.migrations.sort(key=lambda x: x.version)
            
            logger.info(f"Loaded {len(self.migrations)} migrations")
            
        except Exception as e:
            logger.error(f"Failed to load migrations: {str(e)}")
            raise
    
    async def _load_json_migration(self, file_path: Path) -> Optional[Migration]:
        """Load a JSON migration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return Migration(
                version=data["version"],
                name=data["name"],
                description=data["description"],
                migration_type=MigrationType(data["migration_type"]),
                up_sql=data["up_sql"],
                down_sql=data["down_sql"],
                dependencies=data.get("dependencies", []),
                metadata=data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to load JSON migration {file_path}: {str(e)}")
            return None
    
    async def _load_python_migration(self, file_path: Path) -> Optional[Migration]:
        """Load a Python migration file."""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("migration", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get migration data from module
            if not hasattr(module, 'migration_data'):
                logger.warning(f"Migration {file_path} missing migration_data")
                return None
            
            data = module.migration_data
            
            return Migration(
                version=data["version"],
                name=data["name"],
                description=data["description"],
                migration_type=MigrationType(data["migration_type"]),
                up_sql=data["up_sql"],
                down_sql=data["down_sql"],
                dependencies=data.get("dependencies", []),
                metadata=data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to load Python migration {file_path}: {str(e)}")
            return None
    
    async def create_migration(
        self,
        version: str,
        name: str,
        description: str,
        migration_type: MigrationType,
        up_sql: str,
        down_sql: str,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Create a new migration file."""
        try:
            migration = Migration(
                version=version,
                name=name,
                description=description,
                migration_type=migration_type,
                up_sql=up_sql,
                down_sql=down_sql,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            # Create JSON migration file
            filename = f"{version}_{name.replace(' ', '_').lower()}.json"
            file_path = self.migrations_dir / filename
            
            migration_data = {
                "version": migration.version,
                "name": migration.name,
                "description": migration.description,
                "migration_type": migration.migration_type.value,
                "up_sql": migration.up_sql,
                "down_sql": migration.down_sql,
                "dependencies": migration.dependencies,
                "metadata": migration.metadata
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(migration_data, f, indent=2, ensure_ascii=False)
            
            # Add to loaded migrations
            self.migrations.append(migration)
            self.migrations.sort(key=lambda x: x.version)
            
            logger.info(f"Created migration: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create migration: {str(e)}")
            return False
    
    async def run_migrations(
        self, 
        target_version: Optional[str] = None,
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Run migrations up to a target version."""
        if not self.runner:
            raise RuntimeError("Migration runner not initialized")
        
        try:
            results = await self.runner.run_migrations(
                self.migrations, 
                target_version, 
                dry_run
            )
            
            return [
                {
                    "version": result.version,
                    "success": result.success,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                    "affected_rows": result.affected_rows,
                    "metadata": result.metadata
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to run migrations: {str(e)}")
            raise
    
    async def rollback_migrations(
        self, 
        target_version: Optional[str] = None,
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Rollback migrations to a target version."""
        if not self.runner:
            raise RuntimeError("Migration runner not initialized")
        
        try:
            results = await self.runner.rollback_migrations(
                self.migrations, 
                target_version, 
                dry_run
            )
            
            return [
                {
                    "version": result.version,
                    "success": result.success,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                    "affected_rows": result.affected_rows,
                    "metadata": result.metadata
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to rollback migrations: {str(e)}")
            raise
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        if not self.runner:
            return {"error": "Migration runner not initialized"}
        
        try:
            status = await self.runner.get_migration_status()
            status["available_migrations"] = len(self.migrations)
            status["migrations"] = [
                {
                    "version": m.version,
                    "name": m.name,
                    "description": m.description,
                    "type": m.migration_type.value,
                    "dependencies": m.dependencies
                }
                for m in self.migrations
            ]
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {str(e)}")
            return {"error": str(e)}
    
    async def validate_migrations(self) -> Dict[str, Any]:
        """Validate all migrations."""
        try:
            issues = []
            
            # Check for duplicate versions
            versions = [m.version for m in self.migrations]
            duplicates = [v for v in set(versions) if versions.count(v) > 1]
            if duplicates:
                issues.append(f"Duplicate versions: {duplicates}")
            
            # Check dependencies
            for migration in self.migrations:
                for dep in migration.dependencies:
                    if not any(m.version == dep for m in self.migrations):
                        issues.append(f"Migration {migration.version} depends on missing migration {dep}")
            
            # Check for circular dependencies
            plan = MigrationPlan()
            for migration in self.migrations:
                plan.add_migration(migration)
            
            try:
                plan.calculate_execution_order()
            except ValueError as e:
                issues.append(f"Circular dependency: {str(e)}")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "total_migrations": len(self.migrations)
            }
            
        except Exception as e:
            logger.error(f"Failed to validate migrations: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def get_migrations(self) -> List[Migration]:
        """Get all loaded migrations."""
        return self.migrations.copy()
    
    def get_migration_by_version(self, version: str) -> Optional[Migration]:
        """Get a migration by version."""
        return next((m for m in self.migrations if m.version == version), None)
