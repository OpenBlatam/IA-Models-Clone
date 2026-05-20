"""
Migration Types and Definitions
===============================

Type definitions for database migrations.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from abc import ABC, abstractmethod

class MigrationStatus(Enum):
    """Migration execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class MigrationType(Enum):
    """Migration type enumeration"""
    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"
    ALTER_TABLE = "alter_table"
    CREATE_INDEX = "create_index"
    DROP_INDEX = "drop_index"
    INSERT_DATA = "insert_data"
    UPDATE_DATA = "update_data"
    DELETE_DATA = "delete_data"
    CUSTOM = "custom"

@dataclass
class Migration:
    """Database migration definition"""
    version: str
    name: str
    description: str
    migration_type: MigrationType
    up_sql: str
    down_sql: str
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MigrationResult:
    """Migration execution result"""
    version: str
    success: bool
    status: MigrationStatus
    execution_time: float
    error_message: Optional[str] = None
    affected_rows: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseMigration(ABC):
    """Base class for custom migrations."""
    
    def __init__(self, version: str, name: str, description: str):
        self.version = version
        self.name = name
        self.description = description
        self.migration_type = MigrationType.CUSTOM
    
    @abstractmethod
    async def up(self, connection) -> MigrationResult:
        """Execute the migration."""
        pass
    
    @abstractmethod
    async def down(self, connection) -> MigrationResult:
        """Rollback the migration."""
        pass
    
    def get_migration(self) -> Migration:
        """Get the migration definition."""
        return Migration(
            version=self.version,
            name=self.name,
            description=self.description,
            migration_type=self.migration_type,
            up_sql="",  # Custom migrations don't use SQL
            down_sql=""
        )

@dataclass
class DatabaseVersion:
    """Database version information"""
    version: str
    applied_at: datetime
    migration_name: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class MigrationPlan:
    """Migration execution plan"""
    
    def __init__(self):
        self.migrations: List[Migration] = []
        self.dependencies: Dict[str, List[str]] = {}
        self.execution_order: List[str] = []
    
    def add_migration(self, migration: Migration):
        """Add a migration to the plan."""
        self.migrations.append(migration)
        if migration.dependencies:
            self.dependencies[migration.version] = migration.dependencies
    
    def calculate_execution_order(self) -> List[str]:
        """Calculate the order in which migrations should be executed."""
        # Topological sort to handle dependencies
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(version: str):
            if version in temp_visited:
                raise ValueError(f"Circular dependency detected involving {version}")
            if version in visited:
                return
            
            temp_visited.add(version)
            
            # Visit dependencies first
            for dep in self.dependencies.get(version, []):
                visit(dep)
            
            temp_visited.remove(version)
            visited.add(version)
            order.append(version)
        
        # Visit all migrations
        for migration in self.migrations:
            if migration.version not in visited:
                visit(migration.version)
        
        self.execution_order = order
        return order
    
    def get_migration_by_version(self, version: str) -> Optional[Migration]:
        """Get migration by version."""
        for migration in self.migrations:
            if migration.version == version:
                return migration
        return None
