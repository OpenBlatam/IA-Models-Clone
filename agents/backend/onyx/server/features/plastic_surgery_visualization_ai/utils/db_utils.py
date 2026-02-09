"""Database utilities."""

from typing import Any, Dict, List, Optional, Callable, Tuple
from contextlib import asynccontextmanager
import asyncio


class DatabaseConnection:
    """Base database connection interface."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection: Optional[Any] = None
    
    async def connect(self):
        """Connect to database."""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from database."""
        raise NotImplementedError
    
    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute query."""
        raise NotImplementedError
    
    async def fetch_one(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch one row."""
        raise NotImplementedError
    
    async def fetch_all(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Fetch all rows."""
        raise NotImplementedError


class QueryBuilder:
    """SQL query builder."""
    
    def __init__(self, table: str):
        self.table = table
        self.select_fields: List[str] = []
        self.where_conditions: List[str] = []
        self.where_params: Dict[str, Any] = {}
        self.order_by: List[str] = []
        self.limit_value: Optional[int] = None
        self.offset_value: Optional[int] = None
    
    def select(self, *fields: str) -> 'QueryBuilder':
        """Select fields."""
        self.select_fields.extend(fields)
        return self
    
    def where(self, condition: str, value: Any = None) -> 'QueryBuilder':
        """Add where condition."""
        if value is not None:
            param_name = f"param_{len(self.where_params)}"
            self.where_conditions.append(condition.replace("?", f":{param_name}"))
            self.where_params[param_name] = value
        else:
            self.where_conditions.append(condition)
        return self
    
    def order_by_field(self, field: str, desc: bool = False) -> 'QueryBuilder':
        """Add order by."""
        direction = "DESC" if desc else "ASC"
        self.order_by.append(f"{field} {direction}")
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """Set limit."""
        self.limit_value = count
        return self
    
    def offset(self, count: int) -> 'QueryBuilder':
        """Set offset."""
        self.offset_value = count
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build query."""
        fields = ", ".join(self.select_fields) if self.select_fields else "*"
        query = f"SELECT {fields} FROM {self.table}"
        
        if self.where_conditions:
            query += " WHERE " + " AND ".join(self.where_conditions)
        
        if self.order_by:
            query += " ORDER BY " + ", ".join(self.order_by)
        
        if self.limit_value:
            query += f" LIMIT {self.limit_value}"
        
        if self.offset_value:
            query += f" OFFSET {self.offset_value}"
        
        return query, self.where_params


class Transaction:
    """Transaction manager."""
    
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
        self.committed = False
        self.rolled_back = False
    
    async def __aenter__(self):
        """Start transaction."""
        await self.connection.execute("BEGIN")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End transaction."""
        if exc_type:
            await self.rollback()
        elif not self.committed:
            await self.commit()
    
    async def commit(self):
        """Commit transaction."""
        await self.connection.execute("COMMIT")
        self.committed = True
    
    async def rollback(self):
        """Rollback transaction."""
        await self.connection.execute("ROLLBACK")
        self.rolled_back = True


class ConnectionPool:
    """Database connection pool."""
    
    def __init__(
        self,
        connection_factory: Callable,
        min_size: int = 5,
        max_size: int = 20
    ):
        self.connection_factory = connection_factory
        self.min_size = min_size
        self.max_size = max_size
        self.pool: List[Any] = []
        self.in_use: set = set()
        self.semaphore = asyncio.Semaphore(max_size)
    
    async def acquire(self) -> Any:
        """Acquire connection from pool."""
        await self.semaphore.acquire()
        
        if self.pool:
            conn = self.pool.pop()
        else:
            conn = await self.connection_factory()
        
        self.in_use.add(id(conn))
        return conn
    
    async def release(self, conn: Any):
        """Release connection to pool."""
        conn_id = id(conn)
        if conn_id in self.in_use:
            self.in_use.discard(conn_id)
            
            if len(self.pool) < self.max_size:
                self.pool.append(conn)
            else:
                await conn.close()
            
            self.semaphore.release()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection context manager."""
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)
    
    async def close_all(self):
        """Close all connections."""
        for conn in self.pool:
            await conn.close()
        self.pool.clear()


class Migration:
    """Database migration."""
    
    def __init__(self, version: int, name: str):
        self.version = version
        self.name = name
    
    async def up(self, connection: DatabaseConnection):
        """Apply migration."""
        raise NotImplementedError
    
    async def down(self, connection: DatabaseConnection):
        """Rollback migration."""
        raise NotImplementedError


class MigrationManager:
    """Manage database migrations."""
    
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
        self.migrations: List[Migration] = []
    
    def register(self, migration: Migration):
        """Register migration."""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    async def migrate(self):
        """Run all pending migrations."""
        current_version = await self.get_current_version()
        
        for migration in self.migrations:
            if migration.version > current_version:
                async with Transaction(self.connection):
                    await migration.up(self.connection)
                    await self.set_version(migration.version)
    
    async def rollback(self, to_version: int):
        """Rollback to version."""
        current_version = await self.get_current_version()
        
        for migration in reversed(self.migrations):
            if migration.version > to_version and migration.version <= current_version:
                async with Transaction(self.connection):
                    await migration.down(self.connection)
                    await self.set_version(migration.version - 1)
    
    async def get_current_version(self) -> int:
        """Get current migration version."""
        # Implementation depends on database
        return 0
    
    async def set_version(self, version: int):
        """Set migration version."""
        # Implementation depends on database
        pass

