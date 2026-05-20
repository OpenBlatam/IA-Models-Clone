"""
Database Testing Helpers
Specialized helpers for database testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import asyncio
from datetime import datetime


class DatabaseTestHelpers:
    """Helpers for database testing"""
    
    @staticmethod
    def create_mock_db_session(
        return_value: Any = None,
        side_effect: Any = None
    ) -> Mock:
        """Create mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.refresh = AsyncMock()
        session.delete = Mock()
        session.query = Mock()
        session.execute = AsyncMock(return_value=return_value)
        
        if side_effect:
            session.execute.side_effect = side_effect
        
        return session
    
    @staticmethod
    def create_mock_query_builder() -> Mock:
        """Create mock query builder with chaining"""
        query = Mock()
        
        # Chain methods
        query.filter = Mock(return_value=query)
        query.order_by = Mock(return_value=query)
        query.limit = Mock(return_value=query)
        query.offset = Mock(return_value=query)
        query.join = Mock(return_value=query)
        query.group_by = Mock(return_value=query)
        query.having = Mock(return_value=query)
        
        # Terminal methods
        query.first = Mock(return_value=None)
        query.all = Mock(return_value=[])
        query.count = Mock(return_value=0)
        query.one = Mock(return_value=None)
        
        return query
    
    @staticmethod
    def assert_transaction_committed(session: Mock):
        """Assert transaction was committed"""
        assert session.commit.called, "Transaction was not committed"
    
    @staticmethod
    def assert_transaction_rolled_back(session: Mock):
        """Assert transaction was rolled back"""
        assert session.rollback.called, "Transaction was not rolled back"
    
    @staticmethod
    def assert_entity_added(session: Mock, entity: Any = None):
        """Assert entity was added to session"""
        assert session.add.called, "Entity was not added to session"
        if entity:
            session.add.assert_called_with(entity)
    
    @staticmethod
    def assert_query_executed(session: Mock, expected_query: Optional[str] = None):
        """Assert query was executed"""
        assert session.execute.called or session.query.called, "Query was not executed"
        if expected_query:
            # Additional validation can be added
            pass


class MigrationHelpers:
    """Helpers for migration testing"""
    
    @staticmethod
    def create_mock_migration(
        version: str = "001",
        description: str = "test migration"
    ) -> Mock:
        """Create mock migration"""
        migration = Mock()
        migration.version = version
        migration.description = description
        migration.up = AsyncMock()
        migration.down = AsyncMock()
        return migration
    
    @staticmethod
    def assert_migration_applied(migration: Mock):
        """Assert migration was applied"""
        assert migration.up.called, "Migration up was not called"
    
    @staticmethod
    def assert_migration_rolled_back(migration: Mock):
        """Assert migration was rolled back"""
        assert migration.down.called, "Migration down was not called"


class ConnectionPoolHelpers:
    """Helpers for connection pool testing"""
    
    @staticmethod
    def create_mock_connection_pool(
        max_connections: int = 10,
        current_connections: int = 0
    ) -> Mock:
        """Create mock connection pool"""
        pool = Mock()
        pool.max_connections = max_connections
        pool.current_connections = current_connections
        pool.get_connection = AsyncMock(return_value=Mock())
        pool.release_connection = AsyncMock()
        pool.close_all = AsyncMock()
        pool.get_stats = Mock(return_value={
            "max": max_connections,
            "current": current_connections,
            "available": max_connections - current_connections
        })
        return pool
    
    @staticmethod
    def assert_connection_acquired(pool: Mock):
        """Assert connection was acquired"""
        assert pool.get_connection.called, "Connection was not acquired"
    
    @staticmethod
    def assert_connection_released(pool: Mock):
        """Assert connection was released"""
        assert pool.release_connection.called, "Connection was not released"


class TransactionHelpers:
    """Helpers for transaction testing"""
    
    @staticmethod
    async def test_transaction(
        session: Mock,
        operations: List[Callable],
        should_commit: bool = True
    ) -> Dict[str, Any]:
        """Test transaction with operations"""
        results = []
        errors = []
        
        try:
            for operation in operations:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(session)
                else:
                    result = operation(session)
                results.append(result)
            
            if should_commit:
                await session.commit()
            else:
                await session.rollback()
            
            return {
                "success": True,
                "results": results,
                "committed": should_commit
            }
        except Exception as e:
            await session.rollback()
            return {
                "success": False,
                "error": str(e),
                "results": results,
                "committed": False
            }
    
    @staticmethod
    def assert_transaction_success(transaction_result: Dict[str, Any]):
        """Assert transaction was successful"""
        assert transaction_result["success"], "Transaction failed"
        assert transaction_result["committed"], "Transaction was not committed"


# Convenience exports
create_mock_db_session = DatabaseTestHelpers.create_mock_db_session
create_mock_query_builder = DatabaseTestHelpers.create_mock_query_builder
assert_transaction_committed = DatabaseTestHelpers.assert_transaction_committed
assert_transaction_rolled_back = DatabaseTestHelpers.assert_transaction_rolled_back
assert_entity_added = DatabaseTestHelpers.assert_entity_added
assert_query_executed = DatabaseTestHelpers.assert_query_executed

create_mock_migration = MigrationHelpers.create_mock_migration
assert_migration_applied = MigrationHelpers.assert_migration_applied
assert_migration_rolled_back = MigrationHelpers.assert_migration_rolled_back

create_mock_connection_pool = ConnectionPoolHelpers.create_mock_connection_pool
assert_connection_acquired = ConnectionPoolHelpers.assert_connection_acquired
assert_connection_released = ConnectionPoolHelpers.assert_connection_released

test_transaction = TransactionHelpers.test_transaction
assert_transaction_success = TransactionHelpers.assert_transaction_success



