from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import uuid
import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from async_database_api_operations import (
from typing import Any, List, Dict, Optional
import logging
"""
Test Suite for Async Database and API Operations

This test suite covers:
- Unit tests for all database and API managers
- Integration tests for operation orchestration
- Performance tests and benchmarks
- Error handling and retry mechanisms
- Connection pooling and resource management
- Context managers and async patterns
"""


    AsyncOperationOrchestrator, AsyncPostgreSQLManager, AsyncSQLiteManager,
    AsyncRedisManager, AsyncAPIManager, OperationContext, OperationResult,
    OperationType, DatabaseType, execute_with_retry, get_database_connection,
    get_api_session
)


# Test data fixtures
@pytest.fixture
def sample_operation_context():
    """Sample operation context for testing."""
    return OperationContext(
        operation_id=str(uuid.uuid4()),
        operation_type=OperationType.DATABASE_READ,
        database_type=DatabaseType.POSTGRESQL,
        user_id="test_user"
    )


@pytest.fixture
def sample_operation_result():
    """Sample operation result for testing."""
    return OperationResult(
        success=True,
        data={"test": "data"},
        execution_time=0.1,
        operation_context=OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=OperationType.DATABASE_READ
        )
    )


@pytest.fixture
def sample_database_query():
    """Sample database query for testing."""
    return {
        "query": "SELECT * FROM users WHERE id = $1",
        "params": {"1": "test_user_123"}
    }


@pytest.fixture
def sample_api_request():
    """Sample API request for testing."""
    return {
        "method": "GET",
        "endpoint": "/users/123",
        "data": None,
        "params": {"include": "profile"}
    }


# Unit Tests

class TestAsyncPostgreSQLManager:
    """Test cases for AsyncPostgreSQLManager."""
    
    @pytest.mark.asyncio
    async def test_manager_creation(self) -> Any:
        """Test AsyncPostgreSQLManager creation."""
        manager = AsyncPostgreSQLManager("postgresql://test:test@localhost:5432/testdb")
        assert manager.connection_string == "postgresql://test:test@localhost:5432/testdb"
        assert manager.max_connections == 10
        assert manager.pool is None
        assert manager.stats["connections_created"] == 0
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self) -> Any:
        """Test pool initialization."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            manager = AsyncPostgreSQLManager("postgresql://test:test@localhost:5432/testdb")
            await manager.initialize_pool()
            
            mock_create_pool.assert_called_once()
            assert manager.pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_pool_closure(self) -> Any:
        """Test pool closure."""
        manager = AsyncPostgreSQLManager("postgresql://test:test@localhost:5432/testdb")
        manager.pool = AsyncMock()
        
        await manager.close_pool()
        manager.pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_execution_success(self) -> Any:
        """Test successful query execution."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_connection = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
            mock_connection.fetch.return_value = [{"id": 1, "name": "test"}]
            mock_create_pool.return_value = mock_pool
            
            manager = AsyncPostgreSQLManager("postgresql://test:test@localhost:5432/testdb")
            await manager.initialize_pool()
            
            result = await manager.execute_query("SELECT * FROM users", {"id": 1})
            
            assert result.success is True
            assert result.data == [{"id": 1, "name": "test"}]
            assert result.execution_time > 0
            assert manager.stats["operations_performed"] == 1
    
    @pytest.mark.asyncio
    async def test_query_execution_failure(self) -> Any:
        """Test failed query execution."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_connection = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
            mock_connection.fetch.side_effect = Exception("Database error")
            mock_create_pool.return_value = mock_pool
            
            manager = AsyncPostgreSQLManager("postgresql://test:test@localhost:5432/testdb")
            await manager.initialize_pool()
            
            result = await manager.execute_query("SELECT * FROM users")
            
            assert result.success is False
            assert "Database error" in result.error
            assert manager.stats["errors"] == 1


class TestAsyncSQLiteManager:
    """Test cases for AsyncSQLiteManager."""
    
    @pytest.mark.asyncio
    async def test_manager_creation(self) -> Any:
        """Test AsyncSQLiteManager creation."""
        manager = AsyncSQLiteManager("test.db")
        assert manager.connection_string == "test.db"
        assert manager.max_connections == 10
        assert manager.pool == "test.db"
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self) -> Any:
        """Test pool initialization."""
        manager = AsyncSQLiteManager("test.db")
        await manager.initialize_pool()
        assert manager.pool == "test.db"
    
    @pytest.mark.asyncio
    async def test_query_execution_success(self) -> Any:
        """Test successful query execution."""
        with patch('aiosqlite.connect') as mock_connect:
            mock_db = AsyncMock()
            mock_cursor = AsyncMock()
            mock_db.execute.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [{"id": 1, "name": "test"}]
            mock_connect.return_value.__aenter__.return_value = mock_db
            
            manager = AsyncSQLiteManager("test.db")
            
            result = await manager.execute_query("SELECT * FROM users", {"id": 1})
            
            assert result.success is True
            assert result.data == [{"id": 1, "name": "test"}]
            assert result.execution_time > 0
            assert manager.stats["operations_performed"] == 1
    
    @pytest.mark.asyncio
    async def test_query_execution_failure(self) -> Any:
        """Test failed query execution."""
        with patch('aiosqlite.connect') as mock_connect:
            mock_db = AsyncMock()
            mock_db.execute.side_effect = Exception("SQLite error")
            mock_connect.return_value.__aenter__.return_value = mock_db
            
            manager = AsyncSQLiteManager("test.db")
            
            result = await manager.execute_query("SELECT * FROM users")
            
            assert result.success is False
            assert "SQLite error" in result.error
            assert manager.stats["errors"] == 1


class TestAsyncRedisManager:
    """Test cases for AsyncRedisManager."""
    
    @pytest.mark.asyncio
    async def test_manager_creation(self) -> Any:
        """Test AsyncRedisManager creation."""
        manager = AsyncRedisManager("redis://localhost:6379")
        assert manager.connection_string == "redis://localhost:6379"
        assert manager.max_connections == 10
        assert manager.pool is None
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self) -> Any:
        """Test pool initialization."""
        with patch('aioredis.from_url') as mock_from_url:
            mock_pool = AsyncMock()
            mock_from_url.return_value = mock_pool
            
            manager = AsyncRedisManager("redis://localhost:6379")
            await manager.initialize_pool()
            
            mock_from_url.assert_called_once()
            assert manager.pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_query_execution_success(self) -> Any:
        """Test successful query execution."""
        with patch('aioredis.from_url') as mock_from_url:
            mock_pool = AsyncMock()
            mock_pool.get.return_value = '{"name": "test"}'
            mock_from_url.return_value = mock_pool
            
            manager = AsyncRedisManager("redis://localhost:6379")
            await manager.initialize_pool()
            
            result = await manager.execute_query("user:123")
            
            assert result.success is True
            assert result.data == '{"name": "test"}'
            assert result.execution_time > 0
            assert manager.stats["operations_performed"] == 1
    
    @pytest.mark.asyncio
    async def test_query_execution_failure(self) -> Any:
        """Test failed query execution."""
        with patch('aioredis.from_url') as mock_from_url:
            mock_pool = AsyncMock()
            mock_pool.get.side_effect = Exception("Redis error")
            mock_from_url.return_value = mock_pool
            
            manager = AsyncRedisManager("redis://localhost:6379")
            await manager.initialize_pool()
            
            result = await manager.execute_query("user:123")
            
            assert result.success is False
            assert "Redis error" in result.error
            assert manager.stats["errors"] == 1


class TestAsyncAPIManager:
    """Test cases for AsyncAPIManager."""
    
    @pytest.mark.asyncio
    async def test_manager_creation(self) -> Any:
        """Test AsyncAPIManager creation."""
        manager = AsyncAPIManager("https://api.example.com")
        assert manager.base_url == "https://api.example.com"
        assert manager.max_connections == 20
        assert manager.timeout == 30
        assert manager.session is None
    
    @pytest.mark.asyncio
    async def test_session_initialization(self) -> Any:
        """Test session initialization."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            manager = AsyncAPIManager("https://api.example.com")
            await manager.initialize_session()
            
            mock_session_class.assert_called_once()
            assert manager.session == mock_session
    
    @pytest.mark.asyncio
    async def test_session_closure(self) -> Any:
        """Test session closure."""
        manager = AsyncAPIManager("https://api.example.com")
        manager.session = AsyncMock()
        
        await manager.close_session()
        manager.session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async async def test_get_request_success(self) -> Optional[Dict[str, Any]]:
        """Test successful GET request."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"id": 1, "name": "test"}
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            manager = AsyncAPIManager("https://api.example.com")
            await manager.initialize_session()
            
            result = await manager.get("/users/1")
            
            assert result.success is True
            assert result.data == {"id": 1, "name": "test"}
            assert result.execution_time > 0
            assert manager.stats["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async async def test_post_request_success(self) -> Any:
        """Test successful POST request."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json.return_value = {"id": 2, "created": True}
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            manager = AsyncAPIManager("https://api.example.com")
            await manager.initialize_session()
            
            result = await manager.post("/users", {"name": "new_user"})
            
            assert result.success is True
            assert result.data == {"id": 2, "created": True}
            assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async async def test_request_failure(self) -> Any:
        """Test failed request."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.request.side_effect = Exception("Network error")
            mock_session_class.return_value = mock_session
            
            manager = AsyncAPIManager("https://api.example.com")
            await manager.initialize_session()
            
            result = await manager.get("/users/1")
            
            assert result.success is False
            assert "Network error" in result.error
            assert manager.stats["failed_requests"] == 1
    
    @pytest.mark.asyncio
    async async def test_http_error_response(self) -> Any:
        """Test HTTP error response."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.json.return_value = {"error": "Not found"}
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            manager = AsyncAPIManager("https://api.example.com")
            await manager.initialize_session()
            
            result = await manager.get("/users/999")
            
            assert result.success is False
            assert "HTTP 404" in result.error
            assert manager.stats["failed_requests"] == 1


class TestAsyncOperationOrchestrator:
    """Test cases for AsyncOperationOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_creation(self) -> Any:
        """Test AsyncOperationOrchestrator creation."""
        orchestrator = AsyncOperationOrchestrator()
        assert len(orchestrator.database_managers) == 0
        assert len(orchestrator.api_managers) == 0
        assert len(orchestrator.operation_history) == 0
        assert orchestrator.performance_stats["total_operations"] == 0
    
    @pytest.mark.asyncio
    async def test_add_database_manager(self) -> Any:
        """Test adding database manager."""
        with patch('async_database_api_operations.AsyncPostgreSQLManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            orchestrator = AsyncOperationOrchestrator()
            await orchestrator.add_database_manager(
                "test_db",
                DatabaseType.POSTGRESQL,
                "postgresql://test:test@localhost:5432/testdb"
            )
            
            assert "test_db" in orchestrator.database_managers
            mock_manager.initialize_pool.assert_called_once()
    
    @pytest.mark.asyncio
    async async def test_add_api_manager(self) -> Any:
        """Test adding API manager."""
        with patch('async_database_api_operations.AsyncAPIManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            orchestrator = AsyncOperationOrchestrator()
            await orchestrator.add_api_manager("test_api", "https://api.example.com")
            
            assert "test_api" in orchestrator.api_managers
            mock_manager.initialize_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_database_operation(self) -> Any:
        """Test database operation execution."""
        mock_manager = AsyncMock()
        mock_result = OperationResult(
            success=True,
            data={"test": "data"},
            execution_time=0.1,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.DATABASE_READ
            )
        )
        mock_manager.execute_query.return_value = mock_result
        
        orchestrator = AsyncOperationOrchestrator()
        orchestrator.database_managers["test_db"] = mock_manager
        
        result = await orchestrator.execute_database_operation(
            "test_db",
            "SELECT * FROM users",
            {"id": 1}
        )
        
        assert result == mock_result
        assert len(orchestrator.operation_history) == 1
        assert orchestrator.performance_stats["total_operations"] == 1
    
    @pytest.mark.asyncio
    async async def test_execute_api_operation(self) -> Any:
        """Test API operation execution."""
        mock_manager = AsyncMock()
        mock_result = OperationResult(
            success=True,
            data={"id": 1, "name": "test"},
            execution_time=0.2,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.API_GET
            )
        )
        mock_manager.get.return_value = mock_result
        
        orchestrator = AsyncOperationOrchestrator()
        orchestrator.api_managers["test_api"] = mock_manager
        
        result = await orchestrator.execute_api_operation(
            "test_api",
            "GET",
            "/users/1"
        )
        
        assert result == mock_result
        assert len(orchestrator.operation_history) == 1
        assert orchestrator.performance_stats["total_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_batch_operations(self) -> Any:
        """Test batch operation execution."""
        mock_db_manager = AsyncMock()
        mock_api_manager = AsyncMock()
        
        mock_db_result = OperationResult(
            success=True,
            data={"db": "result"},
            execution_time=0.1,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.DATABASE_READ
            )
        )
        mock_api_result = OperationResult(
            success=True,
            data={"api": "result"},
            execution_time=0.2,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.API_GET
            )
        )
        
        mock_db_manager.execute_query.return_value = mock_db_result
        mock_api_manager.get.return_value = mock_api_result
        
        orchestrator = AsyncOperationOrchestrator()
        orchestrator.database_managers["test_db"] = mock_db_manager
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        operations = [
            {
                "type": "database",
                "db_name": "test_db",
                "query": "SELECT * FROM users",
                "params": {"id": 1}
            },
            {
                "type": "api",
                "api_name": "test_api",
                "method": "GET",
                "endpoint": "/users/1"
            }
        ]
        
        results = await orchestrator.execute_batch_operations(operations)
        
        assert len(results) == 2
        assert results[0] == mock_db_result
        assert results[1] == mock_api_result
        assert len(orchestrator.operation_history) == 2
        assert orchestrator.performance_stats["total_operations"] == 2
    
    @pytest.mark.asyncio
    async def test_get_performance_stats(self) -> Optional[Dict[str, Any]]:
        """Test performance statistics retrieval."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Add mock managers
        mock_db_manager = AsyncMock()
        mock_db_manager.stats = {"operations": 5, "errors": 1}
        orchestrator.database_managers["test_db"] = mock_db_manager
        
        mock_api_manager = AsyncMock()
        mock_api_manager.stats = {"requests": 10, "successful": 8}
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        stats = orchestrator.get_performance_stats()
        
        assert "orchestrator" in stats
        assert "databases" in stats
        assert "apis" in stats
        assert "test_db" in stats["databases"]
        assert "test_api" in stats["apis"]
    
    @pytest.mark.asyncio
    async def test_close_all(self) -> Any:
        """Test closing all managers."""
        mock_db_manager = AsyncMock()
        mock_api_manager = AsyncMock()
        
        orchestrator = AsyncOperationOrchestrator()
        orchestrator.database_managers["test_db"] = mock_db_manager
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        await orchestrator.close_all()
        
        mock_db_manager.close_pool.assert_called_once()
        mock_api_manager.close_session.assert_called_once()


# Integration Tests

class TestIntegration:
    """Integration tests for async operations."""
    
    @pytest.mark.asyncio
    async async def test_database_api_integration(self) -> Any:
        """Test integration between database and API operations."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Mock database manager
        mock_db_manager = AsyncMock()
        mock_db_result = OperationResult(
            success=True,
            data=[{"id": 1, "name": "user1"}],
            execution_time=0.1,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.DATABASE_READ
            )
        )
        mock_db_manager.execute_query.return_value = mock_db_result
        orchestrator.database_managers["test_db"] = mock_db_manager
        
        # Mock API manager
        mock_api_manager = AsyncMock()
        mock_api_result = OperationResult(
            success=True,
            data={"profile": "data"},
            execution_time=0.2,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.API_GET
            )
        )
        mock_api_manager.get.return_value = mock_api_result
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        # Execute integrated operations
        operations = [
            {
                "type": "database",
                "db_name": "test_db",
                "query": "SELECT * FROM users WHERE id = $1",
                "params": {"1": 1}
            },
            {
                "type": "api",
                "api_name": "test_api",
                "method": "GET",
                "endpoint": "/users/1/profile"
            }
        ]
        
        results = await orchestrator.execute_batch_operations(operations)
        
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is True
        assert orchestrator.performance_stats["total_operations"] == 2
        assert orchestrator.performance_stats["successful_operations"] == 2


# Performance Tests

class TestPerformance:
    """Performance tests for async operations."""
    
    @pytest.mark.asyncio
    async def test_database_performance(self) -> Any:
        """Test database operation performance."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Mock database manager
        mock_db_manager = AsyncMock()
        mock_db_manager.execute_query.return_value = OperationResult(
            success=True,
            data={"test": "data"},
            execution_time=0.01,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.DATABASE_READ
            )
        )
        orchestrator.database_managers["test_db"] = mock_db_manager
        
        # Generate operations
        operations = [
            {
                "type": "database",
                "db_name": "test_db",
                "query": f"SELECT * FROM users WHERE id = {i}",
                "params": {"id": i}
            }
            for i in range(100)
        ]
        
        start_time = time.time()
        results = await orchestrator.execute_batch_operations(operations)
        total_time = time.time() - start_time
        
        assert len(results) == 100
        assert all(r.success for r in results)
        assert total_time < 5.0  # Should complete within 5 seconds
        assert orchestrator.performance_stats["total_operations"] == 100
    
    @pytest.mark.asyncio
    async async def test_api_performance(self) -> Any:
        """Test API operation performance."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Mock API manager
        mock_api_manager = AsyncMock()
        mock_api_manager.get.return_value = OperationResult(
            success=True,
            data={"id": 1, "name": "test"},
            execution_time=0.05,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.API_GET
            )
        )
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        # Generate operations
        operations = [
            {
                "type": "api",
                "api_name": "test_api",
                "method": "GET",
                "endpoint": f"/users/{i}"
            }
            for i in range(50)
        ]
        
        start_time = time.time()
        results = await orchestrator.execute_batch_operations(operations)
        total_time = time.time() - start_time
        
        assert len(results) == 50
        assert all(r.success for r in results)
        assert total_time < 10.0  # Should complete within 10 seconds
        assert orchestrator.performance_stats["total_operations"] == 50
    
    @pytest.mark.asyncio
    async def test_mixed_operations_performance(self) -> Any:
        """Test mixed database and API operations performance."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Mock managers
        mock_db_manager = AsyncMock()
        mock_db_manager.execute_query.return_value = OperationResult(
            success=True,
            data={"db": "result"},
            execution_time=0.01,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.DATABASE_READ
            )
        )
        
        mock_api_manager = AsyncMock()
        mock_api_manager.get.return_value = OperationResult(
            success=True,
            data={"api": "result"},
            execution_time=0.05,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.API_GET
            )
        )
        
        orchestrator.database_managers["test_db"] = mock_db_manager
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        # Generate mixed operations
        operations = []
        for i in range(25):
            operations.append({
                "type": "database",
                "db_name": "test_db",
                "query": f"SELECT * FROM users WHERE id = {i}",
                "params": {"id": i}
            })
            operations.append({
                "type": "api",
                "api_name": "test_api",
                "method": "GET",
                "endpoint": f"/users/{i}"
            })
        
        start_time = time.time()
        results = await orchestrator.execute_batch_operations(operations)
        total_time = time.time() - start_time
        
        assert len(results) == 50
        assert all(r.success for r in results)
        assert total_time < 15.0  # Should complete within 15 seconds
        assert orchestrator.performance_stats["total_operations"] == 50


# Error Handling Tests

class TestErrorHandling:
    """Error handling tests for async operations."""
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self) -> Any:
        """Test database error handling."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Mock database manager that fails
        mock_db_manager = AsyncMock()
        mock_db_manager.execute_query.side_effect = Exception("Database connection failed")
        orchestrator.database_managers["test_db"] = mock_db_manager
        
        result = await orchestrator.execute_database_operation(
            "test_db",
            "SELECT * FROM users"
        )
        
        assert result.success is False
        assert "Database connection failed" in result.error
        assert orchestrator.performance_stats["failed_operations"] == 1
    
    @pytest.mark.asyncio
    async async def test_api_error_handling(self) -> Any:
        """Test API error handling."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Mock API manager that fails
        mock_api_manager = AsyncMock()
        mock_api_manager.get.side_effect = Exception("Network timeout")
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        result = await orchestrator.execute_api_operation(
            "test_api",
            "GET",
            "/users/1"
        )
        
        assert result.success is False
        assert "Network timeout" in result.error
        assert orchestrator.performance_stats["failed_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_operations_with_errors(self) -> Any:
        """Test batch operations with some errors."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Mock managers with mixed success/failure
        mock_db_manager = AsyncMock()
        mock_db_manager.execute_query.side_effect = [
            OperationResult(
                success=True,
                data={"success": True},
                execution_time=0.1,
                operation_context=OperationContext(
                    operation_id=str(uuid.uuid4()),
                    operation_type=OperationType.DATABASE_READ
                )
            ),
            Exception("Database error")
        ]
        
        mock_api_manager = AsyncMock()
        mock_api_manager.get.side_effect = [
            Exception("API error"),
            OperationResult(
                success=True,
                data={"success": True},
                execution_time=0.2,
                operation_context=OperationContext(
                    operation_id=str(uuid.uuid4()),
                    operation_type=OperationType.API_GET
                )
            )
        ]
        
        orchestrator.database_managers["test_db"] = mock_db_manager
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        operations = [
            {
                "type": "database",
                "db_name": "test_db",
                "query": "SELECT * FROM users"
            },
            {
                "type": "database",
                "db_name": "test_db",
                "query": "SELECT * FROM users"
            },
            {
                "type": "api",
                "api_name": "test_api",
                "method": "GET",
                "endpoint": "/users/1"
            },
            {
                "type": "api",
                "api_name": "test_api",
                "method": "GET",
                "endpoint": "/users/2"
            }
        ]
        
        results = await orchestrator.execute_batch_operations(operations)
        
        assert len(results) == 4
        successful_ops = sum(1 for r in results if r.success)
        failed_ops = len(results) - successful_ops
        
        assert successful_ops == 2
        assert failed_ops == 2
        assert orchestrator.performance_stats["successful_operations"] == 2
        assert orchestrator.performance_stats["failed_operations"] == 2


# Utility Function Tests

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self) -> Any:
        """Test retry mechanism with eventual success."""
        call_count = 0
        
        async def failing_operation():
            
    """failing_operation function."""
nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return OperationResult(
                success=True,
                data={"success": True},
                execution_time=0.1,
                operation_context=OperationContext(
                    operation_id=str(uuid.uuid4()),
                    operation_type=OperationType.DATABASE_READ
                )
            )
        
        result = await execute_with_retry(
            failing_operation,
            max_retries=3,
            delay=0.1
        )
        
        assert result.success is True
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure(self) -> Any:
        """Test retry mechanism with eventual failure."""
        async def always_failing_operation():
            
    """always_failing_operation function."""
raise Exception("Persistent failure")
        
        result = await execute_with_retry(
            always_failing_operation,
            max_retries=2,
            delay=0.1
        )
        
        assert result.success is False
        assert "Persistent failure" in result.error
    
    @pytest.mark.asyncio
    async def test_get_database_connection(self) -> Optional[Dict[str, Any]]:
        """Test database connection context manager."""
        mock_manager = AsyncMock()
        
        with patch('async_database_api_operations.orchestrator') as mock_orchestrator:
            mock_orchestrator.database_managers = {"test_db": mock_manager}
            
            async with get_database_connection("test_db") as manager:
                assert manager == mock_manager
    
    @pytest.mark.asyncio
    async async def test_get_api_session(self) -> Optional[Dict[str, Any]]:
        """Test API session context manager."""
        mock_manager = AsyncMock()
        
        with patch('async_database_api_operations.orchestrator') as mock_orchestrator:
            mock_orchestrator.api_managers = {"test_api": mock_manager}
            
            async with get_api_session("test_api") as manager:
                assert manager == mock_manager


# Benchmark Tests

class TestBenchmarks:
    """Benchmark tests for performance comparison."""
    
    @pytest.mark.asyncio
    async async def test_benchmark_database_vs_api(self) -> Any:
        """Benchmark database vs API operations."""
        orchestrator = AsyncOperationOrchestrator()
        
        # Mock database manager
        mock_db_manager = AsyncMock()
        mock_db_manager.execute_query.return_value = OperationResult(
            success=True,
            data={"db": "result"},
            execution_time=0.01,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.DATABASE_READ
            )
        )
        orchestrator.database_managers["test_db"] = mock_db_manager
        
        # Mock API manager
        mock_api_manager = AsyncMock()
        mock_api_manager.get.return_value = OperationResult(
            success=True,
            data={"api": "result"},
            execution_time=0.05,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.API_GET
            )
        )
        orchestrator.api_managers["test_api"] = mock_api_manager
        
        # Benchmark database operations
        db_operations = [
            {
                "type": "database",
                "db_name": "test_db",
                "query": f"SELECT * FROM users WHERE id = {i}",
                "params": {"id": i}
            }
            for i in range(50)
        ]
        
        db_start = time.time()
        db_results = await orchestrator.execute_batch_operations(db_operations)
        db_time = time.time() - db_start
        
        # Benchmark API operations
        api_operations = [
            {
                "type": "api",
                "api_name": "test_api",
                "method": "GET",
                "endpoint": f"/users/{i}"
            }
            for i in range(50)
        ]
        
        api_start = time.time()
        api_results = await orchestrator.execute_batch_operations(api_operations)
        api_time = time.time() - api_start
        
        # Compare results
        assert len(db_results) == 50
        assert len(api_results) == 50
        assert all(r.success for r in db_results)
        assert all(r.success for r in api_results)
        
        print(f"Database operations: {db_time:.3f}s")
        print(f"API operations: {api_time:.3f}s")
        
        # Database operations should be faster
        assert db_time < api_time


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 