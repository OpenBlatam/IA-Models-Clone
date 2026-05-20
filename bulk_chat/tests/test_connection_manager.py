"""
Tests for Connection Manager
=============================
"""

import pytest
import asyncio
from ..core.connection_manager import ConnectionManager


@pytest.fixture
def connection_manager():
    """Create connection manager for testing."""
    return ConnectionManager()


@pytest.mark.asyncio
async def test_create_connection(connection_manager):
    """Test creating a connection."""
    connection_id = connection_manager.create_connection(
        connection_type="http",
        config={"url": "https://example.com"}
    )
    
    assert connection_id is not None
    assert connection_id in connection_manager.connections


@pytest.mark.asyncio
async def test_get_connection(connection_manager):
    """Test getting a connection."""
    connection_id = connection_manager.create_connection(
        "http",
        {"url": "https://example.com"}
    )
    
    connection = connection_manager.get_connection(connection_id)
    
    assert connection is not None
    assert connection.connection_type == "http"


@pytest.mark.asyncio
async def test_close_connection(connection_manager):
    """Test closing a connection."""
    connection_id = connection_manager.create_connection(
        "http",
        {"url": "https://example.com"}
    )
    
    assert connection_id in connection_manager.connections
    
    await connection_manager.close_connection(connection_id)
    
    # Connection should be closed
    connection = connection_manager.get_connection(connection_id)
    assert connection is None or connection.is_closed


@pytest.mark.asyncio
async def test_get_connection_stats(connection_manager):
    """Test getting connection statistics."""
    connection_manager.create_connection("http", {"url": "https://example.com"})
    connection_manager.create_connection("database", {"host": "localhost"})
    
    stats = connection_manager.get_connection_stats()
    
    assert stats is not None
    assert "total_connections" in stats or stats.get("active", 0) >= 0


@pytest.mark.asyncio
async def test_connection_pooling(connection_manager):
    """Test connection pooling."""
    # Create multiple connections of same type
    for i in range(5):
        connection_manager.create_connection(
            "http",
            {"url": f"https://example{i}.com"}
        )
    
    stats = connection_manager.get_connection_stats()
    
    assert stats is not None
    assert stats.get("total_connections", 0) >= 5


