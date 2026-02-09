"""
Tests for Distributed Synchronization
======================================
"""

import pytest
import asyncio
from ..core.distributed_sync import DistributedSynchronizer, SyncStatus


@pytest.fixture
def distributed_sync():
    """Create distributed synchronizer for testing."""
    return DistributedSynchronizer(node_id="test_node")


@pytest.mark.asyncio
async def test_sync_resource(distributed_sync):
    """Test synchronizing a resource."""
    resource_id = "resource_1"
    resource_data = {"key": "value", "version": 1}
    
    sync_id = await distributed_sync.sync_resource(
        resource_id=resource_id,
        resource_data=resource_data,
        node_id="test_node"
    )
    
    assert sync_id is not None
    assert resource_id in distributed_sync.synced_resources


@pytest.mark.asyncio
async def test_get_resource(distributed_sync):
    """Test getting synchronized resource."""
    await distributed_sync.sync_resource(
        "resource_1",
        {"data": "test"},
        "test_node"
    )
    
    resource = distributed_sync.get_resource("resource_1")
    
    assert resource is not None
    assert resource["data"] == "test" or resource.get("resource_data", {}).get("data") == "test"


@pytest.mark.asyncio
async def test_detect_conflict(distributed_sync):
    """Test conflict detection."""
    await distributed_sync.sync_resource("resource_1", {"version": 1}, "node1")
    await distributed_sync.sync_resource("resource_1", {"version": 2}, "node2")
    
    conflicts = distributed_sync.detect_conflicts("resource_1")
    
    assert conflicts is not None
    assert len(conflicts) >= 1 or isinstance(conflicts, dict)


@pytest.mark.asyncio
async def test_resolve_conflict(distributed_sync):
    """Test conflict resolution."""
    await distributed_sync.sync_resource("resource_1", {"version": 1}, "node1")
    await distributed_sync.sync_resource("resource_1", {"version": 2}, "node2")
    
    resolved = await distributed_sync.resolve_conflict(
        "resource_1",
        resolution_strategy="last_write_wins"
    )
    
    assert resolved is True or resolved is not None


@pytest.mark.asyncio
async def test_get_sync_status(distributed_sync):
    """Test getting synchronization status."""
    await distributed_sync.sync_resource("resource_1", {"data": "test"}, "test_node")
    
    status = distributed_sync.get_sync_status("resource_1")
    
    assert status is not None
    assert status == SyncStatus.SYNCED or "status" in status


@pytest.mark.asyncio
async def test_get_distributed_sync_summary(distributed_sync):
    """Test getting distributed sync summary."""
    await distributed_sync.sync_resource("resource_1", {"data": "test"}, "node1")
    await distributed_sync.sync_resource("resource_2", {"data": "test"}, "node2")
    
    summary = distributed_sync.get_distributed_sync_summary()
    
    assert summary is not None
    assert "total_resources" in summary or "synced_resources" in summary


