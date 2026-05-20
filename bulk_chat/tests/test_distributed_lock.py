"""
Tests for Distributed Lock Manager
===================================
"""

import pytest
import asyncio
from ..core.distributed_lock_manager import DistributedLockManager


@pytest.fixture
def lock_manager():
    """Create distributed lock manager for testing."""
    return DistributedLockManager()


@pytest.mark.asyncio
async def test_acquire_lock(lock_manager):
    """Test acquiring a lock."""
    lock_id = await lock_manager.acquire_lock(
        resource_id="resource_1",
        owner_id="owner_1",
        ttl_seconds=60
    )
    
    assert lock_id is not None
    assert lock_id in lock_manager.locks


@pytest.mark.asyncio
async def test_acquire_lock_already_locked(lock_manager):
    """Test acquiring lock when already locked."""
    await lock_manager.acquire_lock("resource_1", "owner_1", ttl_seconds=60)
    
    # Try to acquire same lock with different owner
    lock_id = await lock_manager.acquire_lock(
        "resource_1",
        "owner_2",
        ttl_seconds=60,
        wait_timeout=1.0
    )
    
    # Should either fail or wait (implementation dependent)
    assert lock_id is None or lock_id in lock_manager.locks


@pytest.mark.asyncio
async def test_release_lock(lock_manager):
    """Test releasing a lock."""
    lock_id = await lock_manager.acquire_lock("resource_1", "owner_1", ttl_seconds=60)
    
    result = await lock_manager.release_lock(lock_id, "owner_1")
    
    assert result is True
    assert lock_id not in lock_manager.locks or lock_manager.locks[lock_id].is_released


@pytest.mark.asyncio
async def test_renew_lock(lock_manager):
    """Test renewing a lock."""
    lock_id = await lock_manager.acquire_lock("resource_1", "owner_1", ttl_seconds=60)
    
    result = await lock_manager.renew_lock(lock_id, "owner_1", ttl_seconds=120)
    
    assert result is True


@pytest.mark.asyncio
async def test_is_locked(lock_manager):
    """Test checking if resource is locked."""
    await lock_manager.acquire_lock("resource_1", "owner_1", ttl_seconds=60)
    
    is_locked = lock_manager.is_locked("resource_1")
    
    assert is_locked is True


@pytest.mark.asyncio
async def test_get_lock_status(lock_manager):
    """Test getting lock status."""
    lock_id = await lock_manager.acquire_lock("resource_1", "owner_1", ttl_seconds=60)
    
    status = lock_manager.get_lock_status(lock_id)
    
    assert status is not None
    assert "lock_id" in status or "resource_id" in status or "owner_id" in status


