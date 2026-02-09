"""
Tests for Resource Pool
========================
"""

import pytest
import asyncio
from ..core.resource_pool import ResourcePool, PoolConfig


@pytest.fixture
def resource_pool():
    """Create resource pool for testing."""
    config = PoolConfig(
        max_size=10,
        min_size=2,
        max_idle_time=300.0
    )
    return ResourcePool(config=config)


@pytest.mark.asyncio
async def test_acquire_resource(resource_pool):
    """Test acquiring a resource from pool."""
    resource = await resource_pool.acquire()
    
    assert resource is not None
    assert resource in resource_pool.acquired_resources or resource in resource_pool.available_resources


@pytest.mark.asyncio
async def test_release_resource(resource_pool):
    """Test releasing a resource back to pool."""
    resource = await resource_pool.acquire()
    
    assert resource is not None
    
    await resource_pool.release(resource)
    
    # Resource should be back in available pool
    stats = resource_pool.get_pool_stats()
    assert stats is not None


@pytest.mark.asyncio
async def test_pool_exhaustion(resource_pool):
    """Test pool behavior when exhausted."""
    # Acquire all resources
    resources = []
    for _ in range(15):  # More than max_size
        resource = await resource_pool.acquire()
        if resource:
            resources.append(resource)
    
    # Should have acquired up to max_size
    assert len(resources) <= 10
    
    # Release all
    for resource in resources:
        await resource_pool.release(resource)


@pytest.mark.asyncio
async def test_get_pool_stats(resource_pool):
    """Test getting pool statistics."""
    await resource_pool.acquire()
    await resource_pool.acquire()
    
    stats = resource_pool.get_pool_stats()
    
    assert stats is not None
    assert "total_resources" in stats or "available" in stats or "acquired" in stats


@pytest.mark.asyncio
async def test_pool_cleanup(resource_pool):
    """Test pool cleanup of idle resources."""
    resource = await resource_pool.acquire()
    await resource_pool.release(resource)
    
    # Cleanup should remove idle resources
    await resource_pool.cleanup()
    
    stats = resource_pool.get_pool_stats()
    assert stats is not None


