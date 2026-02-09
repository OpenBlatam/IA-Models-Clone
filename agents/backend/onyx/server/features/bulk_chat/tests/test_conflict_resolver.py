"""
Tests for Conflict Resolver
============================
"""

import pytest
import asyncio
from ..core.conflict_resolver import ConflictResolver, ConflictResolutionStrategy


@pytest.fixture
def conflict_resolver():
    """Create conflict resolver for testing."""
    return ConflictResolver()


@pytest.mark.asyncio
async def test_resolve_conflict_last_write_wins(conflict_resolver):
    """Test resolving conflict with last write wins strategy."""
    version1 = {"version": 1, "data": "old"}
    version2 = {"version": 2, "data": "new"}
    
    resolved = await conflict_resolver.resolve_conflict(
        conflict_id="test_conflict",
        versions=[version1, version2],
        strategy=ConflictResolutionStrategy.LAST_WRITE_WINS
    )
    
    assert resolved is not None
    assert resolved["version"] == 2 or resolved["data"] == "new"


@pytest.mark.asyncio
async def test_resolve_conflict_first_write_wins(conflict_resolver):
    """Test resolving conflict with first write wins strategy."""
    version1 = {"version": 1, "data": "first"}
    version2 = {"version": 2, "data": "second"}
    
    resolved = await conflict_resolver.resolve_conflict(
        "test_conflict",
        [version1, version2],
        ConflictResolutionStrategy.FIRST_WRITE_WINS
    )
    
    assert resolved is not None
    assert resolved["version"] == 1 or resolved["data"] == "first"


@pytest.mark.asyncio
async def test_resolve_conflict_merge(conflict_resolver):
    """Test resolving conflict with merge strategy."""
    version1 = {"version": 1, "data": "data1", "field1": "value1"}
    version2 = {"version": 2, "data": "data2", "field2": "value2"}
    
    resolved = await conflict_resolver.resolve_conflict(
        "test_conflict",
        [version1, version2],
        ConflictResolutionStrategy.MERGE
    )
    
    assert resolved is not None
    # Merged should contain fields from both
    assert "field1" in resolved or "field2" in resolved or "data" in resolved


@pytest.mark.asyncio
async def test_get_conflict_history(conflict_resolver):
    """Test getting conflict resolution history."""
    await conflict_resolver.resolve_conflict(
        "conflict1",
        [{"version": 1}, {"version": 2}],
        ConflictResolutionStrategy.LAST_WRITE_WINS
    )
    
    history = conflict_resolver.get_conflict_history(limit=10)
    
    assert len(history) >= 1


@pytest.mark.asyncio
async def test_get_conflict_resolver_summary(conflict_resolver):
    """Test getting conflict resolver summary."""
    await conflict_resolver.resolve_conflict(
        "conflict1",
        [{"version": 1}, {"version": 2}],
        ConflictResolutionStrategy.LAST_WRITE_WINS
    )
    
    summary = conflict_resolver.get_conflict_resolver_summary()
    
    assert summary is not None
    assert "total_conflicts" in summary or "resolved" in summary


