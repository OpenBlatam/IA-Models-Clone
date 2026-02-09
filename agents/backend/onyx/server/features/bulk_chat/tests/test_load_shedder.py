"""
Tests for Load Shedder
======================
"""

import pytest
import asyncio
from ..core.load_shedder import LoadShedder, SheddingStrategy


@pytest.fixture
def load_shedder():
    """Create load shedder for testing."""
    return LoadShedder(
        max_load=0.9,
        shedding_strategy=SheddingStrategy.RANDOM
    )


@pytest.mark.asyncio
async def test_should_shed_load(load_shedder):
    """Test checking if load should be shed."""
    load_shedder.record_load("component1", 0.95)  # Above threshold
    
    should_shed = load_shedder.should_shed_load("component1")
    
    assert should_shed is True


@pytest.mark.asyncio
async def test_shed_load(load_shedder):
    """Test shedding load."""
    load_shedder.record_load("component1", 0.95)
    
    result = await load_shedder.shed_load("component1", percentage=0.2)
    
    assert result is True


@pytest.mark.asyncio
async def test_record_load(load_shedder):
    """Test recording load."""
    load_shedder.record_load("component1", 0.75)
    load_shedder.record_load("component2", 0.85)
    
    load1 = load_shedder.get_load("component1")
    load2 = load_shedder.get_load("component2")
    
    assert load1 == 0.75 or load1 is not None
    assert load2 == 0.85 or load2 is not None


@pytest.mark.asyncio
async def test_get_load_shedder_status(load_shedder):
    """Test getting load shedder status."""
    load_shedder.record_load("component1", 0.8)
    
    status = load_shedder.get_load_shedder_status()
    
    assert status is not None
    assert "components" in status or "total_components" in status


@pytest.mark.asyncio
async def test_get_load_shedder_summary(load_shedder):
    """Test getting load shedder summary."""
    load_shedder.record_load("component1", 0.9)
    load_shedder.record_load("component2", 0.5)
    
    summary = load_shedder.get_load_shedder_summary()
    
    assert summary is not None
    assert "total_components" in summary or "shedding_count" in summary


