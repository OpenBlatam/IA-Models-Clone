"""
Tests for Saga Orchestrator
===========================
"""

import pytest
import asyncio
from ..core.saga_orchestrator import SagaOrchestrator, SagaStatus


@pytest.fixture
def saga_orchestrator():
    """Create saga orchestrator for testing."""
    return SagaOrchestrator()


@pytest.mark.asyncio
async def test_create_saga(saga_orchestrator):
    """Test creating a saga."""
    saga_id = saga_orchestrator.create_saga(
        name="test_saga",
        steps=[
            {"name": "step1", "action": "create", "compensation": "delete"},
            {"name": "step2", "action": "update", "compensation": "rollback"},
        ]
    )
    
    assert saga_id is not None
    assert saga_id in saga_orchestrator.sagas


@pytest.mark.asyncio
async def test_execute_saga(saga_orchestrator):
    """Test executing a saga."""
    from unittest.mock import patch
    
    saga_id = saga_orchestrator.create_saga(
        name="test_saga",
        steps=[
            {"name": "step1", "action": "create", "compensation": "delete"},
        ]
    )
    
    # Mock step execution
    with patch.object(saga_orchestrator, '_execute_step', return_value=True):
        result = await saga_orchestrator.execute_saga(saga_id)
        
        assert result is True
        saga = saga_orchestrator.sagas[saga_id]
        assert saga.status == SagaStatus.COMPLETED


@pytest.mark.asyncio
async def test_compensate_saga(saga_orchestrator):
    """Test compensating a saga."""
    from unittest.mock import patch
    
    saga_id = saga_orchestrator.create_saga(
        name="test_saga",
        steps=[
            {"name": "step1", "action": "create", "compensation": "delete"},
        ]
    )
    
    # Mock compensation
    with patch.object(saga_orchestrator, '_compensate_step', return_value=True):
        result = await saga_orchestrator.compensate_saga(saga_id)
        
        assert result is True
        saga = saga_orchestrator.sagas[saga_id]
        assert saga.status == SagaStatus.COMPENSATED


@pytest.mark.asyncio
async def test_get_saga_status(saga_orchestrator):
    """Test getting saga status."""
    saga_id = saga_orchestrator.create_saga(
        name="test_saga",
        steps=[{"name": "step1", "action": "test"}]
    )
    
    status = saga_orchestrator.get_saga_status(saga_id)
    
    assert status == SagaStatus.PENDING


@pytest.mark.asyncio
async def test_saga_not_found(saga_orchestrator):
    """Test error handling for non-existent saga."""
    with pytest.raises(ValueError):
        await saga_orchestrator.execute_saga("non_existent")


@pytest.mark.asyncio
async def test_get_saga_orchestrator_summary(saga_orchestrator):
    """Test getting saga orchestrator summary."""
    saga_orchestrator.create_saga(
        name="test_saga",
        steps=[{"name": "step1", "action": "test"}]
    )
    
    summary = saga_orchestrator.get_saga_orchestrator_summary()
    
    assert summary["total_sagas"] >= 1
    assert "pending" in summary["sagas_by_status"]

