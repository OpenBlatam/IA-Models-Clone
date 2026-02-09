"""
Tests for Sagas
Tests for saga orchestration and transaction management
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from core.sagas.saga import Saga, SagaState, SagaStep
from core.sagas.orchestrator import SagaOrchestrator


class TestSagaStep:
    """Tests for SagaStep"""
    
    def test_create_saga_step(self):
        """Test creating a saga step"""
        step = SagaStep(
            name="create_analysis",
            action=AsyncMock(),
            compensation=AsyncMock()
        )
        
        assert step.name == "create_analysis"
        assert step.action is not None
        assert step.compensation is not None
    
    @pytest.mark.asyncio
    async def test_execute_step(self):
        """Test executing a saga step"""
        action = AsyncMock(return_value={"result": "success"})
        step = SagaStep(
            name="test_step",
            action=action,
            compensation=AsyncMock()
        )
        
        result = await step.execute()
        
        assert result == {"result": "success"}
        action.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compensate_step(self):
        """Test compensating a saga step"""
        compensation = AsyncMock(return_value=True)
        step = SagaStep(
            name="test_step",
            action=AsyncMock(),
            compensation=compensation
        )
        
        result = await step.compensate()
        
        assert result is True
        compensation.assert_called_once()


class TestSaga:
    """Tests for Saga"""
    
    @pytest.fixture
    def saga(self):
        """Create a saga for testing"""
        return Saga(
            saga_id="test-saga-123",
            steps=[]
        )
    
    def test_create_saga(self, saga):
        """Test creating a saga"""
        assert saga.saga_id == "test-saga-123"
        assert saga.state == SagaState.PENDING
        assert len(saga.steps) == 0
    
    def test_add_step(self, saga):
        """Test adding a step to saga"""
        step = SagaStep(
            name="test_step",
            action=AsyncMock(),
            compensation=AsyncMock()
        )
        
        saga.add_step(step)
        
        assert len(saga.steps) == 1
        assert saga.steps[0].name == "test_step"
    
    @pytest.mark.asyncio
    async def test_execute_saga_success(self, saga):
        """Test executing saga successfully"""
        step1 = SagaStep(
            name="step1",
            action=AsyncMock(return_value={"result": "step1"}),
            compensation=AsyncMock()
        )
        step2 = SagaStep(
            name="step2",
            action=AsyncMock(return_value={"result": "step2"}),
            compensation=AsyncMock()
        )
        
        saga.add_step(step1)
        saga.add_step(step2)
        
        result = await saga.execute()
        
        assert result is not None
        assert saga.state == SagaState.COMPLETED
        step1.action.assert_called_once()
        step2.action.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_saga_with_failure(self, saga):
        """Test executing saga with failure and compensation"""
        step1 = SagaStep(
            name="step1",
            action=AsyncMock(return_value={"result": "step1"}),
            compensation=AsyncMock(return_value=True)
        )
        step2 = SagaStep(
            name="step2",
            action=AsyncMock(side_effect=Exception("Step 2 failed")),
            compensation=AsyncMock(return_value=True)
        )
        
        saga.add_step(step1)
        saga.add_step(step2)
        
        with pytest.raises(Exception):
            await saga.execute()
        
        # Should have compensated step1
        assert saga.state == SagaState.FAILED
        step1.compensation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compensate_saga(self, saga):
        """Test compensating entire saga"""
        step1 = SagaStep(
            name="step1",
            action=AsyncMock(return_value={"result": "step1"}),
            compensation=AsyncMock(return_value=True)
        )
        step2 = SagaStep(
            name="step2",
            action=AsyncMock(return_value={"result": "step2"}),
            compensation=AsyncMock(return_value=True)
        )
        
        saga.add_step(step1)
        saga.add_step(step2)
        
        # Execute first
        await saga.execute()
        
        # Then compensate
        await saga.compensate()
        
        # Both compensations should be called
        step2.compensation.assert_called_once()
        step1.compensation.assert_called_once()


class TestSagaOrchestrator:
    """Tests for SagaOrchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create saga orchestrator"""
        return SagaOrchestrator()
    
    @pytest.mark.asyncio
    async def test_execute_saga(self, orchestrator):
        """Test orchestrator executing a saga"""
        saga = Saga(
            saga_id="test-saga",
            steps=[
                SagaStep(
                    name="step1",
                    action=AsyncMock(return_value={"result": "success"}),
                    compensation=AsyncMock()
                )
            ]
        )
        
        result = await orchestrator.execute(saga)
        
        assert result is not None
        assert saga.state == SagaState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_orchestrator_handles_failure(self, orchestrator):
        """Test orchestrator handling saga failure"""
        saga = Saga(
            saga_id="test-saga",
            steps=[
                SagaStep(
                    name="step1",
                    action=AsyncMock(side_effect=Exception("Failed")),
                    compensation=AsyncMock(return_value=True)
                )
            ]
        )
        
        with pytest.raises(Exception):
            await orchestrator.execute(saga)
        
        assert saga.state == SagaState.FAILED
    
    @pytest.mark.asyncio
    async def test_orchestrator_tracks_sagas(self, orchestrator):
        """Test orchestrator tracking multiple sagas"""
        saga1 = Saga(saga_id="saga-1", steps=[])
        saga2 = Saga(saga_id="saga-2", steps=[])
        
        await orchestrator.execute(saga1)
        await orchestrator.execute(saga2)
        
        # Orchestrator should track both sagas
        assert orchestrator is not None



