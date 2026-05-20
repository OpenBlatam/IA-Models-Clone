"""
Tests for Workflow Engine V2
==============================
"""

import pytest
import asyncio
from ..core.workflow_engine_v2 import WorkflowEngineV2, WorkflowStatus


@pytest.fixture
def workflow_engine_v2():
    """Create workflow engine V2 for testing."""
    return WorkflowEngineV2()


@pytest.mark.asyncio
async def test_create_workflow_v2(workflow_engine_v2):
    """Test creating a workflow V2."""
    workflow_id = workflow_engine_v2.create_workflow(
        workflow_id="test_workflow",
        name="Test Workflow",
        steps=[
            {"step_id": "step1", "action": "create", "parameters": {}}
        ],
        supports_parallel=True
    )
    
    assert workflow_id == "test_workflow"
    assert workflow_id in workflow_engine_v2.workflows


@pytest.mark.asyncio
async def test_execute_workflow_parallel(workflow_engine_v2):
    """Test executing workflow with parallel steps."""
    workflow_id = workflow_engine_v2.create_workflow(
        "test_workflow",
        "Test",
        steps=[
            {"step_id": "step1", "action": "action1", "parallel": True},
            {"step_id": "step2", "action": "action2", "parallel": True}
        ],
        supports_parallel=True
    )
    
    # Mock step execution
    with patch.object(workflow_engine_v2, '_execute_step', return_value=True):
        result = await workflow_engine_v2.execute_workflow(workflow_id)
        
        assert result is not None


@pytest.mark.asyncio
async def test_execute_workflow_with_compensation(workflow_engine_v2):
    """Test executing workflow with compensation."""
    workflow_id = workflow_engine_v2.create_workflow(
        "test_workflow",
        "Test",
        steps=[
            {
                "step_id": "step1",
                "action": "create",
                "compensation": "rollback",
                "parameters": {}
            }
        ]
    )
    
    # Mock execution
    with patch.object(workflow_engine_v2, '_execute_step', return_value=True):
        result = await workflow_engine_v2.execute_workflow(workflow_id)
        
        assert result is not None


@pytest.mark.asyncio
async def test_get_workflow_status_v2(workflow_engine_v2):
    """Test getting workflow status V2."""
    workflow_id = workflow_engine_v2.create_workflow(
        "test_workflow",
        "Test",
        steps=[{"step_id": "step1", "action": "test"}]
    )
    
    status = workflow_engine_v2.get_workflow_status(workflow_id)
    
    assert status is not None
    assert status in WorkflowStatus


@pytest.mark.asyncio
async def test_get_workflow_engine_v2_summary(workflow_engine_v2):
    """Test getting workflow engine V2 summary."""
    workflow_engine_v2.create_workflow(
        "workflow1",
        "Test 1",
        steps=[{"step_id": "step1", "action": "test"}]
    )
    
    summary = workflow_engine_v2.get_workflow_engine_v2_summary()
    
    assert summary is not None
    assert "total_workflows" in summary or "workflows_by_status" in summary


from unittest.mock import patch


