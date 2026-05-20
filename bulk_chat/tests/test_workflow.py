"""
Tests for Workflow Engine
==========================
"""

import pytest
import asyncio
from ..core.workflow import WorkflowEngine, Workflow, WorkflowStep, WorkflowStatus


@pytest.fixture
def workflow_engine():
    """Create workflow engine for testing."""
    return WorkflowEngine()


@pytest.mark.asyncio
async def test_create_workflow(workflow_engine):
    """Test creating a workflow."""
    workflow_id = workflow_engine.create_workflow(
        workflow_id="test_workflow",
        name="Test Workflow",
        steps=[
            WorkflowStep(
                step_id="step1",
                action="create_session",
                parameters={"user_id": "test"}
            ),
            WorkflowStep(
                step_id="step2",
                action="send_message",
                parameters={"content": "Hello"}
            )
        ]
    )
    
    assert workflow_id == "test_workflow"
    assert workflow_id in workflow_engine.workflows


@pytest.mark.asyncio
async def test_execute_workflow(workflow_engine):
    """Test executing a workflow."""
    workflow_id = workflow_engine.create_workflow(
        "test_workflow",
        "Test",
        steps=[
            WorkflowStep("step1", "test_action", {})
        ]
    )
    
    # Mock step execution
    with patch.object(workflow_engine, '_execute_step', return_value=True):
        result = await workflow_engine.execute_workflow(workflow_id)
        
        assert result is not None
        workflow = workflow_engine.workflows[workflow_id]
        assert workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING]


@pytest.mark.asyncio
async def test_get_workflow_status(workflow_engine):
    """Test getting workflow status."""
    workflow_id = workflow_engine.create_workflow(
        "test_workflow",
        "Test",
        steps=[WorkflowStep("step1", "test", {})]
    )
    
    status = workflow_engine.get_workflow_status(workflow_id)
    
    assert status is not None
    assert status in WorkflowStatus


@pytest.mark.asyncio
async def test_pause_workflow(workflow_engine):
    """Test pausing a workflow."""
    workflow_id = workflow_engine.create_workflow(
        "test_workflow",
        "Test",
        steps=[WorkflowStep("step1", "test", {})]
    )
    
    await workflow_engine.pause_workflow(workflow_id)
    
    workflow = workflow_engine.workflows[workflow_id]
    assert workflow.status == WorkflowStatus.PAUSED


@pytest.mark.asyncio
async def test_get_workflow_history(workflow_engine):
    """Test getting workflow execution history."""
    workflow_id = workflow_engine.create_workflow(
        "test_workflow",
        "Test",
        steps=[WorkflowStep("step1", "test", {})]
    )
    
    history = workflow_engine.get_workflow_history(workflow_id)
    
    assert isinstance(history, list)


from unittest.mock import patch


