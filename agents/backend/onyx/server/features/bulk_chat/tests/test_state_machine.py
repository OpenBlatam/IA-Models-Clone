"""
Tests for State Machine Manager
=================================
"""

import pytest
import asyncio
from ..core.state_machine import StateMachineManager, StateMachine, State


@pytest.fixture
def state_machine_manager():
    """Create state machine manager for testing."""
    return StateMachineManager()


@pytest.mark.asyncio
async def test_create_state_machine(state_machine_manager):
    """Test creating a state machine."""
    machine_id = state_machine_manager.create_state_machine(
        machine_id="test_machine",
        initial_state="initial",
        states=["initial", "processing", "completed"],
        transitions=[
            {"from": "initial", "to": "processing", "trigger": "start"},
            {"from": "processing", "to": "completed", "trigger": "finish"}
        ]
    )
    
    assert machine_id == "test_machine"
    assert machine_id in state_machine_manager.machines


@pytest.mark.asyncio
async def test_transition_state(state_machine_manager):
    """Test transitioning state."""
    state_machine_manager.create_state_machine(
        "test_machine",
        "initial",
        ["initial", "processing"],
        [{"from": "initial", "to": "processing", "trigger": "start"}]
    )
    
    result = await state_machine_manager.transition(
        "test_machine",
        "start"
    )
    
    assert result is True
    machine = state_machine_manager.machines["test_machine"]
    assert machine.current_state == "processing"


@pytest.mark.asyncio
async def test_get_state(state_machine_manager):
    """Test getting current state."""
    state_machine_manager.create_state_machine(
        "test_machine",
        "initial",
        ["initial", "processing"],
        []
    )
    
    state = state_machine_manager.get_state("test_machine")
    
    assert state == "initial"


@pytest.mark.asyncio
async def test_invalid_transition(state_machine_manager):
    """Test invalid transition."""
    state_machine_manager.create_state_machine(
        "test_machine",
        "initial",
        ["initial", "processing"],
        [{"from": "initial", "to": "processing", "trigger": "start"}]
    )
    
    # Try invalid transition
    result = await state_machine_manager.transition("test_machine", "invalid_trigger")
    
    assert result is False


@pytest.mark.asyncio
async def test_get_state_machine_summary(state_machine_manager):
    """Test getting state machine summary."""
    state_machine_manager.create_state_machine(
        "machine1",
        "initial",
        ["initial", "done"],
        []
    )
    
    summary = state_machine_manager.get_state_machine_summary()
    
    assert summary is not None
    assert "total_machines" in summary or "machines" in summary


