"""
Workflow Testing Helpers
Specialized helpers for workflow and process testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import asyncio
from enum import Enum


class WorkflowState(Enum):
    """Workflow state enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowTestHelpers:
    """Helpers for workflow testing"""
    
    @staticmethod
    def create_mock_workflow(
        steps: Optional[List[Dict[str, Any]]] = None,
        current_state: WorkflowState = WorkflowState.PENDING
    ) -> Mock:
        """Create mock workflow"""
        workflow_steps = steps or []
        workflow = Mock()
        workflow.steps = workflow_steps
        workflow.state = current_state
        workflow.current_step = 0
        
        async def execute_side_effect():
            workflow.state = WorkflowState.RUNNING
            for i, step in enumerate(workflow_steps):
                workflow.current_step = i
                if asyncio.iscoroutinefunction(step.get("action")):
                    await step["action"]()
                elif step.get("action"):
                    step["action"]()
            workflow.state = WorkflowState.COMPLETED
            return {"success": True}
        
        workflow.execute = AsyncMock(side_effect=execute_side_effect)
        workflow.cancel = Mock(side_effect=lambda: setattr(workflow, "state", WorkflowState.CANCELLED))
        workflow.get_state = Mock(return_value=workflow.state)
        return workflow
    
    @staticmethod
    def assert_workflow_completed(workflow: Mock):
        """Assert workflow completed successfully"""
        assert workflow.state == WorkflowState.COMPLETED, \
            f"Workflow state is {workflow.state}, expected COMPLETED"
    
    @staticmethod
    def assert_workflow_failed(workflow: Mock):
        """Assert workflow failed"""
        assert workflow.state == WorkflowState.FAILED, \
            f"Workflow state is {workflow.state}, expected FAILED"


class StateMachineHelpers:
    """Helpers for state machine testing"""
    
    @staticmethod
    def create_mock_state_machine(
        states: List[str],
        transitions: Dict[str, List[str]],
        initial_state: str
    ) -> Mock:
        """Create mock state machine"""
        machine = Mock()
        machine.states = states
        machine.transitions = transitions
        machine.current_state = initial_state
        
        def transition_side_effect(new_state: str):
            if new_state in machine.transitions.get(machine.current_state, []):
                machine.current_state = new_state
                return True
            return False
        
        machine.transition = Mock(side_effect=transition_side_effect)
        machine.can_transition = Mock(
            side_effect=lambda state: state in machine.transitions.get(machine.current_state, [])
        )
        return machine
    
    @staticmethod
    def assert_state_transition(
        machine: Mock,
        from_state: str,
        to_state: str
    ):
        """Assert state transition occurred"""
        assert machine.current_state == to_state, \
            f"Current state is {machine.current_state}, expected {to_state}"
        assert machine.transition.called, "State transition was not called"


class ProcessHelpers:
    """Helpers for process testing"""
    
    @staticmethod
    async def test_process_flow(
        steps: List[Callable],
        expected_results: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Test process flow execution"""
        results = []
        errors = []
        
        for i, step in enumerate(steps):
            try:
                if asyncio.iscoroutinefunction(step):
                    result = await step()
                else:
                    result = step()
                results.append(result)
                
                if expected_results and i < len(expected_results):
                    expected = expected_results[i]
                    if result != expected:
                        errors.append({
                            "step": i,
                            "expected": expected,
                            "actual": result
                        })
            except Exception as e:
                errors.append({
                    "step": i,
                    "error": str(e)
                })
                break
        
        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors
        }
    
    @staticmethod
    def assert_process_success(process_result: Dict[str, Any]):
        """Assert process completed successfully"""
        assert process_result["success"], \
            f"Process failed with errors: {process_result.get('errors')}"


# Convenience exports
create_mock_workflow = WorkflowTestHelpers.create_mock_workflow
assert_workflow_completed = WorkflowTestHelpers.assert_workflow_completed
assert_workflow_failed = WorkflowTestHelpers.assert_workflow_failed

create_mock_state_machine = StateMachineHelpers.create_mock_state_machine
assert_state_transition = StateMachineHelpers.assert_state_transition

test_process_flow = ProcessHelpers.test_process_flow
assert_process_success = ProcessHelpers.assert_process_success

