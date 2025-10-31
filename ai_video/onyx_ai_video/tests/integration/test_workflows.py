from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from ...workflows.video_workflow import (
from ...workflows.workflow_manager import (
from ...core.models import VideoRequest, VideoResponse, VideoQuality, VideoFormat, WorkflowStep
from typing import Any, List, Dict, Optional
import logging
"""
Integration tests for workflow modules.
"""


    OnyxVideoWorkflow, VideoWorkflowConfig, VideoWorkflowStep,
    create_video_workflow, execute_workflow_step, validate_workflow,
    get_workflow_status, create_workflow_report
)
    WorkflowManager, WorkflowDefinition, WorkflowExecution,
    register_workflow, execute_workflow, get_workflow_info,
    list_workflows, delete_workflow, update_workflow
)


class TestOnyxVideoWorkflow:
    """Test OnyxVideoWorkflow class."""
    
    @pytest.mark.integration
    async def test_workflow_initialization(self, temp_dir) -> Any:
        """Test workflow initialization."""
        config = VideoWorkflowConfig(
            max_steps=10,
            timeout=300,
            retry_attempts=3,
            enable_parallel_processing=True,
            cache_enabled=True,
            cache_size=100
        )
        
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        assert workflow.initialized is True
        assert workflow.config == config
        assert workflow.steps is not None
        assert workflow.execution_history is not None
        assert workflow.cache is not None
    
    @pytest.mark.integration
    async def test_basic_video_generation(self, temp_dir, sample_video_request) -> Any:
        """Test basic video generation workflow."""
        config = VideoWorkflowConfig()
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        # Mock LLM and video generation
        with patch.object(workflow, '_generate_script', new_callable=AsyncMock, return_value="Generated script"):
            with patch.object(workflow, '_generate_video', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = {
                    "output_path": str(temp_dir / "output.mp4"),
                    "duration": 10.5,
                    "file_size": 1024000,
                    "resolution": "1920x1080",
                    "fps": 30.0
                }
                
                response = await workflow.generate_video(sample_video_request)
                
                assert response.request_id == sample_video_request.request_id
                assert response.status == "completed"
                assert response.output_path == str(temp_dir / "output.mp4")
                assert response.duration == 10.5
                assert response.file_size == 1024000
                assert response.resolution == "1920x1080"
                assert response.fps == 30.0
    
    @pytest.mark.integration
    async def test_video_generation_with_vision(self, temp_dir, sample_video_request) -> Any:
        """Test video generation with vision capabilities."""
        config = VideoWorkflowConfig()
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        image_data = b"fake_image_data"
        
        # Mock vision processing and video generation
        with patch.object(workflow, '_process_image', new_callable=AsyncMock, return_value="Image description"):
            with patch.object(workflow, '_generate_script_with_vision', new_callable=AsyncMock, return_value="Vision script"):
                with patch.object(workflow, '_generate_video', new_callable=AsyncMock) as mock_generate:
                    mock_generate.return_value = {
                        "output_path": str(temp_dir / "vision_output.mp4"),
                        "duration": 12.0,
                        "file_size": 2048000,
                        "resolution": "1920x1080",
                        "fps": 30.0
                    }
                    
                    response = await workflow.generate_video_with_vision(sample_video_request, image_data)
                    
                    assert response.request_id == sample_video_request.request_id
                    assert response.status == "completed"
                    assert response.output_path == str(temp_dir / "vision_output.mp4")
                    assert response.duration == 12.0
                    assert response.metadata["vision_used"] is True
    
    @pytest.mark.integration
    async def test_workflow_steps_execution(self, temp_dir, sample_video_request) -> Any:
        """Test individual workflow steps execution."""
        config = VideoWorkflowConfig()
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        # Test text processing step
        with patch.object(workflow, '_process_text', new_callable=AsyncMock, return_value="Processed text"):
            result = await workflow._execute_step("text_processing", sample_video_request)
            assert result["status"] == "completed"
            assert result["output"] == "Processed text"
        
        # Test script generation step
        with patch.object(workflow, '_generate_script', new_callable=AsyncMock, return_value="Generated script"):
            result = await workflow._execute_step("script_generation", sample_video_request)
            assert result["status"] == "completed"
            assert result["output"] == "Generated script"
        
        # Test video generation step
        with patch.object(workflow, '_generate_video', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {
                "output_path": str(temp_dir / "test.mp4"),
                "duration": 10.0
            }
            result = await workflow._execute_step("video_generation", sample_video_request)
            assert result["status"] == "completed"
            assert "output_path" in result["output"]
    
    @pytest.mark.integration
    async def test_workflow_error_handling(self, sample_video_request) -> Any:
        """Test workflow error handling."""
        config = VideoWorkflowConfig()
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        # Mock step to fail
        with patch.object(workflow, '_generate_script', new_callable=AsyncMock, side_effect=Exception("Script generation failed")):
            with pytest.raises(Exception, match="Script generation failed"):
                await workflow.generate_video(sample_video_request)
    
    @pytest.mark.integration
    async def test_workflow_retry_mechanism(self, sample_video_request) -> Any:
        """Test workflow retry mechanism."""
        config = VideoWorkflowConfig(retry_attempts=3)
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        # Mock step that fails twice then succeeds
        call_count = 0
        async def failing_then_succeeding():
            
    """failing_then_succeeding function."""
nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "Success after retries"
        
        with patch.object(workflow, '_generate_script', new_callable=AsyncMock, side_effect=failing_then_succeeding):
            with patch.object(workflow, '_generate_video', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = {
                    "output_path": "/tmp/test.mp4",
                    "duration": 10.0
                }
                
                response = await workflow.generate_video(sample_video_request)
                
                assert response.status == "completed"
                assert call_count == 3  # Should have retried twice
    
    @pytest.mark.integration
    async def test_workflow_timeout_handling(self, sample_video_request) -> Any:
        """Test workflow timeout handling."""
        config = VideoWorkflowConfig(timeout=1)  # 1 second timeout
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        # Mock step that takes too long
        async def slow_operation():
            
    """slow_operation function."""
await asyncio.sleep(2)  # Takes 2 seconds
            return "Too late"
        
        with patch.object(workflow, '_generate_script', new_callable=AsyncMock, side_effect=slow_operation):
            with pytest.raises(asyncio.TimeoutError):
                await workflow.generate_video(sample_video_request)
    
    @pytest.mark.integration
    async def test_workflow_caching(self, sample_video_request) -> Any:
        """Test workflow caching functionality."""
        config = VideoWorkflowConfig(cache_enabled=True, cache_size=10)
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        # Mock script generation
        with patch.object(workflow, '_generate_script', new_callable=AsyncMock, return_value="Cached script"):
            with patch.object(workflow, '_generate_video', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = {
                    "output_path": "/tmp/test.mp4",
                    "duration": 10.0
                }
                
                # First request
                response1 = await workflow.generate_video(sample_video_request)
                assert response1.status == "completed"
                
                # Second request with same input (should use cache)
                response2 = await workflow.generate_video(sample_video_request)
                assert response2.status == "completed"
                
                # Verify cache was used
                assert len(workflow.cache) > 0
    
    @pytest.mark.integration
    async def test_workflow_parallel_processing(self, temp_dir) -> Any:
        """Test parallel processing in workflow."""
        config = VideoWorkflowConfig(enable_parallel_processing=True)
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        # Create multiple requests
        requests = []
        for i in range(3):
            request = VideoRequest(
                input_text=f"Test video {i}",
                user_id=f"user{i}",
                quality=VideoQuality.LOW,
                duration=10,
                output_format=VideoFormat.MP4
            )
            requests.append(request)
        
        # Mock video generation
        with patch.object(workflow, '_generate_script', new_callable=AsyncMock, return_value="Script"):
            with patch.object(workflow, '_generate_video', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = {
                    "output_path": str(temp_dir / "test.mp4"),
                    "duration": 10.0
                }
                
                # Execute requests in parallel
                tasks = [workflow.generate_video(req) for req in requests]
                responses = await asyncio.gather(*tasks)
                
                assert len(responses) == 3
                for response in responses:
                    assert response.status == "completed"
    
    @pytest.mark.integration
    async def test_workflow_status_tracking(self, sample_video_request) -> Any:
        """Test workflow status tracking."""
        config = VideoWorkflowConfig()
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        # Mock video generation
        with patch.object(workflow, '_generate_script', new_callable=AsyncMock, return_value="Script"):
            with patch.object(workflow, '_generate_video', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = {
                    "output_path": "/tmp/test.mp4",
                    "duration": 10.0
                }
                
                await workflow.generate_video(sample_video_request)
                
                # Check execution history
                assert len(workflow.execution_history) > 0
                assert workflow.execution_history[0]["request_id"] == sample_video_request.request_id
                assert workflow.execution_history[0]["status"] == "completed"
    
    @pytest.mark.integration
    async def test_workflow_shutdown(self) -> Any:
        """Test workflow shutdown."""
        config = VideoWorkflowConfig()
        workflow = OnyxVideoWorkflow(config)
        await workflow.initialize()
        
        await workflow.shutdown()
        
        assert workflow.initialized is False
        assert workflow.shutdown_requested is True


class TestWorkflowManager:
    """Test WorkflowManager class."""
    
    @pytest.mark.integration
    async def test_workflow_manager_initialization(self) -> Any:
        """Test workflow manager initialization."""
        manager = WorkflowManager()
        await manager.initialize()
        
        assert manager.initialized is True
        assert manager.workflows is not None
        assert manager.executions is not None
    
    @pytest.mark.integration
    async def test_register_workflow(self) -> Any:
        """Test workflow registration."""
        manager = WorkflowManager()
        await manager.initialize()
        
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            steps=[
                WorkflowStep(name="step1", description="First step"),
                WorkflowStep(name="step2", description="Second step")
            ],
            timeout=300,
            retry_attempts=3
        )
        
        await manager.register_workflow(workflow_def)
        
        assert "test_workflow" in manager.workflows
        assert manager.workflows["test_workflow"] == workflow_def
    
    @pytest.mark.integration
    async def test_execute_workflow(self, sample_video_request) -> Any:
        """Test workflow execution."""
        manager = WorkflowManager()
        await manager.initialize()
        
        # Register a test workflow
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            steps=[
                WorkflowStep(name="step1", description="First step"),
                WorkflowStep(name="step2", description="Second step")
            ],
            timeout=300,
            retry_attempts=3
        )
        
        await manager.register_workflow(workflow_def)
        
        # Mock step execution
        async def mock_step_execution(step_name, context) -> Any:
            return {"status": "completed", "output": f"Step {step_name} completed"}
        
        with patch.object(manager, '_execute_workflow_step', new_callable=AsyncMock, side_effect=mock_step_execution):
            execution = await manager.execute_workflow("test_workflow", sample_video_request)
            
            assert execution.workflow_name == "test_workflow"
            assert execution.status == "completed"
            assert len(execution.step_results) == 2
    
    @pytest.mark.integration
    async def test_get_workflow_info(self) -> Optional[Dict[str, Any]]:
        """Test getting workflow information."""
        manager = WorkflowManager()
        await manager.initialize()
        
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            steps=[
                WorkflowStep(name="step1", description="First step"),
                WorkflowStep(name="step2", description="Second step")
            ],
            timeout=300,
            retry_attempts=3
        )
        
        await manager.register_workflow(workflow_def)
        
        info = await manager.get_workflow_info("test_workflow")
        
        assert info["name"] == "test_workflow"
        assert info["description"] == "Test workflow"
        assert len(info["steps"]) == 2
        assert info["timeout"] == 300
        assert info["retry_attempts"] == 3
    
    @pytest.mark.integration
    async def test_list_workflows(self) -> List[Any]:
        """Test listing workflows."""
        manager = WorkflowManager()
        await manager.initialize()
        
        # Register multiple workflows
        workflows = [
            WorkflowDefinition(name="workflow1", description="First workflow"),
            WorkflowDefinition(name="workflow2", description="Second workflow"),
            WorkflowDefinition(name="workflow3", description="Third workflow")
        ]
        
        for workflow in workflows:
            await manager.register_workflow(workflow)
        
        workflow_list = await manager.list_workflows()
        
        assert len(workflow_list) == 3
        assert any(w["name"] == "workflow1" for w in workflow_list)
        assert any(w["name"] == "workflow2" for w in workflow_list)
        assert any(w["name"] == "workflow3" for w in workflow_list)
    
    @pytest.mark.integration
    async def test_delete_workflow(self) -> Any:
        """Test workflow deletion."""
        manager = WorkflowManager()
        await manager.initialize()
        
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow"
        )
        
        await manager.register_workflow(workflow_def)
        assert "test_workflow" in manager.workflows
        
        await manager.delete_workflow("test_workflow")
        assert "test_workflow" not in manager.workflows
    
    @pytest.mark.integration
    async def test_update_workflow(self) -> Any:
        """Test workflow update."""
        manager = WorkflowManager()
        await manager.initialize()
        
        # Register initial workflow
        initial_workflow = WorkflowDefinition(
            name="test_workflow",
            description="Initial description",
            steps=[WorkflowStep(name="step1", description="First step")]
        )
        
        await manager.register_workflow(initial_workflow)
        
        # Update workflow
        updated_workflow = WorkflowDefinition(
            name="test_workflow",
            description="Updated description",
            steps=[
                WorkflowStep(name="step1", description="Updated first step"),
                WorkflowStep(name="step2", description="New second step")
            ]
        )
        
        await manager.update_workflow("test_workflow", updated_workflow)
        
        stored_workflow = manager.workflows["test_workflow"]
        assert stored_workflow.description == "Updated description"
        assert len(stored_workflow.steps) == 2
    
    @pytest.mark.integration
    async def test_workflow_execution_tracking(self, sample_video_request) -> Any:
        """Test workflow execution tracking."""
        manager = WorkflowManager()
        await manager.initialize()
        
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            steps=[WorkflowStep(name="step1", description="First step")]
        )
        
        await manager.register_workflow(workflow_def)
        
        # Mock step execution
        async def mock_step_execution(step_name, context) -> Any:
            return {"status": "completed", "output": "Step completed"}
        
        with patch.object(manager, '_execute_workflow_step', new_callable=AsyncMock, side_effect=mock_step_execution):
            execution = await manager.execute_workflow("test_workflow", sample_video_request)
            
            # Check execution tracking
            assert len(manager.executions) > 0
            assert manager.executions[0].workflow_name == "test_workflow"
            assert manager.executions[0].status == "completed"
    
    @pytest.mark.integration
    async def test_workflow_error_handling(self, sample_video_request) -> Any:
        """Test workflow error handling."""
        manager = WorkflowManager()
        await manager.initialize()
        
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            steps=[WorkflowStep(name="step1", description="First step")]
        )
        
        await manager.register_workflow(workflow_def)
        
        # Mock step to fail
        async def failing_step(step_name, context) -> Any:
            raise Exception("Step failed")
        
        with patch.object(manager, '_execute_workflow_step', new_callable=AsyncMock, side_effect=failing_step):
            with pytest.raises(Exception, match="Step failed"):
                await manager.execute_workflow("test_workflow", sample_video_request)
    
    @pytest.mark.integration
    async def test_workflow_timeout_handling(self, sample_video_request) -> Any:
        """Test workflow timeout handling."""
        manager = WorkflowManager()
        await manager.initialize()
        
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            steps=[WorkflowStep(name="step1", description="First step")],
            timeout=1  # 1 second timeout
        )
        
        await manager.register_workflow(workflow_def)
        
        # Mock step that takes too long
        async def slow_step(step_name, context) -> Any:
            await asyncio.sleep(2)  # Takes 2 seconds
            return {"status": "completed"}
        
        with patch.object(manager, '_execute_workflow_step', new_callable=AsyncMock, side_effect=slow_step):
            with pytest.raises(asyncio.TimeoutError):
                await manager.execute_workflow("test_workflow", sample_video_request)


class TestWorkflowUtilities:
    """Test workflow utility functions."""
    
    @pytest.mark.integration
    async def test_create_video_workflow(self) -> Any:
        """Test create_video_workflow function."""
        config = VideoWorkflowConfig()
        workflow = await create_video_workflow(config)
        
        assert workflow is not None
        assert workflow.initialized is True
        assert workflow.config == config
    
    @pytest.mark.integration
    async def test_execute_workflow_step(self) -> Any:
        """Test execute_workflow_step function."""
        step = WorkflowStep(name="test_step", description="Test step")
        context = {"input": "test input"}
        
        # Mock step execution
        async def mock_step_function(step_name, context) -> Any:
            return {"status": "completed", "output": "Step completed"}
        
        with patch('onyx_ai_video.workflows.video_workflow._execute_step_function', 
                  new_callable=AsyncMock, side_effect=mock_step_function):
            result = await execute_workflow_step(step, context)
            
            assert result["status"] == "completed"
            assert result["output"] == "Step completed"
    
    @pytest.mark.integration
    async def test_validate_workflow(self) -> bool:
        """Test validate_workflow function."""
        # Valid workflow
        valid_workflow = WorkflowDefinition(
            name="valid_workflow",
            description="Valid workflow",
            steps=[WorkflowStep(name="step1", description="First step")]
        )
        
        is_valid, errors = validate_workflow(valid_workflow)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid workflow (no steps)
        invalid_workflow = WorkflowDefinition(
            name="invalid_workflow",
            description="Invalid workflow",
            steps=[]
        )
        
        is_valid, errors = validate_workflow(invalid_workflow)
        assert is_valid is False
        assert len(errors) > 0
    
    @pytest.mark.integration
    async def test_get_workflow_status(self) -> Optional[Dict[str, Any]]:
        """Test get_workflow_status function."""
        manager = WorkflowManager()
        await manager.initialize()
        
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            steps=[WorkflowStep(name="step1", description="First step")]
        )
        
        await manager.register_workflow(workflow_def)
        
        status = await get_workflow_status(manager, "test_workflow")
        
        assert status["name"] == "test_workflow"
        assert status["registered"] is True
        assert "execution_count" in status
    
    @pytest.mark.integration
    async def test_create_workflow_report(self) -> Any:
        """Test create_workflow_report function."""
        manager = WorkflowManager()
        await manager.initialize()
        
        # Register some workflows and executions
        workflows = [
            WorkflowDefinition(name="workflow1", description="First workflow"),
            WorkflowDefinition(name="workflow2", description="Second workflow")
        ]
        
        for workflow in workflows:
            await manager.register_workflow(workflow)
        
        report = await create_workflow_report(manager)
        
        assert "total_workflows" in report
        assert "workflow_list" in report
        assert "execution_summary" in report
        assert report["total_workflows"] == 2 