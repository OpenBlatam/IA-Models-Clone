"""
Tests for ContinuousGenerator
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ..core.continuous_generator import ContinuousGenerator


class TestContinuousGenerator:
    """Test suite for ContinuousGenerator"""

    def test_init(self, temp_dir):
        """Test ContinuousGenerator initialization"""
        generator = ContinuousGenerator(base_output_dir=str(temp_dir / "projects"))
        assert generator.is_running is False
        assert generator.queue == []
        assert generator.processed_projects == []
        assert generator.project_generator is not None

    def test_init_with_queue_file(self, temp_dir):
        """Test ContinuousGenerator with custom queue file"""
        queue_file = str(temp_dir / "custom_queue.json")
        generator = ContinuousGenerator(
            base_output_dir=str(temp_dir / "projects"),
            queue_file=queue_file
        )
        assert generator.queue_file == queue_file

    def test_load_queue_empty(self, temp_dir):
        """Test loading empty queue"""
        generator = ContinuousGenerator(base_output_dir=str(temp_dir / "projects"))
        assert generator.queue == []
        assert generator.processed_projects == []

    def test_load_queue_existing(self, temp_dir):
        """Test loading existing queue file"""
        queue_file = temp_dir / "test_queue.json"
        queue_data = {
            "queue": [
                {"id": "test-1", "description": "Test 1", "status": "pending"},
                {"id": "test-2", "description": "Test 2", "status": "pending"}
            ],
            "processed": [{"id": "test-0", "description": "Test 0", "status": "completed"}]
        }
        queue_file.write_text(json.dumps(queue_data))
        
        generator = ContinuousGenerator(
            base_output_dir=str(temp_dir / "projects"),
            queue_file=str(queue_file)
        )
        assert len(generator.queue) == 2
        assert len(generator.processed_projects) == 1

    def test_save_queue(self, temp_dir):
        """Test saving queue to file"""
        queue_file = temp_dir / "test_save_queue.json"
        generator = ContinuousGenerator(
            base_output_dir=str(temp_dir / "projects"),
            queue_file=str(queue_file)
        )
        
        generator.queue = [{"id": "test-1", "description": "Test 1"}]
        generator.processed_projects = [{"id": "test-0", "description": "Test 0"}]
        generator._save_queue()
        
        assert queue_file.exists()
        data = json.loads(queue_file.read_text())
        assert len(data["queue"]) == 1
        assert len(data["processed"]) == 1

    def test_add_project(self, continuous_generator):
        """Test adding project to queue"""
        project_id = continuous_generator.add_project(
            description="Test project",
            project_name="test_project",
            author="Test Author",
            priority=5
        )
        
        assert project_id is not None
        assert len(continuous_generator.queue) == 1
        assert continuous_generator.queue[0]["description"] == "Test project"
        assert continuous_generator.queue[0]["project_name"] == "test_project"
        assert continuous_generator.queue[0]["author"] == "Test Author"
        assert continuous_generator.queue[0]["priority"] == 5
        assert continuous_generator.queue[0]["status"] == "pending"

    def test_add_project_priority_sorting(self, continuous_generator):
        """Test that projects are sorted by priority"""
        continuous_generator.add_project("Low priority", priority=1)
        continuous_generator.add_project("High priority", priority=10)
        continuous_generator.add_project("Medium priority", priority=5)
        
        assert len(continuous_generator.queue) == 3
        # Should be sorted by priority (descending)
        priorities = [p["priority"] for p in continuous_generator.queue]
        assert priorities == sorted(priorities, reverse=True)

    def test_get_queue(self, continuous_generator):
        """Test getting queue status"""
        continuous_generator.add_project("Project 1")
        continuous_generator.add_project("Project 2")
        
        queue_info = continuous_generator.get_queue()
        assert queue_info["total"] == 2
        assert queue_info["pending"] == 2
        assert len(queue_info["projects"]) == 2

    def test_get_project_status(self, continuous_generator):
        """Test getting project status"""
        project_id = continuous_generator.add_project("Test project")
        
        status = continuous_generator.get_project_status(project_id)
        assert status["id"] == project_id
        assert status["status"] == "pending"
        assert status["description"] == "Test project"

    def test_get_project_status_not_found(self, continuous_generator):
        """Test getting status of non-existent project"""
        status = continuous_generator.get_project_status("non-existent")
        assert status is None

    def test_remove_project(self, continuous_generator):
        """Test removing project from queue"""
        project_id = continuous_generator.add_project("Test project")
        assert len(continuous_generator.queue) == 1
        
        result = continuous_generator.remove_project(project_id)
        assert result is True
        assert len(continuous_generator.queue) == 0

    def test_remove_project_not_found(self, continuous_generator):
        """Test removing non-existent project"""
        result = continuous_generator.remove_project("non-existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_start_stop(self, continuous_generator):
        """Test starting and stopping the generator"""
        assert continuous_generator.is_running is False
        
        # Start generator
        continuous_generator.start()
        assert continuous_generator.is_running is True
        assert continuous_generator.current_task is not None
        
        # Stop generator
        await continuous_generator.stop()
        assert continuous_generator.is_running is False

    @pytest.mark.asyncio
    async def test_process_queue_empty(self, continuous_generator):
        """Test processing empty queue"""
        continuous_generator.start()
        await asyncio.sleep(0.1)  # Give it time to process
        await continuous_generator.stop()
        
        # Should not crash with empty queue
        assert continuous_generator.is_running is False

    @pytest.mark.asyncio
    async def test_process_queue_with_project(self, continuous_generator, temp_dir):
        """Test processing queue with a project"""
        project_id = continuous_generator.add_project("Test project")
        
        with patch.object(continuous_generator.project_generator, 'generate_project', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {
                "project_id": project_id,
                "name": "test_project",
                "status": "completed"
            }
            
            continuous_generator.start()
            await asyncio.sleep(0.5)  # Give it time to process
            await continuous_generator.stop()
            
            # Check that project was processed
            status = continuous_generator.get_project_status(project_id)
            # Project should be removed from queue after processing
            assert status is None or status["status"] != "pending"
            mock_generate.assert_called()

    @pytest.mark.asyncio
    async def test_process_queue_error_handling(self, continuous_generator):
        """Test error handling during queue processing"""
        project_id = continuous_generator.add_project("Test project that will fail")
        
        with patch.object(continuous_generator.project_generator, 'generate_project',
                         new_callable=AsyncMock, side_effect=Exception("Generation error")):
            
            continuous_generator.start()
            await asyncio.sleep(0.5)
            await continuous_generator.stop()
            
            # Should handle error gracefully
            assert continuous_generator.is_running is False

    def test_get_stats(self, continuous_generator):
        """Test getting generator statistics"""
        continuous_generator.add_project("Project 1")
        continuous_generator.add_project("Project 2")
        continuous_generator.processed_projects = [{"id": "proc-1"}]
        
        stats = continuous_generator.get_stats()
        assert stats["queue_size"] == 2
        assert stats["processed_count"] == 1
        assert stats["is_running"] is False

    def test_clear_queue(self, continuous_generator):
        """Test clearing the queue"""
        continuous_generator.add_project("Project 1")
        continuous_generator.add_project("Project 2")
        assert len(continuous_generator.queue) == 2
        
        continuous_generator.clear_queue()
        assert len(continuous_generator.queue) == 0

