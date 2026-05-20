"""
Improved test examples demonstrating best practices
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..core.project_generator import ProjectGenerator
from .test_helpers import TestHelpers
from .test_utils_helpers import (
    AsyncTestHelpers,
    FileTestHelpers,
    MockHelpers,
    PerformanceTestHelpers,
    ValidationHelpers,
    DataGenerators
)


class TestImprovedExamples:
    """Examples of improved test patterns"""
    
    @pytest.mark.unit
    def test_using_test_helpers(self, temp_dir, sample_project_structure):
        """Example: Using TestHelpers for validation"""
        # Use TestHelpers to validate structure
        TestHelpers.assert_project_structure(
            sample_project_structure,
            required_files=["README.md", "backend/main.py", "frontend/package.json"]
        )
        
        # Validate JSON files
        package_json = sample_project_structure / "frontend" / "package.json"
        TestHelpers.assert_valid_json(package_json)
        
        # Validate Python files
        main_py = sample_project_structure / "backend" / "main.py"
        TestHelpers.assert_valid_python(main_py)
    
    @pytest.mark.unit
    def test_using_file_helpers(self, temp_dir):
        """Example: Using FileTestHelpers"""
        # Create temporary file
        temp_file = FileTestHelpers.create_temp_file("test content", suffix=".txt")
        assert temp_file.exists()
        
        # Assert file size
        FileTestHelpers.assert_file_size(temp_file, min_size=1)
        
        # Cleanup
        FileTestHelpers.cleanup_path(temp_file)
    
    @pytest.mark.unit
    def test_using_mock_helpers(self):
        """Example: Using MockHelpers"""
        # Create async mock
        async_mock = MockHelpers.create_async_mock(return_value={"success": True})
        
        # Create mock response
        mock_response = MockHelpers.create_mock_response(
            status_code=200,
            json_data={"data": "test"}
        )
        
        assert mock_response.status_code == 200
        assert mock_response.json()["data"] == "test"
    
    @pytest.mark.async
    @pytest.mark.unit
    async def test_using_async_helpers(self):
        """Example: Using AsyncTestHelpers"""
        condition_met = False
        
        async def set_condition():
            nonlocal condition_met
            await asyncio.sleep(0.1)
            condition_met = True
        
        # Set condition asynchronously
        asyncio.create_task(set_condition())
        
        # Wait for condition
        await AsyncTestHelpers.wait_for_condition(
            lambda: condition_met,
            timeout=1.0,
            error_message="Condition not met"
        )
        
        assert condition_met is True
    
    @pytest.mark.performance
    def test_using_performance_helpers(self):
        """Example: Using PerformanceTestHelpers"""
        def fast_operation():
            return sum(range(100))
        
        # Measure time
        with PerformanceTestHelpers.measure_time() as elapsed:
            result = fast_operation()
        
        # Assert performance
        PerformanceTestHelpers.assert_performance(
            elapsed,
            max_time=0.1,
            operation_name="Fast operation"
        )
        
        assert result == 4950
    
    @pytest.mark.unit
    def test_using_validation_helpers(self, sample_api_response):
        """Example: Using ValidationHelpers"""
        # Validate structure
        ValidationHelpers.validate_json_structure(
            sample_api_response,
            required_keys=["project_id", "status", "project_path"],
            key_types={"project_id": str, "status": str}
        )
    
    @pytest.mark.unit
    def test_using_data_generators(self):
        """Example: Using DataGenerators"""
        # Generate unique project name
        project_name = DataGenerators.generate_project_name("test")
        assert project_name.startswith("test_")
        assert len(project_name) > 5
        
        # Generate project description
        desc = DataGenerators.generate_project_description(
            ai_type="vision",
            features=["object_detection", "image_classification"]
        )
        assert "vision" in desc.lower()
        assert "object_detection" in desc
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_improved_project_generation(self, project_generator, temp_dir):
        """Example: Improved project generation test"""
        description = "A chat AI system with authentication"
        
        # Generate project
        project = await project_generator.generate_project(description)
        
        # Validate using helpers
        assert project is not None
        TestHelpers.assert_dict_contains(
            project,
            required_keys=["project_id", "project_path", "name"]
        )
        
        # Validate project structure
        project_path = Path(project["project_path"])
        TestHelpers.assert_project_structure(
            project_path,
            required_files=["README.md"]
        )
    
    @pytest.mark.unit
    def test_improved_error_handling(self):
        """Example: Improved error handling in tests"""
        # Test with retry
        attempts = 0
        
        async def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Temporary error")
            return "success"
        
        # Should succeed after retries
        result = asyncio.run(
            AsyncTestHelpers.retry_async(
                flaky_operation,
                max_attempts=3,
                exceptions=(ValueError,)
            )
        )
        
        assert result == "success"
        assert attempts == 3
    
    @pytest.mark.unit
    def test_using_fixtures_effectively(self, sample_project_structure, sample_descriptions):
        """Example: Using multiple fixtures effectively"""
        # Use sample project structure
        assert sample_project_structure.exists()
        assert (sample_project_structure / "backend").exists()
        
        # Use sample descriptions
        assert len(sample_descriptions) == 5
        assert all(isinstance(desc, str) for desc in sample_descriptions)

