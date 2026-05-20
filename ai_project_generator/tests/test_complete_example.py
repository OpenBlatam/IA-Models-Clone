"""
Complete test example demonstrating all best practices
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
from .test_assertions import CustomAssertions
from .debug_helpers import DebugHelpers


@pytest.mark.unit
class TestCompleteExample:
    """Complete example demonstrating all testing best practices"""
    
    def test_complete_workflow_with_all_helpers(
        self,
        project_generator,
        temp_dir,
        sample_description,
        debug
    ):
        """
        Complete test example using all helpers and best practices.
        
        This test demonstrates:
        - Using fixtures
        - Using multiple helper classes
        - Custom assertions
        - Performance testing
        - Quality checks
        - Debug helpers
        """
        # Arrange
        description = sample_description
        debug.print_test_info("Complete Workflow Test", description=description)
        
        # Act - Generate project with performance measurement
        with PerformanceTestHelpers.measure_time() as elapsed:
            project = project_generator.generate_project(description)
        
        # Assert - Using custom assertions
        CustomAssertions.assert_project_exists(
            Path(project["project_path"]),
            project["name"]
        )
        
        # Assert - Using TestHelpers
        TestHelpers.assert_dict_contains(
            project,
            required_keys=["project_id", "project_path", "name", "status"]
        )
        
        # Assert - Validate structure
        project_path = Path(project["project_path"])
        CustomAssertions.assert_backend_structure(project_path)
        CustomAssertions.assert_frontend_structure(project_path)
        
        # Assert - Performance
        PerformanceTestHelpers.assert_performance(
            elapsed,
            max_time=30.0,
            operation_name="Project generation"
        )
        
        # Debug - Print structure if needed
        if debug:
            DebugHelpers.print_project_structure(project_path, max_depth=2)
    
    @pytest.mark.async
    @pytest.mark.unit
    async def test_async_workflow_with_retry(
        self,
        project_generator,
        temp_dir
    ):
        """
        Example of async test with retry logic.
        
        Demonstrates:
        - Async testing
        - Retry logic
        - Error handling
        """
        description = "A chat AI system"
        
        # Define flaky operation
        async def generate_with_retry():
            return await project_generator.generate_project(description)
        
        # Retry on failure
        project = await AsyncTestHelpers.retry_async(
            generate_with_retry,
            max_attempts=3,
            delay=0.5,
            exceptions=(Exception,)
        )
        
        # Assert
        assert project is not None
        TestHelpers.assert_dict_contains(
            project,
            required_keys=["project_id", "project_path"]
        )
    
    @pytest.mark.unit
    def test_with_mocks_and_validation(
        self,
        project_generator,
        temp_dir
    ):
        """
        Example using mocks and validation helpers.
        
        Demonstrates:
        - Mock creation
        - Response validation
        - Structure validation
        """
        # Create mock
        mock_response = MockHelpers.create_mock_response(
            status_code=200,
            json_data={"success": True, "data": "test"}
        )
        
        # Validate mock response
        ValidationHelpers.validate_api_response(
            mock_response,
            expected_status=200,
            required_fields=["success", "data"]
        )
        
        # Create mock project data
        mock_data = MockHelpers.create_mock_project_data(
            project_id="mock-123",
            name="mock_project"
        )
        
        # Validate structure
        ValidationHelpers.validate_json_structure(
            mock_data,
            required_keys=["project_id", "name", "status"],
            key_types={"project_id": str, "name": str}
        )
    
    @pytest.mark.unit
    def test_file_operations_with_helpers(
        self,
        temp_dir
    ):
        """
        Example of file operations using helpers.
        
        Demonstrates:
        - File creation
        - File validation
        - File size checks
        """
        # Create temporary file
        temp_file = FileTestHelpers.create_temp_file(
            "Test content for file operations",
            suffix=".txt"
        )
        
        # Assert file exists and has content
        assert temp_file.exists()
        FileTestHelpers.assert_file_size(temp_file, min_size=1)
        
        # Validate content
        TestHelpers.assert_file_contains(temp_file, "Test content")
        
        # Cleanup
        FileTestHelpers.cleanup_path(temp_file)
        assert not temp_file.exists()
    
    @pytest.mark.unit
    def test_data_generation(
        self
    ):
        """
        Example of using data generators.
        
        Demonstrates:
        - Generating test data
        - Unique names
        - Project descriptions
        """
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
        assert "object_detection" in desc.lower()
        
        # Generate large string
        large_string = DataGenerators.generate_large_string(size=1000)
        assert len(large_string) == 1000
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_integration_with_all_features(
        self,
        project_generator,
        complex_project_structure,
        quality_checker
    ):
        """
        Complete integration test with all features.
        
        Demonstrates:
        - Integration testing
        - Quality checks
        - Complex scenarios
        """
        # Use complex project structure
        assert complex_project_structure.exists()
        
        # Run quality checks
        quality_results = quality_checker.run_all_checks(complex_project_structure)
        
        # Assert quality
        assert quality_results["overall_valid"] is True
        
        # Validate structure
        CustomAssertions.assert_backend_structure(complex_project_structure)
        CustomAssertions.assert_frontend_structure(complex_project_structure)
    
    @pytest.mark.performance
    def test_performance_benchmarking(
        self,
        project_generator
    ):
        """
        Example of performance benchmarking.
        
        Demonstrates:
        - Performance measurement
        - Benchmarking
        - Performance assertions
        """
        def generate_operation():
            return project_generator._sanitize_name("Test Project Name")
        
        # Benchmark function
        stats = PerformanceTestHelpers.benchmark_function(
            generate_operation,
            iterations=100
        )
        
        # Assert performance
        assert stats["avg"] < 0.001  # Should be very fast
        assert stats["max"] < 0.01   # Even max should be fast
        
        print(f"\nPerformance Stats: {stats}")

