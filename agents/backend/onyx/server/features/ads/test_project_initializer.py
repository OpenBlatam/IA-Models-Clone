from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import asyncio
import tempfile
import os
import shutil
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from onyx.server.features.ads.project_initializer import (
from typing import Any, List, Dict, Optional
import logging
"""
Test suite for Project Initializer

Tests the comprehensive project initialization system including:
- Problem definition validation
- Dataset analysis functionality
- Project structure creation
- Configuration generation
- Documentation creation
- Integration with existing systems
"""

    ProjectInitializer, DatasetAnalyzer, create_project, analyze_dataset,
    create_problem_definition, create_dataset_info,
    ProjectType, DatasetType, ProblemComplexity,
    ProblemDefinition, DatasetInfo
)

class TestProblemDefinition:
    """Test problem definition creation and validation."""
    
    def test_create_problem_definition_basic(self) -> Any:
        """Test basic problem definition creation."""
        problem_def = create_problem_definition(
            title="Test Project",
            description="Test description",
            project_type=ProjectType.CLASSIFICATION,
            complexity=ProblemComplexity.MODERATE,
            business_objective="Test objective"
        )
        
        assert problem_def.title == "Test Project"
        assert problem_def.description == "Test description"
        assert problem_def.project_type == ProjectType.CLASSIFICATION
        assert problem_def.complexity == ProblemComplexity.MODERATE
        assert problem_def.business_objective == "Test objective"
        assert isinstance(problem_def.success_metrics, list)
        assert isinstance(problem_def.constraints, list)
    
    def test_create_problem_definition_with_optional_params(self) -> Any:
        """Test problem definition with all optional parameters."""
        problem_def = create_problem_definition(
            title="Test Project",
            description="Test description",
            project_type=ProjectType.REGRESSION,
            complexity=ProblemComplexity.COMPLEX,
            business_objective="Test objective",
            success_metrics=["Accuracy > 90%", "F1 > 0.8"],
            constraints=["Real-time processing"],
            assumptions=["Clean data"],
            risks=["Data quality issues"],
            stakeholders=["Data team"]
        )
        
        assert problem_def.success_metrics == ["Accuracy > 90%", "F1 > 0.8"]
        assert problem_def.constraints == ["Real-time processing"]
        assert problem_def.assumptions == ["Clean data"]
        assert problem_def.risks == ["Data quality issues"]
        assert problem_def.stakeholders == ["Data team"]
    
    def test_problem_definition_to_dict(self) -> Any:
        """Test problem definition serialization."""
        problem_def = create_problem_definition(
            title="Test Project",
            description="Test description",
            project_type=ProjectType.CLASSIFICATION,
            complexity=ProblemComplexity.MODERATE,
            business_objective="Test objective"
        )
        
        problem_dict = problem_def.to_dict()
        
        assert problem_dict['title'] == "Test Project"
        assert problem_dict['project_type'] == "classification"
        assert problem_dict['complexity'] == "moderate"
        assert isinstance(problem_dict['success_metrics'], list)
        assert isinstance(problem_dict['constraints'], list)

class TestDatasetInfo:
    """Test dataset information creation and validation."""
    
    def test_create_dataset_info_basic(self) -> Any:
        """Test basic dataset info creation."""
        dataset_info = create_dataset_info(
            name="Test Dataset",
            type=DatasetType.TABULAR,
            path="./data/test.csv"
        )
        
        assert dataset_info.name == "Test Dataset"
        assert dataset_info.type == DatasetType.TABULAR
        assert dataset_info.path == "./data/test.csv"
        assert dataset_info.size == 0
        assert isinstance(dataset_info.features, list)
        assert dataset_info.target_column is None
        assert dataset_info.validation_split == 0.2
        assert dataset_info.test_split == 0.1
    
    def test_create_dataset_info_with_optional_params(self) -> Any:
        """Test dataset info with all optional parameters."""
        dataset_info = create_dataset_info(
            name="Test Dataset",
            type=DatasetType.TEXT,
            path="./data/test.txt",
            features=["text", "label"],
            target_column="label",
            description="Test dataset",
            source="Test source",
            license="MIT"
        )
        
        assert dataset_info.features == ["text", "label"]
        assert dataset_info.target_column == "label"
        assert dataset_info.description == "Test dataset"
        assert dataset_info.source == "Test source"
        assert dataset_info.license == "MIT"
        assert dataset_info.last_updated is not None

class TestDatasetAnalyzer:
    """Test dataset analysis functionality."""
    
    @pytest.fixture
    def temp_csv_file(self) -> Any:
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create test data
            data = {
                'text': ['Hello world', 'Test message', 'Another test'],
                'category': ['A', 'B', 'A'],
                'value': [1.0, 2.0, 3.0],
                'missing': [1.0, None, 3.0]
            }
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_text_file(self) -> Any:
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test line.\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("Another test line with more words.\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("Third line for testing.\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_analyze_tabular_dataset(self, temp_csv_file) -> Any:
        """Test analysis of tabular dataset."""
        dataset_info = create_dataset_info(
            name="Test CSV",
            type=DatasetType.TABULAR,
            path=temp_csv_file
        )
        
        analyzer = DatasetAnalyzer(dataset_info)
        analysis = await analyzer.analyze_dataset()
        
        assert 'basic_info' in analysis
        assert 'data_quality' in analysis
        assert 'statistics' in analysis
        assert 'recommendations' in analysis
        
        basic_info = analysis['basic_info']
        assert basic_info['shape'] == (3, 4)
        assert 'text' in basic_info['columns']
        assert 'category' in basic_info['columns']
    
    @pytest.mark.asyncio
    async def test_analyze_text_dataset(self, temp_text_file) -> Any:
        """Test analysis of text dataset."""
        dataset_info = create_dataset_info(
            name="Test Text",
            type=DatasetType.TEXT,
            path=temp_text_file
        )
        
        analyzer = DatasetAnalyzer(dataset_info)
        analysis = await analyzer.analyze_dataset()
        
        assert 'basic_info' in analysis
        basic_info = analysis['basic_info']
        assert basic_info['total_lines'] == 3
        assert basic_info['total_words'] > 0
        assert basic_info['avg_line_length'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_data_quality(self, temp_csv_file) -> Any:
        """Test data quality analysis."""
        dataset_info = create_dataset_info(
            name="Test CSV",
            type=DatasetType.TABULAR,
            path=temp_csv_file
        )
        
        analyzer = DatasetAnalyzer(dataset_info)
        analysis = await analyzer.analyze_dataset()
        
        data_quality = analysis['data_quality']
        assert 'duplicates' in data_quality
        assert 'null_counts' in data_quality
        assert 'unique_counts' in data_quality
        
        # Check that missing values are detected
        null_counts = data_quality['null_counts']
        assert null_counts['missing'] == 1  # One missing value
    
    @pytest.mark.asyncio
    async def test_analyze_statistics(self, temp_csv_file) -> Any:
        """Test statistical analysis."""
        dataset_info = create_dataset_info(
            name="Test CSV",
            type=DatasetType.TABULAR,
            path=temp_csv_file
        )
        
        analyzer = DatasetAnalyzer(dataset_info)
        analysis = await analyzer.analyze_dataset()
        
        statistics = analysis['statistics']
        assert 'numeric_statistics' in statistics
        
        numeric_stats = statistics['numeric_statistics']
        assert 'value' in numeric_stats
        assert 'mean' in numeric_stats['value']
        assert 'std' in numeric_stats['value']
    
    @pytest.mark.asyncio
    async def test_analyze_missing_data(self, temp_csv_file) -> Any:
        """Test missing data analysis."""
        dataset_info = create_dataset_info(
            name="Test CSV",
            type=DatasetType.TABULAR,
            path=temp_csv_file
        )
        
        analyzer = DatasetAnalyzer(dataset_info)
        analysis = await analyzer.analyze_dataset()
        
        missing_data = analysis['missing_data']
        assert 'total_missing' in missing_data
        assert 'missing_percentage' in missing_data
        assert 'missing_by_column' in missing_data
        
        assert missing_data['total_missing'] == 1
        assert missing_data['missing_percentage'] > 0
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, temp_csv_file) -> Any:
        """Test recommendation generation."""
        dataset_info = create_dataset_info(
            name="Test CSV",
            type=DatasetType.TABULAR,
            path=temp_csv_file
        )
        
        analyzer = DatasetAnalyzer(dataset_info)
        analysis = await analyzer.analyze_dataset()
        
        recommendations = analysis['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend handling missing values
        missing_recs = [rec for rec in recommendations if 'missing' in rec.lower()]
        assert len(missing_recs) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_dataset_error_handling(self) -> Any:
        """Test error handling in dataset analysis."""
        dataset_info = create_dataset_info(
            name="Non-existent Dataset",
            type=DatasetType.TABULAR,
            path="./non_existent_file.csv"
        )
        
        analyzer = DatasetAnalyzer(dataset_info)
        analysis = await analyzer.analyze_dataset()
        
        assert 'basic_info' in analysis
        assert 'error' in analysis['basic_info']

class TestProjectInitializer:
    """Test project initialization functionality."""
    
    @pytest.fixture
    def temp_project_dir(self) -> Any:
        """Create a temporary directory for project testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_problem_definition(self) -> Any:
        """Create a sample problem definition."""
        return create_problem_definition(
            title="Test Project",
            description="Test project description",
            project_type=ProjectType.CLASSIFICATION,
            complexity=ProblemComplexity.MODERATE,
            business_objective="Test objective",
            success_metrics=["Accuracy > 90%"],
            constraints=["Real-time processing"],
            assumptions=["Clean data"],
            risks=["Data quality issues"],
            stakeholders=["Data team"]
        )
    
    @pytest.fixture
    def sample_dataset_info(self, temp_csv_file) -> Any:
        """Create a sample dataset info."""
        return create_dataset_info(
            name="Test Dataset",
            type=DatasetType.TABULAR,
            path=temp_csv_file,
            features=["text", "category", "value"],
            target_column="category",
            description="Test dataset",
            source="Test source"
        )
    
    @pytest.mark.asyncio
    async def test_create_project_structure(self, temp_project_dir) -> Any:
        """Test project directory structure creation."""
        initializer = ProjectInitializer("test_project", temp_project_dir)
        await initializer._create_project_structure()
        
        project_path = Path(temp_project_dir) / "test_project"
        
        # Check that all required directories exist
        required_dirs = [
            'data', 'models', 'notebooks', 'src', 'tests',
            'configs', 'logs', 'results', 'docs', 'scripts'
        ]
        
        for directory in required_dirs:
            assert (project_path / directory).exists()
            assert (project_path / directory).is_dir()
        
        # Check data subdirectories
        data_subdirs = ['raw', 'processed', 'interim', 'external']
        for subdir in data_subdirs:
            assert (project_path / 'data' / subdir).exists()
        
        # Check src subdirectories
        src_subdirs = ['data', 'models', 'features', 'utils', 'api']
        for subdir in src_subdirs:
            assert (project_path / 'src' / subdir).exists()
    
    @pytest.mark.asyncio
    async def test_generate_project_config(self, temp_project_dir, sample_problem_definition, sample_dataset_info) -> Any:
        """Test project configuration generation."""
        initializer = ProjectInitializer("test_project", temp_project_dir)
        await initializer._create_project_structure()
        
        config = await initializer._generate_project_config()
        
        assert 'project' in config
        assert 'problem_definition' in config
        assert 'dataset' in config
        assert 'training' in config
        assert 'optimization' in config
        
        # Check project info
        project_info = config['project']
        assert project_info['name'] == "test_project"
        assert project_info['type'] == "classification"
        assert project_info['complexity'] == "moderate"
        
        # Check problem definition
        problem_def = config['problem_definition']
        assert problem_def['title'] == "Test Project"
        assert problem_def['business_objective'] == "Test objective"
        
        # Check dataset info
        dataset_info = config['dataset']
        assert dataset_info['name'] == "Test Dataset"
        assert dataset_info['type'] == "tabular"
        
        # Check that config files were created
        config_path = Path(temp_project_dir) / "test_project" / "configs"
        assert (config_path / "project_config.yaml").exists()
        assert (config_path / "config.py").exists()
    
    @pytest.mark.asyncio
    async def test_create_documentation(self, temp_project_dir, sample_problem_definition, sample_dataset_info) -> Any:
        """Test documentation creation."""
        initializer = ProjectInitializer("test_project", temp_project_dir)
        await initializer._create_project_structure()
        
        # Mock dataset analysis results
        analysis_results = {
            'basic_info': {'shape': (100, 5)},
            'data_quality': {'duplicates': 0},
            'statistics': {'numeric_statistics': {}},
            'recommendations': ['Test recommendation']
        }
        
        await initializer._create_documentation(analysis_results)
        
        docs_path = Path(temp_project_dir) / "test_project" / "docs"
        
        # Check that documentation files were created
        assert (docs_path / "problem_definition.md").exists()
        assert (docs_path / "dataset_analysis.md").exists()
        assert (docs_path / "project_plan.md").exists()
        
        # Check README was created
        project_path = Path(temp_project_dir) / "test_project"
        assert (project_path / "README.md").exists()
    
    @pytest.mark.asyncio
    async def test_validate_project_setup(self, temp_project_dir, sample_problem_definition, sample_dataset_info) -> bool:
        """Test project setup validation."""
        initializer = ProjectInitializer("test_project", temp_project_dir)
        
        # Create project structure
        await initializer._create_project_structure()
        
        # Generate config
        config = await initializer._generate_project_config()
        
        # Validate setup
        validation = await initializer._validate_project_setup()
        
        assert 'structure' in validation
        assert 'configuration' in validation
        assert 'dataset' in validation
        assert 'dependencies' in validation
        assert 'overall_status' in validation
        
        # All validations should pass
        assert validation['overall_status'] == 'valid'
        assert validation['structure']['status'] == 'valid'
        assert validation['configuration']['status'] == 'valid'
        assert validation['dataset']['status'] == 'valid'
    
    @pytest.mark.asyncio
    async def test_validate_project_structure(self, temp_project_dir) -> bool:
        """Test project structure validation."""
        initializer = ProjectInitializer("test_project", temp_project_dir)
        
        # Test with missing structure
        validation = await initializer._validate_project_structure()
        assert validation['status'] == 'failed'
        assert len(validation['missing_directories']) > 0
        
        # Create structure and test again
        await initializer._create_project_structure()
        validation = await initializer._validate_project_structure()
        assert validation['status'] == 'valid'
        assert len(validation['missing_directories']) == 0
    
    @pytest.mark.asyncio
    async def test_validate_configuration(self, temp_project_dir) -> bool:
        """Test configuration validation."""
        initializer = ProjectInitializer("test_project", temp_project_dir)
        
        # Test with missing config
        validation = await initializer._validate_configuration()
        assert validation['status'] == 'failed'
        
        # Create structure and config
        await initializer._create_project_structure()
        await initializer._generate_project_config()
        
        validation = await initializer._validate_configuration()
        assert validation['status'] == 'valid'
    
    @pytest.mark.asyncio
    async def test_validate_dataset(self, temp_project_dir, temp_csv_file) -> bool:
        """Test dataset validation."""
        initializer = ProjectInitializer("test_project", temp_project_dir)
        
        # Test with non-existent dataset
        dataset_info = create_dataset_info(
            name="Test",
            type=DatasetType.TABULAR,
            path="./non_existent.csv"
        )
        initializer.dataset_info = dataset_info
        
        validation = await initializer._validate_dataset()
        assert validation['status'] == 'failed'
        
        # Test with existing dataset
        dataset_info = create_dataset_info(
            name="Test",
            type=DatasetType.TABULAR,
            path=temp_csv_file
        )
        initializer.dataset_info = dataset_info
        
        validation = await initializer._validate_dataset()
        assert validation['status'] == 'valid'
        assert 'shape' in validation
    
    @pytest.mark.asyncio
    async def test_validate_dependencies(self, temp_project_dir) -> bool:
        """Test dependencies validation."""
        initializer = ProjectInitializer("test_project", temp_project_dir)
        
        validation = await initializer._validate_dependencies()
        assert 'status' in validation
        assert 'missing_packages' in validation
        
        # Should have basic packages available
        assert validation['status'] == 'valid'

class TestIntegration:
    """Test integration with existing systems."""
    
    @pytest.fixture
    def temp_project_dir(self) -> Any:
        """Create a temporary directory for integration testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_create_project_integration(self, temp_project_dir, temp_csv_file) -> Any:
        """Test complete project creation integration."""
        # Create problem definition
        problem_def = create_problem_definition(
            title="Integration Test Project",
            description="Test project for integration",
            project_type=ProjectType.CLASSIFICATION,
            complexity=ProblemComplexity.MODERATE,
            business_objective="Test integration",
            success_metrics=["Accuracy > 90%"],
            constraints=["Real-time processing"],
            assumptions=["Clean data"],
            risks=["Data quality issues"],
            stakeholders=["Data team"]
        )
        
        # Create dataset info
        dataset_info = create_dataset_info(
            name="Integration Test Dataset",
            type=DatasetType.TABULAR,
            path=temp_csv_file,
            features=["text", "category", "value"],
            target_column="category",
            description="Integration test dataset",
            source="Test source"
        )
        
        # Create project
        result = await create_project(
            project_name="integration_test",
            problem_definition=problem_def,
            dataset_info=dataset_info,
            base_path=temp_project_dir
        )
        
        # Verify result structure
        assert result['project_name'] == "integration_test"
        assert result['status'] == "initialized"
        assert 'problem_definition' in result
        assert 'dataset_analysis' in result
        assert 'configuration' in result
        assert 'validation' in result
        
        # Verify project structure
        project_path = Path(temp_project_dir) / "integration_test"
        assert project_path.exists()
        
        # Verify key files exist
        assert (project_path / "README.md").exists()
        assert (project_path / "configs" / "project_config.yaml").exists()
        assert (project_path / "docs" / "problem_definition.md").exists()
        assert (project_path / "docs" / "dataset_analysis.md").exists()
        
        # Verify validation passed
        validation = result['validation']
        assert validation['overall_status'] == 'valid'
    
    @pytest.mark.asyncio
    async def test_analyze_dataset_integration(self, temp_csv_file) -> Any:
        """Test dataset analysis integration."""
        dataset_info = create_dataset_info(
            name="Analysis Test Dataset",
            type=DatasetType.TABULAR,
            path=temp_csv_file,
            features=["text", "category", "value"],
            target_column="category"
        )
        
        analysis = await analyze_dataset(dataset_info)
        
        # Verify analysis structure
        assert 'basic_info' in analysis
        assert 'data_quality' in analysis
        assert 'statistics' in analysis
        assert 'distributions' in analysis
        assert 'correlations' in analysis
        assert 'missing_data' in analysis
        assert 'outliers' in analysis
        assert 'recommendations' in analysis
        
        # Verify basic info
        basic_info = analysis['basic_info']
        assert 'shape' in basic_info
        assert 'columns' in basic_info
        assert basic_info['shape'][0] > 0  # Has rows
        assert basic_info['shape'][1] > 0  # Has columns
        
        # Verify recommendations
        recommendations = analysis['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 0

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_dataset_path(self) -> Any:
        """Test handling of invalid dataset path."""
        dataset_info = create_dataset_info(
            name="Invalid Dataset",
            type=DatasetType.TABULAR,
            path="./non_existent_file.csv"
        )
        
        analyzer = DatasetAnalyzer(dataset_info)
        analysis = await analyzer.analyze_dataset()
        
        assert 'basic_info' in analysis
        assert 'error' in analysis['basic_info']
    
    @pytest.mark.asyncio
    async def test_empty_dataset(self) -> Any:
        """Test handling of empty dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create empty CSV
            f.write("text,category,value\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            temp_file = f.name
        
        try:
            dataset_info = create_dataset_info(
                name="Empty Dataset",
                type=DatasetType.TABULAR,
                path=temp_file
            )
            
            analyzer = DatasetAnalyzer(dataset_info)
            analysis = await analyzer.analyze_dataset()
            
            assert 'basic_info' in analysis
            basic_info = analysis['basic_info']
            assert basic_info['shape'][0] == 0  # No rows
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_malformed_csv(self) -> Any:
        """Test handling of malformed CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create malformed CSV
            f.write("text,category,value\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("hello,world\n")  # Missing column
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("test,category,value,extra\n")  # Extra column
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            temp_file = f.name
        
        try:
            dataset_info = create_dataset_info(
                name="Malformed Dataset",
                type=DatasetType.TABULAR,
                path=temp_file
            )
            
            analyzer = DatasetAnalyzer(dataset_info)
            analysis = await analyzer.analyze_dataset()
            
            # Should handle gracefully
            assert 'basic_info' in analysis
        finally:
            os.unlink(temp_file)

class TestPerformance:
    """Test performance with large datasets."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_analysis(self) -> Any:
        """Test analysis of large dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create large dataset
            data = {
                'text': [f'Text {i}' for i in range(10000)],
                'category': [f'Category {i % 10}' for i in range(10000)],
                'value': [float(i) for i in range(10000)]
            }
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            dataset_info = create_dataset_info(
                name="Large Dataset",
                type=DatasetType.TABULAR,
                path=temp_file
            )
            
            analyzer = DatasetAnalyzer(dataset_info)
            
            # Time the analysis
            start_time = asyncio.get_event_loop().time()
            analysis = await analyzer.analyze_dataset()
            end_time = asyncio.get_event_loop().time()
            
            # Should complete within reasonable time
            assert end_time - start_time < 30  # 30 seconds max
            
            # Verify analysis results
            assert 'basic_info' in analysis
            basic_info = analysis['basic_info']
            assert basic_info['shape'] == (10000, 3)
            
        finally:
            os.unlink(temp_file)

# Run tests
match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 