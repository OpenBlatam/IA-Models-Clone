"""
Tests for FrontendGenerator

This test suite covers:
- Frontend generator initialization
- Framework validation
- Project generation
- Directory structure creation
- File generation
- Error handling

Test Generation Principles:
- Unique: Each test covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, errors, and boundaries
- Intuitive: Clear names and assertions express intent
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import shutil

from ..core.frontend_generator import (
    FrontendGenerator,
    _validate_project_info
)
from ..core.constants import FrameworkType


class TestFrontendGenerator:
    """Test suite for FrontendGenerator class"""

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_init_with_react_framework(self):
        """Test FrontendGenerator initializes with React framework"""
        # Happy path: React framework
        generator = FrontendGenerator(framework=FrameworkType.REACT)
        assert generator.framework == FrameworkType.REACT
        assert generator.file_generator is not None

    def test_init_with_default_framework(self):
        """Test FrontendGenerator uses default framework"""
        # Happy path: Default framework
        generator = FrontendGenerator()
        assert generator.framework == FrameworkType.REACT

    def test_init_raises_value_error_with_unsupported_framework(self):
        """Test FrontendGenerator raises ValueError with unsupported framework"""
        # Error condition: Unsupported framework
        with pytest.raises(ValueError, match="Unsupported framework"):
            FrontendGenerator(framework="unsupported")

    def test_init_raises_value_error_with_vue_framework(self):
        """Test FrontendGenerator raises ValueError with Vue (not yet supported)"""
        # Error condition: Vue not yet supported
        with pytest.raises(ValueError, match="Unsupported framework"):
            FrontendGenerator(framework=FrameworkType.VUE)

    def test_init_raises_value_error_with_nextjs_framework(self):
        """Test FrontendGenerator raises ValueError with NextJS (not yet supported)"""
        # Error condition: NextJS not yet supported
        with pytest.raises(ValueError, match="Unsupported framework"):
            FrontendGenerator(framework=FrameworkType.NEXTJS)

    # ========================================================================
    # Generate Method Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_generate_with_valid_parameters(self, frontend_generator, temp_dir):
        """Test generate creates frontend structure with valid parameters"""
        # Happy path: Valid parameters
        project_dir = temp_dir / "frontend"
        description = "A chat AI system"
        keywords = {"ai_type": "chat", "requires_api": True}
        project_info = {"name": "test_project", "version": "1.0.0"}
        
        with patch.object(frontend_generator.file_generator, 'create_directory_structure') as mock_create, \
             patch.object(frontend_generator.file_generator, 'generate_config_files') as mock_config, \
             patch.object(frontend_generator.file_generator, 'generate_html_and_css') as mock_html, \
             patch.object(frontend_generator.file_generator, 'generate_react_files') as mock_react, \
             patch.object(frontend_generator.file_generator, 'generate_utility_files') as mock_utils, \
             patch.object(frontend_generator.file_generator, 'generate_readme') as mock_readme:
            
            result = await frontend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert result is not None
            assert result["framework"] == "React"
            assert result["port"] == 3000
            assert "structure" in result
            mock_create.assert_called_once()
            mock_config.assert_called_once()
            mock_react.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_raises_value_error_with_none_project_dir(self, frontend_generator):
        """Test generate raises ValueError with None project_dir"""
        # Error condition: None project_dir
        with pytest.raises(ValueError, match="project_dir cannot be None"):
            await frontend_generator.generate(
                project_dir=None,
                description="Test",
                keywords={},
                project_info={"name": "test"}
            )

    @pytest.mark.asyncio
    async def test_generate_raises_value_error_with_empty_description(self, frontend_generator, temp_dir):
        """Test generate raises ValueError with empty description"""
        # Error condition: Empty description
        with pytest.raises(ValueError, match="description cannot be empty"):
            await frontend_generator.generate(
                project_dir=temp_dir,
                description="",
                keywords={},
                project_info={"name": "test"}
            )

    @pytest.mark.asyncio
    async def test_generate_creates_project_directory(self, frontend_generator, temp_dir):
        """Test generate creates project directory"""
        # Happy path: Directory creation
        project_dir = temp_dir / "frontend"
        description = "A test project"
        keywords = {}
        project_info = {"name": "test_project"}
        
        with patch.object(frontend_generator.file_generator, 'create_directory_structure'), \
             patch.object(frontend_generator.file_generator, 'generate_config_files'), \
             patch.object(frontend_generator.file_generator, 'generate_html_and_css'), \
             patch.object(frontend_generator.file_generator, 'generate_react_files'), \
             patch.object(frontend_generator.file_generator, 'generate_utility_files'), \
             patch.object(frontend_generator.file_generator, 'generate_readme'):
            
            await frontend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert project_dir.exists()

    @pytest.mark.asyncio
    async def test_generate_returns_frontend_info(self, frontend_generator, temp_dir):
        """Test generate returns frontend information"""
        # Happy path: Return value
        project_dir = temp_dir / "frontend"
        description = "A test project"
        keywords = {}
        project_info = {"name": "test_project", "version": "1.0.0"}
        
        with patch.object(frontend_generator.file_generator, 'create_directory_structure'), \
             patch.object(frontend_generator.file_generator, 'generate_config_files'), \
             patch.object(frontend_generator.file_generator, 'generate_html_and_css'), \
             patch.object(frontend_generator.file_generator, 'generate_react_files'), \
             patch.object(frontend_generator.file_generator, 'generate_utility_files'), \
             patch.object(frontend_generator.file_generator, 'generate_readme'):
            
            result = await frontend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert isinstance(result, dict)
            assert result["framework"] == "React"
            assert result["port"] == 3000
            assert "structure" in result
            assert "src" in result["structure"]

    @pytest.mark.asyncio
    async def test_generate_with_default_version(self, frontend_generator, temp_dir):
        """Test generate uses default version when not provided"""
        # Edge case: Default version
        project_dir = temp_dir / "frontend"
        description = "A test project"
        keywords = {}
        project_info = {"name": "test_project"}  # No version
        
        with patch.object(frontend_generator.file_generator, 'create_directory_structure'), \
             patch.object(frontend_generator.file_generator, 'generate_config_files') as mock_config, \
             patch.object(frontend_generator.file_generator, 'generate_html_and_css'), \
             patch.object(frontend_generator.file_generator, 'generate_react_files'), \
             patch.object(frontend_generator.file_generator, 'generate_utility_files'), \
             patch.object(frontend_generator.file_generator, 'generate_readme'):
            
            await frontend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            # Check that generate_config_files was called with default version
            call_args = mock_config.call_args
            assert call_args[0][2] == "1.0.0"  # Default version

    @pytest.mark.asyncio
    async def test_generate_with_custom_version(self, frontend_generator, temp_dir):
        """Test generate uses custom version when provided"""
        # Happy path: Custom version
        project_dir = temp_dir / "frontend"
        description = "A test project"
        keywords = {}
        project_info = {"name": "test_project", "version": "2.0.0"}
        
        with patch.object(frontend_generator.file_generator, 'create_directory_structure'), \
             patch.object(frontend_generator.file_generator, 'generate_config_files') as mock_config, \
             patch.object(frontend_generator.file_generator, 'generate_html_and_css'), \
             patch.object(frontend_generator.file_generator, 'generate_react_files'), \
             patch.object(frontend_generator.file_generator, 'generate_utility_files'), \
             patch.object(frontend_generator.file_generator, 'generate_readme'):
            
            await frontend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            # Check that generate_config_files was called with custom version
            call_args = mock_config.call_args
            assert call_args[0][2] == "2.0.0"

    @pytest.mark.asyncio
    async def test_generate_calls_all_file_generator_methods(self, frontend_generator, temp_dir):
        """Test generate calls all required file generator methods"""
        # Happy path: All methods called
        project_dir = temp_dir / "frontend"
        description = "A test project"
        keywords = {}
        project_info = {"name": "test_project"}
        
        with patch.object(frontend_generator.file_generator, 'create_directory_structure') as mock_create, \
             patch.object(frontend_generator.file_generator, 'generate_config_files') as mock_config, \
             patch.object(frontend_generator.file_generator, 'generate_html_and_css') as mock_html, \
             patch.object(frontend_generator.file_generator, 'generate_react_files') as mock_react, \
             patch.object(frontend_generator.file_generator, 'generate_utility_files') as mock_utils, \
             patch.object(frontend_generator.file_generator, 'generate_readme') as mock_readme:
            
            await frontend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            # Verify all methods were called
            mock_create.assert_called_once()
            mock_config.assert_called_once()
            mock_html.assert_called_once()
            mock_react.assert_called_once()
            mock_utils.assert_called_once()
            mock_readme.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_complex_keywords(self, frontend_generator, temp_dir):
        """Test generate handles complex keyword combinations"""
        # Edge case: Complex keywords
        project_dir = temp_dir / "frontend"
        description = "A complex AI system"
        keywords = {
            "ai_type": "chat",
            "requires_auth": True,
            "requires_websocket": True,
            "requires_file_upload": True
        }
        project_info = {"name": "test_project"}
        
        with patch.object(frontend_generator.file_generator, 'create_directory_structure'), \
             patch.object(frontend_generator.file_generator, 'generate_config_files'), \
             patch.object(frontend_generator.file_generator, 'generate_html_and_css'), \
             patch.object(frontend_generator.file_generator, 'generate_react_files'), \
             patch.object(frontend_generator.file_generator, 'generate_utility_files'), \
             patch.object(frontend_generator.file_generator, 'generate_readme'):
            
            result = await frontend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert result is not None
            assert result["framework"] == "React"

    @pytest.mark.asyncio
    async def test_generate_returns_complete_structure_info(self, frontend_generator, temp_dir):
        """Test generate returns complete structure information"""
        # Happy path: Structure info
        project_dir = temp_dir / "frontend"
        description = "A test project"
        keywords = {}
        project_info = {"name": "test_project"}
        
        with patch.object(frontend_generator.file_generator, 'create_directory_structure'), \
             patch.object(frontend_generator.file_generator, 'generate_config_files'), \
             patch.object(frontend_generator.file_generator, 'generate_html_and_css'), \
             patch.object(frontend_generator.file_generator, 'generate_react_files'), \
             patch.object(frontend_generator.file_generator, 'generate_utility_files'), \
             patch.object(frontend_generator.file_generator, 'generate_readme'):
            
            result = await frontend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert "structure" in result
            assert "src" in result["structure"]
            assert "components" in result["structure"]
            assert "pages" in result["structure"]
            assert "services" in result["structure"]


@pytest.fixture
def frontend_generator():
    """Fixture for FrontendGenerator instance"""
    return FrontendGenerator(framework=FrameworkType.REACT)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp(prefix="test_frontend_generator_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)
