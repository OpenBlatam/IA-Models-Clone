"""
Tests for BackendGenerator

This test suite covers:
- Backend generator initialization
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

from ..core.backend_generator import (
    BackendGenerator,
    _validate_project_info,
    _create_directory_structure,
    _write_file_safe
)
from ..core.constants import FrameworkType


class TestValidateProjectInfo:
    """Test suite for _validate_project_info function"""

    def test_validate_project_info_with_valid_info(self):
        """Test _validate_project_info accepts valid project info"""
        # Happy path: Valid info
        project_info = {"name": "test_project"}
        _validate_project_info(project_info)
        # Should not raise

    def test_validate_project_info_with_additional_fields(self):
        """Test _validate_project_info accepts info with additional fields"""
        # Happy path: Additional fields
        project_info = {
            "name": "test_project",
            "version": "1.0.0",
            "author": "Test Author"
        }
        _validate_project_info(project_info)
        # Should not raise

    def test_validate_project_info_raises_value_error_with_empty_dict(self):
        """Test _validate_project_info raises ValueError with empty dict"""
        # Error condition: Empty dict
        with pytest.raises(ValueError, match="project_info cannot be empty"):
            _validate_project_info({})

    def test_validate_project_info_raises_value_error_with_none(self):
        """Test _validate_project_info raises ValueError with None"""
        # Error condition: None
        with pytest.raises(ValueError, match="project_info cannot be empty"):
            _validate_project_info(None)

    def test_validate_project_info_raises_value_error_without_name(self):
        """Test _validate_project_info raises ValueError without name"""
        # Error condition: Missing name
        project_info = {"version": "1.0.0"}
        with pytest.raises(ValueError, match="project_info must contain 'name'"):
            _validate_project_info(project_info)

    def test_validate_project_info_raises_value_error_with_empty_name(self):
        """Test _validate_project_info raises ValueError with empty name"""
        # Error condition: Empty name
        project_info = {"name": ""}
        with pytest.raises(ValueError, match="project name cannot be empty"):
            _validate_project_info(project_info)

    def test_validate_project_info_raises_value_error_with_none_name(self):
        """Test _validate_project_info raises ValueError with None name"""
        # Error condition: None name
        project_info = {"name": None}
        with pytest.raises(ValueError, match="project name cannot be empty"):
            _validate_project_info(project_info)


class TestCreateDirectoryStructure:
    """Test suite for _create_directory_structure function"""

    def test_create_directory_structure_with_single_directory(self, temp_dir):
        """Test _create_directory_structure creates single directory"""
        # Happy path: Single directory
        directories = ["app"]
        _create_directory_structure(temp_dir, directories)
        assert (temp_dir / "app").exists()
        assert (temp_dir / "app").is_dir()

    def test_create_directory_structure_with_multiple_directories(self, temp_dir):
        """Test _create_directory_structure creates multiple directories"""
        # Happy path: Multiple directories
        directories = ["app", "tests", "config"]
        _create_directory_structure(temp_dir, directories)
        for directory in directories:
            assert (temp_dir / directory).exists()
            assert (temp_dir / directory).is_dir()

    def test_create_directory_structure_with_nested_directories(self, temp_dir):
        """Test _create_directory_structure creates nested directories"""
        # Happy path: Nested directories
        directories = ["app/api", "app/core", "app/models"]
        _create_directory_structure(temp_dir, directories)
        for directory in directories:
            assert (temp_dir / directory).exists()
            assert (temp_dir / directory).is_dir()

    def test_create_directory_structure_with_empty_list(self, temp_dir):
        """Test _create_directory_structure handles empty list"""
        # Edge case: Empty list
        directories = []
        _create_directory_structure(temp_dir, directories)
        # Should not raise

    def test_create_directory_structure_with_existing_directories(self, temp_dir):
        """Test _create_directory_structure handles existing directories"""
        # Edge case: Existing directories
        (temp_dir / "app").mkdir()
        directories = ["app", "tests"]
        _create_directory_structure(temp_dir, directories)
        assert (temp_dir / "app").exists()
        assert (temp_dir / "tests").exists()


class TestWriteFileSafe:
    """Test suite for _write_file_safe function"""

    def test_write_file_safe_creates_new_file(self, temp_dir):
        """Test _write_file_safe creates new file"""
        # Happy path: Create new file
        file_path = temp_dir / "test.txt"
        content = "Test content"
        _write_file_safe(file_path, content)
        assert file_path.exists()
        assert file_path.read_text() == content

    def test_write_file_safe_overwrites_existing_file(self, temp_dir):
        """Test _write_file_safe overwrites existing file"""
        # Happy path: Overwrite existing
        file_path = temp_dir / "test.txt"
        file_path.write_text("Old content")
        _write_file_safe(file_path, "New content")
        assert file_path.read_text() == "New content"

    def test_write_file_safe_with_custom_encoding(self, temp_dir):
        """Test _write_file_safe uses custom encoding"""
        # Edge case: Custom encoding
        file_path = temp_dir / "unicode.txt"
        content = "Test with unicode: ñáéíóú"
        _write_file_safe(file_path, content, encoding="utf-8")
        assert file_path.read_text(encoding="utf-8") == content

    def test_write_file_safe_with_empty_content(self, temp_dir):
        """Test _write_file_safe handles empty content"""
        # Edge case: Empty content
        file_path = temp_dir / "empty.txt"
        _write_file_safe(file_path, "")
        assert file_path.exists()
        assert file_path.read_text() == ""

    def test_write_file_safe_creates_parent_directories(self, temp_dir):
        """Test _write_file_safe creates parent directories"""
        # Happy path: Parent directories
        file_path = temp_dir / "parent" / "child" / "file.txt"
        _write_file_safe(file_path, "Content")
        assert file_path.exists()


class TestBackendGenerator:
    """Test suite for BackendGenerator class"""

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_init_with_fastapi_framework(self):
        """Test BackendGenerator initializes with FastAPI framework"""
        # Happy path: FastAPI framework
        generator = BackendGenerator(framework=FrameworkType.FASTAPI)
        assert generator.framework == FrameworkType.FASTAPI
        assert generator.dl_generator is not None
        assert generator.file_generator is not None

    def test_init_with_default_framework(self):
        """Test BackendGenerator uses default framework"""
        # Happy path: Default framework
        generator = BackendGenerator()
        assert generator.framework == FrameworkType.FASTAPI

    def test_init_raises_value_error_with_unsupported_framework(self):
        """Test BackendGenerator raises ValueError with unsupported framework"""
        # Error condition: Unsupported framework
        with pytest.raises(ValueError, match="Unsupported framework"):
            BackendGenerator(framework="unsupported")

    def test_init_raises_value_error_with_flask_framework(self):
        """Test BackendGenerator raises ValueError with Flask (not yet supported)"""
        # Error condition: Flask not yet supported
        with pytest.raises(ValueError, match="Unsupported framework"):
            BackendGenerator(framework=FrameworkType.FLASK)

    def test_init_raises_value_error_with_django_framework(self):
        """Test BackendGenerator raises ValueError with Django (not yet supported)"""
        # Error condition: Django not yet supported
        with pytest.raises(ValueError, match="Unsupported framework"):
            BackendGenerator(framework=FrameworkType.DJANGO)

    # ========================================================================
    # Generate Method Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_generate_with_valid_parameters(self, backend_generator, temp_dir):
        """Test generate creates backend structure with valid parameters"""
        # Happy path: Valid parameters
        project_dir = temp_dir / "backend"
        description = "A chat AI system"
        keywords = {"ai_type": "chat", "requires_api": True}
        project_info = {"name": "test_project", "version": "1.0.0"}
        
        with patch.object(backend_generator.file_generator, 'create_directory_structure') as mock_create, \
             patch.object(backend_generator.file_generator, 'generate_main_py') as mock_main, \
             patch.object(backend_generator.file_generator, 'generate_init_files') as mock_init:
            
            result = await backend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert result is not None
            assert "framework" in result
            mock_create.assert_called_once()
            mock_main.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_raises_value_error_with_none_project_dir(self, backend_generator):
        """Test generate raises ValueError with None project_dir"""
        # Error condition: None project_dir
        with pytest.raises(ValueError, match="project_dir cannot be None"):
            await backend_generator.generate(
                project_dir=None,
                description="Test",
                keywords={},
                project_info={"name": "test"}
            )

    @pytest.mark.asyncio
    async def test_generate_raises_value_error_with_empty_description(self, backend_generator, temp_dir):
        """Test generate raises ValueError with empty description"""
        # Error condition: Empty description
        with pytest.raises(ValueError, match="description cannot be empty"):
            await backend_generator.generate(
                project_dir=temp_dir,
                description="",
                keywords={},
                project_info={"name": "test"}
            )

    @pytest.mark.asyncio
    async def test_generate_with_deep_learning_keywords(self, backend_generator, temp_dir):
        """Test generate includes deep learning code when keywords indicate it"""
        # Happy path: Deep learning keywords
        project_dir = temp_dir / "backend"
        description = "A deep learning model"
        keywords = {
            "is_deep_learning": True,
            "requires_pytorch": True,
            "requires_training": True
        }
        project_info = {"name": "test_project"}
        
        with patch.object(backend_generator.dl_generator, 'generate_all') as mock_dl, \
             patch.object(backend_generator.file_generator, 'create_directory_structure'), \
             patch.object(backend_generator.file_generator, 'generate_main_py'), \
             patch.object(backend_generator.file_generator, 'generate_init_files'):
            
            await backend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            mock_dl.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_creates_project_directory(self, backend_generator, temp_dir):
        """Test generate creates project directory"""
        # Happy path: Directory creation
        project_dir = temp_dir / "backend"
        description = "A test project"
        keywords = {}
        project_info = {"name": "test_project"}
        
        with patch.object(backend_generator.file_generator, 'create_directory_structure'), \
             patch.object(backend_generator.file_generator, 'generate_main_py'), \
             patch.object(backend_generator.file_generator, 'generate_init_files'):
            
            await backend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert project_dir.exists()

    @pytest.mark.asyncio
    async def test_generate_returns_backend_info(self, backend_generator, temp_dir):
        """Test generate returns backend information"""
        # Happy path: Return value
        project_dir = temp_dir / "backend"
        description = "A test project"
        keywords = {}
        project_info = {"name": "test_project"}
        
        with patch.object(backend_generator.file_generator, 'create_directory_structure'), \
             patch.object(backend_generator.file_generator, 'generate_main_py'), \
             patch.object(backend_generator.file_generator, 'generate_init_files'):
            
            result = await backend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert isinstance(result, dict)
            assert "framework" in result

    @pytest.mark.asyncio
    async def test_generate_with_complex_keywords(self, backend_generator, temp_dir):
        """Test generate handles complex keyword combinations"""
        # Edge case: Complex keywords
        project_dir = temp_dir / "backend"
        description = "A complex AI system"
        keywords = {
            "ai_type": "chat",
            "requires_auth": True,
            "requires_database": True,
            "requires_websocket": True,
            "is_deep_learning": True,
            "requires_training": True
        }
        project_info = {"name": "test_project"}
        
        with patch.object(backend_generator.dl_generator, 'generate_all'), \
             patch.object(backend_generator.file_generator, 'create_directory_structure'), \
             patch.object(backend_generator.file_generator, 'generate_main_py'), \
             patch.object(backend_generator.file_generator, 'generate_init_files'):
            
            result = await backend_generator.generate(
                project_dir=project_dir,
                description=description,
                keywords=keywords,
                project_info=project_info
            )
            
            assert result is not None


@pytest.fixture
def backend_generator():
    """Fixture for BackendGenerator instance"""
    return BackendGenerator(framework=FrameworkType.FASTAPI)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp(prefix="test_backend_generator_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)
