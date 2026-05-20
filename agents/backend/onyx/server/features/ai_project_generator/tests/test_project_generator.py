"""
Tests for ProjectGenerator

This test suite covers:
- Project generation functionality
- Name sanitization
- Keyword extraction
- Integration with sub-generators
- Error handling
- Edge cases

Test Generation Principles:
- Unique: Each test covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, errors, and boundaries
- Intuitive: Clear names and assertions express intent
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import string
import unicodedata

from ..core.project_generator import ProjectGenerator
from .test_helpers import TestHelpers


class TestProjectGenerator:
    """Test suite for ProjectGenerator"""

    def test_init(self, temp_dir):
        """Test ProjectGenerator initialization"""
        generator = ProjectGenerator(base_output_dir=str(temp_dir / "projects"))
        assert generator.base_output_dir == Path(temp_dir / "projects")
        assert generator.backend_framework == "fastapi"
        assert generator.frontend_framework == "react"
        assert generator.base_output_dir.exists()

    def test_init_creates_directory(self, temp_dir):
        """Test that initialization creates output directory"""
        output_dir = temp_dir / "new_projects"
        generator = ProjectGenerator(base_output_dir=str(output_dir))
        assert output_dir.exists()

    # ========================================================================
    # Name Sanitization Tests - Comprehensive Coverage
    # ========================================================================
    
    def test_sanitize_name_with_typical_inputs(self, project_generator):
        """Test name sanitization with typical project names"""
        # Happy path: Standard project names
        assert project_generator._sanitize_name("Test Project") == "test_project"
        assert project_generator._sanitize_name("My Awesome Project") == "my_awesome_project"
        assert project_generator._sanitize_name("AI Chat Bot") == "ai_chat_bot"
    
    def test_sanitize_name_with_special_characters(self, project_generator):
        """Test name sanitization handles special characters correctly"""
        # Edge case: Various special characters
        assert project_generator._sanitize_name("Test-Project") == "test_project"
        assert project_generator._sanitize_name("Test@Project#123") == "test_project_123"
        assert project_generator._sanitize_name("Test$Project%100") == "test_project_100"
        assert project_generator._sanitize_name("Test&Project*Name") == "test_project_name"
    
    def test_sanitize_name_with_whitespace(self, project_generator):
        """Test name sanitization handles whitespace correctly"""
        # Edge case: Various whitespace patterns
        assert project_generator._sanitize_name("  Test  Project  ") == "test_project"
        assert project_generator._sanitize_name("Test\tProject") == "test_project"
        assert project_generator._sanitize_name("Test\nProject") == "test_project"
        assert project_generator._sanitize_name("Test  Multiple   Spaces") == "test_multiple_spaces"
    
    def test_sanitize_name_with_unicode_characters(self, project_generator):
        """Test name sanitization handles Unicode characters"""
        # Edge case: Unicode and international characters
        assert project_generator._sanitize_name("Proyecto Español") == "proyecto_espaol"
        assert project_generator._sanitize_name("プロジェクト") == "___"  # Japanese characters
        assert project_generator._sanitize_name("Проект") == "___"  # Cyrillic characters
        assert project_generator._sanitize_name("Projet Français") == "projet_franais"
    
    def test_sanitize_name_with_numbers(self, project_generator):
        """Test name sanitization preserves numbers correctly"""
        # Edge case: Numbers in various positions
        assert project_generator._sanitize_name("Project 123") == "project_123"
        assert project_generator._sanitize_name("123 Project") == "123_project"
        assert project_generator._sanitize_name("Project v2.0") == "project_v2_0"
        assert project_generator._sanitize_name("Test-2024") == "test_2024"
    
    def test_sanitize_name_with_length_boundaries(self, project_generator):
        """Test name sanitization handles length boundaries"""
        # Boundary value: Maximum length
        long_name = "A" * 100
        result = project_generator._sanitize_name(long_name)
        assert len(result) == 50  # MAX_PROJECT_NAME_LENGTH
        assert result == "a" * 50
        
        # Boundary value: Single character
        assert project_generator._sanitize_name("A") == "a"
        
        # Boundary value: Exactly at limit
        exact_length_name = "A" * 50
        result = project_generator._sanitize_name(exact_length_name)
        assert len(result) == 50
    
    def test_sanitize_name_with_mixed_case(self, project_generator):
        """Test name sanitization normalizes case correctly"""
        # Edge case: Various case patterns
        assert project_generator._sanitize_name("TEST_PROJECT") == "test_project"
        assert project_generator._sanitize_name("TestProject") == "testproject"
        assert project_generator._sanitize_name("testProject") == "testproject"
        assert project_generator._sanitize_name("TeSt PrOjEcT") == "test_project"
    
    def test_sanitize_name_raises_value_error_with_empty_string(self, project_generator):
        """Test name sanitization raises ValueError with empty string"""
        # Error condition: Empty input
        with pytest.raises(ValueError, match="name cannot be empty"):
            project_generator._sanitize_name("")
    
    def test_sanitize_name_raises_value_error_with_whitespace_only(self, project_generator):
        """Test name sanitization raises ValueError with whitespace-only string"""
        # Error condition: Only whitespace
        with pytest.raises(ValueError, match="name cannot be empty"):
            project_generator._sanitize_name("   ")
        with pytest.raises(ValueError, match="name cannot be empty"):
            project_generator._sanitize_name("\t\n\r")
    
    def test_sanitize_name_with_consecutive_special_chars(self, project_generator):
        """Test name sanitization handles consecutive special characters"""
        # Edge case: Multiple special characters in a row
        assert project_generator._sanitize_name("Test---Project") == "test_project"
        assert project_generator._sanitize_name("Test___Project") == "test_project"
        assert project_generator._sanitize_name("Test...Project") == "test_project"
    
    def test_sanitize_name_preserves_underscores(self, project_generator):
        """Test name sanitization preserves existing underscores"""
        # Edge case: Underscores in input
        assert project_generator._sanitize_name("test_project") == "test_project"
        assert project_generator._sanitize_name("test__project") == "test__project"
        assert project_generator._sanitize_name("_test_project_") == "_test_project_"

    # ========================================================================
    # Keyword Extraction Tests - Comprehensive Coverage
    # ========================================================================
    
    def test_extract_keywords_chat_with_typical_description(self, project_generator):
        """Test keyword extraction identifies chat AI from typical description"""
        # Happy path: Clear chat AI description
        description = "A chat AI system that responds to user questions"
        keywords = project_generator._extract_keywords(description)
        assert keywords["ai_type"] == "chat"
        assert keywords["requires_api"] is True
    
    def test_extract_keywords_chat_with_variations(self, project_generator):
        """Test keyword extraction identifies chat AI from various phrasings"""
        # Edge case: Different ways to describe chat AI
        descriptions = [
            "A conversational AI assistant",
            "A chatbot that helps users",
            "An AI that chats with customers",
            "A messaging AI system"
        ]
        for desc in descriptions:
            keywords = project_generator._extract_keywords(desc)
            assert keywords["ai_type"] == "chat", f"Failed for: {desc}"

    def test_extract_keywords_vision(self, project_generator):
        """Test keyword extraction for vision AI"""
        description = "An image recognition system that detects objects in photos"
        keywords = project_generator._extract_keywords(description)
        assert keywords["ai_type"] == "vision"
        assert keywords["requires_file_upload"] is True

    def test_extract_keywords_audio(self, project_generator):
        """Test keyword extraction for audio AI"""
        description = "A speech recognition system that transcribes audio"
        keywords = project_generator._extract_keywords(description)
        assert keywords["ai_type"] == "audio"

    def test_extract_keywords_websocket(self, project_generator):
        """Test keyword extraction for websocket requirement"""
        description = "A real-time chat system with websocket support"
        keywords = project_generator._extract_keywords(description)
        assert keywords["requires_websocket"] is True

    def test_extract_keywords_database(self, project_generator):
        """Test keyword extraction for database requirement"""
        description = "A system that stores user data in a database"
        keywords = project_generator._extract_keywords(description)
        assert keywords["requires_database"] is True

    def test_extract_keywords_auth(self, project_generator):
        """Test keyword extraction for authentication requirement"""
        description = "A system with user authentication and login"
        keywords = project_generator._extract_keywords(description)
        assert keywords["requires_auth"] is True

    def test_extract_keywords_deep_learning(self, project_generator):
        """Test keyword extraction for deep learning"""
        description = "A deep learning model using PyTorch for image classification"
        keywords = project_generator._extract_keywords(description)
        assert keywords["is_deep_learning"] is True
        assert keywords["requires_pytorch"] is True

    def test_extract_keywords_transformer(self, project_generator):
        """Test keyword extraction for transformer models"""
        description = "A transformer model for natural language processing"
        keywords = project_generator._extract_keywords(description)
        assert keywords["is_transformer"] is True

    def test_extract_keywords_llm(self, project_generator):
        """Test keyword extraction for LLM"""
        description = "A large language model for text generation"
        keywords = project_generator._extract_keywords(description)
        assert keywords["is_llm"] is True

    def test_extract_keywords_openai(self, project_generator):
        """Test keyword extraction for OpenAI provider"""
        description = "A chat system using OpenAI GPT models"
        keywords = project_generator._extract_keywords(description)
        assert "openai" in keywords["model_providers"]

    def test_extract_keywords_complexity(self, project_generator):
        """Test complexity detection"""
        simple_desc = "A simple chat bot"
        medium_desc = "A standard AI system"
        complex_desc = "An advanced enterprise AI platform"
        
        simple_keywords = project_generator._extract_keywords(simple_desc)
        medium_keywords = project_generator._extract_keywords(medium_desc)
        complex_keywords = project_generator._extract_keywords(complex_desc)
        
        assert simple_keywords["complexity"] == "simple"
        assert medium_keywords["complexity"] == "medium"
        assert complex_keywords["complexity"] == "complex"

    # ========================================================================
    # Project Generation Tests - Comprehensive Coverage
    # ========================================================================
    
    @pytest.mark.asyncio
    async def test_generate_project_with_all_parameters(self, project_generator, temp_dir):
        """Test project generation with all parameters provided"""
        # Happy path: All parameters explicitly provided
        description = "A simple chat AI system"
        project_name = "test_chat"
        author = "Test Author"
        version = "1.0.0"
        
        with patch.object(project_generator.backend_generator, 'generate', new_callable=AsyncMock) as mock_backend, \
             patch.object(project_generator.frontend_generator, 'generate', new_callable=AsyncMock) as mock_frontend, \
             patch.object(project_generator.test_generator, 'generate_backend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.test_generator, 'generate_frontend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.cicd_generator, 'generate_github_actions', new_callable=AsyncMock), \
             patch.object(project_generator.cache_manager, 'get_cached_project', new_callable=AsyncMock, return_value=None), \
             patch.object(project_generator.cache_manager, 'cache_project', new_callable=AsyncMock), \
             patch.object(project_generator.validator, 'validate_project', new_callable=AsyncMock, return_value={"valid": True}):
            
            mock_backend.return_value = {"framework": "fastapi", "files": []}
            mock_frontend.return_value = {"framework": "react", "files": []}
            
            result = await project_generator.generate_project(
                description=description,
                project_name=project_name,
                author=author,
                version=version
            )
            
            # Verify all expected fields are present
            assert "project_id" in result
            assert result["name"] == project_name
            assert result["author"] == author
            assert result["version"] == version
            assert result["description"] == description
            assert "keywords" in result
            assert "created_at" in result
            assert "project_dir" in result
            
            # Verify generators were called
            mock_backend.assert_called_once()
            mock_frontend.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_project_with_minimal_parameters(self, project_generator, temp_dir):
        """Test project generation with only required parameters"""
        # Happy path: Minimal required parameters
        description = "A simple chat AI system"
        
        with patch.object(project_generator.backend_generator, 'generate', new_callable=AsyncMock) as mock_backend, \
             patch.object(project_generator.frontend_generator, 'generate', new_callable=AsyncMock) as mock_frontend, \
             patch.object(project_generator.test_generator, 'generate_backend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.test_generator, 'generate_frontend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.cicd_generator, 'generate_github_actions', new_callable=AsyncMock), \
             patch.object(project_generator.cache_manager, 'get_cached_project', new_callable=AsyncMock, return_value=None), \
             patch.object(project_generator.cache_manager, 'cache_project', new_callable=AsyncMock), \
             patch.object(project_generator.validator, 'validate_project', new_callable=AsyncMock, return_value={"valid": True}):
            
            mock_backend.return_value = {"framework": "fastapi", "files": []}
            mock_frontend.return_value = {"framework": "react", "files": []}
            
            result = await project_generator.generate_project(description=description)
            
            # Verify project was generated with auto-generated name
            assert "name" in result
            assert result["name"]  # Should have a generated name
            assert result["description"] == description
            assert result["author"]  # Should have default author
            assert result["version"]  # Should have default version

    @pytest.mark.asyncio
    async def test_generate_project_with_cache(self, project_generator):
        """Test project generation with cached result"""
        description = "A cached project"
        cached_result = {
            "project_id": "cached-123",
            "name": "cached_project",
            "description": description
        }
        
        with patch.object(project_generator.cache_manager, 'get_cached_project', 
                         new_callable=AsyncMock, return_value=cached_result):
            
            result = await project_generator.generate_project(description=description)
            assert result == cached_result

    @pytest.mark.asyncio
    async def test_generate_project_auto_name(self, project_generator, temp_dir):
        """Test project generation with auto-generated name"""
        description = "A test AI project for testing"
        
        with patch.object(project_generator.backend_generator, 'generate', new_callable=AsyncMock) as mock_backend, \
             patch.object(project_generator.frontend_generator, 'generate', new_callable=AsyncMock) as mock_frontend, \
             patch.object(project_generator.test_generator, 'generate_backend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.test_generator, 'generate_frontend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.cicd_generator, 'generate_github_actions', new_callable=AsyncMock), \
             patch.object(project_generator.cache_manager, 'get_cached_project', new_callable=AsyncMock, return_value=None), \
             patch.object(project_generator.cache_manager, 'cache_project', new_callable=AsyncMock), \
             patch.object(project_generator.validator, 'validate_project', new_callable=AsyncMock, return_value={"valid": True}):
            
            mock_backend.return_value = {"framework": "fastapi"}
            mock_frontend.return_value = {"framework": "react"}
            
            result = await project_generator.generate_project(description=description)
            assert "name" in result
            assert result["name"]  # Should have a generated name

    @pytest.mark.asyncio
    async def test_generate_project_duplicate_name(self, project_generator, temp_dir):
        """Test project generation with duplicate name handling"""
        description = "Test project"
        project_name = "duplicate_test"
        
        # Create existing directory
        existing_dir = project_generator.base_output_dir / project_name
        existing_dir.mkdir(parents=True, exist_ok=True)
        
        with patch.object(project_generator.backend_generator, 'generate', new_callable=AsyncMock) as mock_backend, \
             patch.object(project_generator.frontend_generator, 'generate', new_callable=AsyncMock) as mock_frontend, \
             patch.object(project_generator.test_generator, 'generate_backend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.test_generator, 'generate_frontend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.cicd_generator, 'generate_github_actions', new_callable=AsyncMock), \
             patch.object(project_generator.cache_manager, 'get_cached_project', new_callable=AsyncMock, return_value=None), \
             patch.object(project_generator.cache_manager, 'cache_project', new_callable=AsyncMock), \
             patch.object(project_generator.validator, 'validate_project', new_callable=AsyncMock, return_value={"valid": True}):
            
            mock_backend.return_value = {"framework": "fastapi"}
            mock_frontend.return_value = {"framework": "react"}
            
            result = await project_generator.generate_project(
                description=description,
                project_name=project_name
            )
            
            # Should have timestamp appended or different name
            assert result["name"] != project_name or "_" in result["name"]

    @pytest.mark.asyncio
    async def test_generate_project_raises_value_error_with_empty_description(self, project_generator):
        """Test project generation raises ValueError with empty description"""
        # Error condition: Empty description
        with pytest.raises(ValueError, match="description cannot be empty"):
            await project_generator.generate_project(description="")
    
    @pytest.mark.asyncio
    async def test_generate_project_raises_value_error_with_empty_author(self, project_generator):
        """Test project generation raises ValueError with empty author"""
        # Error condition: Empty author
        with pytest.raises(ValueError, match="author cannot be empty"):
            await project_generator.generate_project(
                description="Test project",
                author=""
            )
    
    @pytest.mark.asyncio
    async def test_generate_project_raises_value_error_with_empty_version(self, project_generator):
        """Test project generation raises ValueError with empty version"""
        # Error condition: Empty version
        with pytest.raises(ValueError, match="version cannot be empty"):
            await project_generator.generate_project(
                description="Test project",
                version=""
            )
    
    @pytest.mark.asyncio
    async def test_generate_project_propagates_backend_error(self, project_generator):
        """Test project generation propagates errors from backend generator"""
        # Error condition: Backend generator failure
        description = "A project that will fail"
        backend_error = Exception("Backend generation failed")
        
        with patch.object(project_generator.backend_generator, 'generate', 
                         new_callable=AsyncMock, side_effect=backend_error), \
             patch.object(project_generator.cache_manager, 'get_cached_project', 
                         new_callable=AsyncMock, return_value=None):
            
            with pytest.raises(Exception, match="Backend generation failed"):
                await project_generator.generate_project(description=description)
    
    @pytest.mark.asyncio
    async def test_generate_project_propagates_frontend_error(self, project_generator):
        """Test project generation propagates errors from frontend generator"""
        # Error condition: Frontend generator failure
        description = "A project that will fail"
        frontend_error = Exception("Frontend generation failed")
        
        with patch.object(project_generator.backend_generator, 'generate', 
                         new_callable=AsyncMock, return_value={"framework": "fastapi"}), \
             patch.object(project_generator.frontend_generator, 'generate', 
                         new_callable=AsyncMock, side_effect=frontend_error), \
             patch.object(project_generator.cache_manager, 'get_cached_project', 
                         new_callable=AsyncMock, return_value=None):
            
            with pytest.raises(Exception, match="Frontend generation failed"):
                await project_generator.generate_project(description=description)
    
    @pytest.mark.asyncio
    async def test_generate_project_handles_validation_errors_gracefully(self, project_generator, temp_dir):
        """Test project generation handles validation errors without failing"""
        # Edge case: Validation errors don't stop generation
        description = "A project with validation issues"
        
        with patch.object(project_generator.backend_generator, 'generate', new_callable=AsyncMock) as mock_backend, \
             patch.object(project_generator.frontend_generator, 'generate', new_callable=AsyncMock) as mock_frontend, \
             patch.object(project_generator.test_generator, 'generate_backend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.test_generator, 'generate_frontend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.cicd_generator, 'generate_github_actions', new_callable=AsyncMock), \
             patch.object(project_generator.cache_manager, 'get_cached_project', new_callable=AsyncMock, return_value=None), \
             patch.object(project_generator.cache_manager, 'cache_project', new_callable=AsyncMock), \
             patch.object(project_generator.validator, 'validate_project', new_callable=AsyncMock, 
                         return_value={"valid": False, "error_count": 2, "errors": ["Error 1", "Error 2"]}):
            
            mock_backend.return_value = {"framework": "fastapi", "files": []}
            mock_frontend.return_value = {"framework": "react", "files": []}
            
            result = await project_generator.generate_project(description=description)
            
            # Project should still be generated despite validation errors
            assert result is not None
            assert result["validation"]["valid"] is False
            assert result["validation"]["error_count"] == 2

    def test_extract_keywords_with_comprehensive_feature_list(self, project_generator):
        """Test extraction identifies all features from comprehensive description"""
        # Happy path: Complex description with many features
        description = """
        An advanced AI system with:
        - User authentication and login
        - Database storage with PostgreSQL
        - WebSocket for real-time communication
        - File upload capabilities
        - Redis cache
        - Background task queue
        - Streaming data processing
        - Dashboard and admin panel
        - REST API and GraphQL
        - Monitoring and logging
        - Docker containers
        - Automated testing
        """
        keywords = project_generator._extract_keywords(description)
        
        # Verify all major features are detected
        assert keywords["requires_auth"] is True, "Authentication should be detected"
        assert keywords["requires_database"] is True, "Database should be detected"
        assert keywords["requires_websocket"] is True, "WebSocket should be detected"
        assert keywords["requires_file_upload"] is True, "File upload should be detected"
        assert keywords["requires_cache"] is True, "Cache should be detected"
        assert keywords["requires_queue"] is True, "Queue should be detected"
        assert keywords["requires_streaming"] is True, "Streaming should be detected"
        
        # Verify feature list contains expected items
        features = keywords.get("features", [])
        assert "dashboard" in features or "admin" in str(features).lower()
        assert "rest_api" in features or "api" in str(features).lower()
        assert "monitoring" in features or "logging" in features
        assert "testing" in features or "test" in str(features).lower()
        assert "docker" in features or "container" in str(features).lower()
    
    def test_extract_keywords_with_empty_description(self, project_generator):
        """Test keyword extraction handles empty description"""
        # Edge case: Empty description
        keywords = project_generator._extract_keywords("")
        assert isinstance(keywords, dict)
        assert "ai_type" in keywords
    
    def test_extract_keywords_with_very_long_description(self, project_generator):
        """Test keyword extraction handles very long descriptions"""
        # Edge case: Very long description
        long_description = "A chat AI system. " * 1000
        keywords = project_generator._extract_keywords(long_description)
        assert isinstance(keywords, dict)
        assert "ai_type" in keywords
    
    def test_extract_keywords_with_special_characters(self, project_generator):
        """Test keyword extraction handles special characters in description"""
        # Edge case: Special characters
        description = "A chat AI system with @#$% special characters & symbols!"
        keywords = project_generator._extract_keywords(description)
        assert isinstance(keywords, dict)
        assert keywords["ai_type"] == "chat"
    
    def test_extract_keywords_with_multilingual_description(self, project_generator):
        """Test keyword extraction handles multilingual descriptions"""
        # Edge case: Multilingual content
        description = "Un système d'IA de chat / Un sistema de chat con IA"
        keywords = project_generator._extract_keywords(description)
        assert isinstance(keywords, dict)
        # Should still identify as chat despite different language
        assert keywords["ai_type"] == "chat"

