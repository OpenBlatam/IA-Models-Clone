"""
Real-world use case tests
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..core.project_generator import ProjectGenerator
from ..core.continuous_generator import ContinuousGenerator


class TestUseCases:
    """Test suite for real-world use cases"""

    @pytest.mark.asyncio
    async def test_use_case_chat_ai(self, project_generator, temp_dir):
        """Test generating a chat AI project"""
        description = """
        A conversational AI chatbot that can answer questions about programming,
        help with debugging, and provide code examples. It should use OpenAI GPT models
        and have a web interface for users to interact with.
        """
        
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
                project_name="chat_ai_bot"
            )
            
            keywords = result["keywords"]
            assert keywords["ai_type"] == "chat"
            assert "openai" in keywords["model_providers"]
            assert keywords["requires_api"] is True

    @pytest.mark.asyncio
    async def test_use_case_image_classification(self, project_generator, temp_dir):
        """Test generating an image classification project"""
        description = """
        An image classification system using deep learning that can classify images
        into different categories. It should use PyTorch, have a training pipeline,
        and provide a REST API for predictions. Include a web interface for uploading images.
        """
        
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
                project_name="image_classifier"
            )
            
            keywords = result["keywords"]
            assert keywords["ai_type"] == "vision"
            assert keywords["is_deep_learning"] is True
            assert keywords["requires_pytorch"] is True
            assert keywords["requires_file_upload"] is True

    @pytest.mark.asyncio
    async def test_use_case_realtime_chat(self, project_generator, temp_dir):
        """Test generating a real-time chat system"""
        description = """
        A real-time chat application with WebSocket support for instant messaging.
        Users can create rooms, send messages, and receive notifications. It should
        have user authentication and store messages in a database.
        """
        
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
                project_name="realtime_chat"
            )
            
            keywords = result["keywords"]
            assert keywords["requires_websocket"] is True
            assert keywords["requires_auth"] is True
            assert keywords["requires_database"] is True

    @pytest.mark.asyncio
    async def test_use_case_batch_processing(self, project_generator, temp_dir):
        """Test generating multiple projects in batch"""
        descriptions = [
            "A simple calculator AI",
            "An image recognition system",
            "A text summarization tool",
            "A sentiment analysis API",
            "A recommendation engine"
        ]
        
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
            
            tasks = [
                project_generator.generate_project(
                    description=desc,
                    project_name=f"batch_{i}"
                )
                for i, desc in enumerate(descriptions)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all("project_id" in r for r in results)

    @pytest.mark.asyncio
    async def test_use_case_continuous_generation(self, continuous_generator, temp_dir):
        """Test continuous generation workflow"""
        # Add multiple projects with different priorities
        continuous_generator.add_project("High priority project", priority=10)
        continuous_generator.add_project("Low priority project", priority=1)
        continuous_generator.add_project("Medium priority project", priority=5)
        
        # Verify queue
        queue_info = continuous_generator.get_queue()
        assert queue_info["total"] == 3
        assert queue_info["pending"] == 3
        
        # Verify priority sorting
        priorities = [p["priority"] for p in continuous_generator.queue]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_use_case_project_with_all_features(self, project_generator, temp_dir):
        """Test generating a project with all features"""
        description = """
        A comprehensive AI platform with:
        - User authentication and authorization
        - Database for storing data
        - WebSocket for real-time updates
        - File upload capabilities
        - Redis cache for performance
        - Background task queue
        - REST API and GraphQL
        - Admin dashboard
        - Monitoring and logging
        - Docker deployment
        - Automated testing
        """
        
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
                project_name="full_featured_platform"
            )
            
            keywords = result["keywords"]
            assert keywords["requires_auth"] is True
            assert keywords["requires_database"] is True
            assert keywords["requires_websocket"] is True
            assert keywords["requires_file_upload"] is True
            assert keywords["requires_cache"] is True
            assert keywords["requires_queue"] is True

