"""
Tests for GitHubIntegration utility
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, Mock

from ..utils.github_integration import GitHubIntegration


class TestGitHubIntegration:
    """Test suite for GitHubIntegration"""

    def test_init(self):
        """Test GitHubIntegration initialization"""
        integration = GitHubIntegration()
        assert integration.github_token is None
        assert integration.github_username is None

    def test_init_with_token(self):
        """Test GitHubIntegration with token"""
        integration = GitHubIntegration(github_token="test_token")
        assert integration.github_token == "test_token"

    @pytest.mark.asyncio
    async def test_create_repository_success(self):
        """Test successful repository creation"""
        integration = GitHubIntegration(github_token="test_token")
        
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "html_url": "https://github.com/user/repo",
            "clone_url": "https://github.com/user/repo.git",
            "ssh_url": "git@github.com:user/repo.git",
            "full_name": "user/repo"
        }
        mock_response.raise_for_status = Mock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await integration.create_repository(
                project_name="test_repo",
                description="Test repository",
                private=False
            )
            
            assert result["success"] is True
            assert "repository_url" in result
            assert result["repository_url"] == "https://github.com/user/repo"

    @pytest.mark.asyncio
    async def test_create_repository_private(self):
        """Test creating private repository"""
        integration = GitHubIntegration(github_token="test_token")
        
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "html_url": "https://github.com/user/private_repo",
            "clone_url": "https://github.com/user/private_repo.git",
            "ssh_url": "git@github.com:user/private_repo.git",
            "full_name": "user/private_repo"
        }
        mock_response.raise_for_status = Mock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await integration.create_repository(
                project_name="private_repo",
                description="Private repository",
                private=True
            )
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_repository_no_token(self):
        """Test repository creation without token"""
        integration = GitHubIntegration()
        
        result = await integration.create_repository(
            project_name="test_repo",
            description="Test"
        )
        
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_repository_api_error(self):
        """Test handling API errors"""
        integration = GitHubIntegration(github_token="test_token")
        
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await integration.create_repository(
                project_name="test_repo",
                description="Test"
            )
            
            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_create_repository_with_auto_init(self):
        """Test repository creation with auto_init"""
        integration = GitHubIntegration(github_token="test_token")
        
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "html_url": "https://github.com/user/repo",
            "clone_url": "https://github.com/user/repo.git",
            "ssh_url": "git@github.com:user/repo.git",
            "full_name": "user/repo"
        }
        mock_response.raise_for_status = Mock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post
            
            result = await integration.create_repository(
                project_name="test_repo",
                description="Test",
                auto_init=True
            )
            
            # Verify auto_init was passed
            call_args = mock_post.call_args
            assert call_args[1]["json"]["auto_init"] is True

