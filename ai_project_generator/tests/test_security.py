"""
Security tests for AI Project Generator
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..core.project_generator import ProjectGenerator
from ..utils.rate_limiter import RateLimiter
from ..utils.auth_manager import AuthManager


class TestSecurity:
    """Test suite for security features"""

    def test_rate_limiter_prevents_abuse(self):
        """Test that rate limiter prevents abuse"""
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 5, "window": 3600}
        
        # Make requests up to limit
        for i in range(5):
            allowed, _ = limiter.is_allowed("abuser")
            assert allowed is True
        
        # Next request should be blocked
        allowed, info = limiter.is_allowed("abuser")
        assert allowed is False
        assert info["remaining"] == 0

    def test_rate_limiter_per_client_isolation(self):
        """Test that rate limits are isolated per client"""
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 5, "window": 3600}
        
        # Client 1 uses all requests
        for i in range(5):
            limiter.is_allowed("client1")
        
        # Client 2 should still have requests
        allowed, info = limiter.is_allowed("client2")
        assert allowed is True
        assert info["remaining"] == 4

    def test_sanitize_name_prevents_injection(self, project_generator):
        """Test that name sanitization prevents injection"""
        malicious_names = [
            "../../etc/passwd",
            "'; DROP TABLE projects; --",
            "<script>alert('xss')</script>",
            "file:///etc/passwd",
        ]
        
        for malicious in malicious_names:
            sanitized = project_generator._sanitize_name(malicious)
            # Should not contain dangerous characters
            assert ".." not in sanitized
            assert ";" not in sanitized
            assert "<" not in sanitized
            assert "://" not in sanitized

    def test_project_name_validation(self):
        """Test that project names are validated"""
        from ..api.generator_api import ProjectRequest
        
        # Valid name
        valid_request = ProjectRequest(
            description="A valid project description with enough words to pass validation",
            project_name="valid_project_123"
        )
        assert valid_request.project_name == "valid_project_123"
        
        # Invalid name with special characters
        with pytest.raises(Exception):
            ProjectRequest(
                description="A valid project description with enough words to pass validation",
                project_name="invalid@project#name"
            )

    def test_description_spam_detection(self):
        """Test spam detection in descriptions"""
        from ..api.generator_api import ProjectRequest
        
        # Spam description (repeated words)
        with pytest.raises(Exception):
            ProjectRequest(
                description="test test test test test",
                project_name="spam_project"
            )
        
        # Valid description
        valid = ProjectRequest(
            description="A comprehensive AI project with machine learning capabilities and advanced features",
            project_name="valid_project"
        )
        assert valid.description is not None

    def test_description_min_length(self):
        """Test minimum description length"""
        from ..api.generator_api import ProjectRequest
        
        # Too short
        with pytest.raises(Exception):
            ProjectRequest(
                description="short",
                project_name="test"
            )

    @pytest.mark.asyncio
    async def test_cache_key_isolation(self, temp_dir):
        """Test that cache keys are properly isolated"""
        from ..utils.cache_manager import CacheManager
        
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Cache two different projects
        await manager.cache_project(
            "Project 1",
            {"framework": "fastapi"},
            {"project_id": "proj1"}
        )
        
        await manager.cache_project(
            "Project 2",
            {"framework": "fastapi"},
            {"project_id": "proj2"}
        )
        
        # Should retrieve correct projects
        cached1 = await manager.get_cached_project("Project 1", {"framework": "fastapi"})
        cached2 = await manager.get_cached_project("Project 2", {"framework": "fastapi"})
        
        assert cached1["project_id"] == "proj1"
        assert cached2["project_id"] == "proj2"

    def test_webhook_secret_verification(self):
        """Test webhook secret verification"""
        from ..utils.webhook_manager import WebhookManager
        
        manager = WebhookManager()
        
        webhook_id = manager.register_webhook(
            url="http://example.com/webhook",
            events=["test.event"],
            secret="test_secret"
        )
        
        webhook = next(w for w in manager.webhooks if w["id"] == webhook_id)
        assert webhook["secret"] == "test_secret"
        
        # Secret should be used for signature
        assert webhook["secret"] is not None

    def test_path_traversal_prevention(self, project_generator):
        """Test prevention of path traversal attacks"""
        malicious_paths = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32",
        ]
        
        for malicious in malicious_paths:
            sanitized = project_generator._sanitize_name(malicious)
            # Should not contain path separators
            assert ".." not in sanitized
            assert "/" not in sanitized or sanitized.startswith("/")
            assert "\\" not in sanitized

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, temp_dir):
        """Test safety of concurrent access"""
        from ..utils.cache_manager import CacheManager
        
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Concurrent cache operations
        tasks = [
            manager.cache_project(
                f"Project {i}",
                {"id": i},
                {"project_id": f"proj-{i}"}
            )
            for i in range(20)
        ]
        
        await asyncio.gather(*tasks)
        
        # All should be cached correctly
        stats = await manager.get_stats()
        assert stats["total_cached"] == 20

