"""
Advanced contract testing
"""

import pytest
from typing import Dict, Any, Protocol, runtime_checkable


@runtime_checkable
class CacheContract(Protocol):
    """Contract for cache implementations"""
    
    async def cache_project(self, description: str, config: Dict, info: Dict) -> None:
        """Must cache a project"""
        ...
    
    async def get_cached_project(self, description: str, config: Dict) -> Dict:
        """Must retrieve cached project"""
        ...


class TestAdvancedContracts:
    """Advanced contract tests"""
    
    @pytest.mark.async
    async def test_cache_contract(self, temp_dir):
        """Test that CacheManager satisfies cache contract"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        # Verify it implements the contract
        assert isinstance(cache, CacheContract)
        
        # Test contract methods
        await cache.cache_project("Test", {}, {"id": "test-123"})
        result = await cache.get_cached_project("Test", {})
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_validator_contract(self, temp_dir):
        """Test that Validator satisfies its contract"""
        from ..utils.validator import ProjectValidator
        
        validator = ProjectValidator()
        
        # Contract: validate_project returns dict with 'valid' key
        project_dir = temp_dir / "test_project"
        project_dir.mkdir()
        
        import asyncio
        result = asyncio.run(validator.validate_project(project_dir, {"name": "test"}))
        
        assert isinstance(result, dict)
        assert "valid" in result
        assert isinstance(result["valid"], bool)
    
    def test_rate_limiter_contract(self):
        """Test that RateLimiter satisfies its contract"""
        from ..utils.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        
        # Contract: is_allowed returns tuple (bool, dict)
        allowed, info = limiter.is_allowed("client", "endpoint")
        
        assert isinstance(allowed, bool)
        assert isinstance(info, dict)
        assert "limit" in info
        assert "remaining" in info

