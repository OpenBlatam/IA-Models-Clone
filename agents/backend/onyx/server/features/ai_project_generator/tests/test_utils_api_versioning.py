"""
Tests for APIVersionManager utility
"""

import pytest
from fastapi import APIRouter

from ..utils.api_versioning import APIVersionManager


class TestAPIVersionManager:
    """Test suite for APIVersionManager"""

    def test_init(self):
        """Test APIVersionManager initialization"""
        manager = APIVersionManager()
        assert manager.current_version == "v1"
        assert "v1" in manager.supported_versions
        assert manager.deprecated_versions == []

    def test_get_version_info(self):
        """Test getting version info"""
        manager = APIVersionManager()
        
        info = manager.get_version_info()
        
        assert "current_version" in info
        assert "supported_versions" in info
        assert "deprecated_versions" in info
        assert "api_base_url" in info
        assert info["current_version"] == "v1"
        assert info["api_base_url"] == "/api/v1"

    def test_create_versioned_router(self):
        """Test creating versioned router"""
        manager = APIVersionManager()
        
        router = manager.create_versioned_router("v1")
        
        assert isinstance(router, APIRouter)
        assert router.prefix == "/api/v1"

    def test_create_versioned_router_invalid(self):
        """Test creating router with invalid version"""
        manager = APIVersionManager()
        
        with pytest.raises(ValueError, match="no soportada"):
            manager.create_versioned_router("v99")

    def test_mark_deprecated(self):
        """Test marking version as deprecated"""
        manager = APIVersionManager()
        
        # Add a version first
        manager.supported_versions.append("v2")
        
        manager.mark_deprecated("v2", "2024-12-31")
        
        assert "v2" not in manager.supported_versions
        assert len(manager.deprecated_versions) == 1
        assert manager.deprecated_versions[0]["version"] == "v2"

    def test_add_version(self):
        """Test adding a new version"""
        manager = APIVersionManager()
        
        manager.add_version("v2")
        
        assert "v2" in manager.supported_versions

    def test_set_current_version(self):
        """Test setting current version"""
        manager = APIVersionManager()
        
        manager.supported_versions.append("v2")
        manager.set_current_version("v2")
        
        assert manager.current_version == "v2"

