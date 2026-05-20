"""
Tests for API Versioning
Tests for API version management and routing
"""

import pytest
from unittest.mock import Mock
from fastapi import Request

from core.infrastructure.api_versioning import (
    APIVersion,
    APIVersionManager
)


class TestAPIVersion:
    """Tests for APIVersion"""
    
    def test_version_enum(self):
        """Test API version enumeration"""
        assert APIVersion.V1.value == "v1"
        assert APIVersion.V2.value == "v2"
        assert APIVersion.V3.value == "v3"
        assert APIVersion.LATEST.value == "latest"
    
    def test_version_comparison(self):
        """Test version comparison"""
        assert APIVersion.V1 != APIVersion.V2
        assert APIVersion.V2 != APIVersion.V3


class TestAPIVersionManager:
    """Tests for APIVersionManager"""
    
    @pytest.fixture
    def version_manager(self):
        """Create API version manager"""
        return APIVersionManager(default_version=APIVersion.V1)
    
    def test_register_handler(self, version_manager):
        """Test registering version handler"""
        def handler():
            return "v1_response"
        
        version_manager.register_handler(
            "/analysis",
            APIVersion.V1,
            handler
        )
        
        assert "/analysis" in version_manager.version_handlers
        assert "v1" in version_manager.version_handlers["/analysis"]
    
    def test_get_handler(self, version_manager):
        """Test getting version handler"""
        def v1_handler():
            return "v1"
        
        def v2_handler():
            return "v2"
        
        version_manager.register_handler("/test", APIVersion.V1, v1_handler)
        version_manager.register_handler("/test", APIVersion.V2, v2_handler)
        
        handler = version_manager.get_handler("/test", APIVersion.V1)
        assert handler == v1_handler
        
        handler = version_manager.get_handler("/test", APIVersion.V2)
        assert handler == v2_handler
    
    def test_get_handler_default_version(self, version_manager):
        """Test getting handler with default version"""
        def default_handler():
            return "default"
        
        version_manager.register_handler("/test", APIVersion.V1, default_handler)
        
        handler = version_manager.get_handler("/test", None)
        assert handler == default_handler
    
    def test_mark_version_deprecated(self, version_manager):
        """Test marking version as deprecated"""
        version_manager.mark_deprecated(APIVersion.V1)
        
        assert "v1" in version_manager.deprecated_versions
    
    def test_is_version_deprecated(self, version_manager):
        """Test checking if version is deprecated"""
        version_manager.mark_deprecated(APIVersion.V1)
        
        assert version_manager.is_deprecated(APIVersion.V1) is True
        assert version_manager.is_deprecated(APIVersion.V2) is False
    
    def test_parse_version_from_header(self, version_manager):
        """Test parsing version from header"""
        request = Mock(spec=Request)
        request.headers = {"X-API-Version": "v2"}
        
        version = version_manager.parse_version(request)
        
        assert version == APIVersion.V2
    
    def test_parse_version_from_path(self, version_manager):
        """Test parsing version from path"""
        request = Mock(spec=Request)
        request.url = Mock(path="/v2/analysis")
        
        version = version_manager.parse_version_from_path("/v2/analysis")
        
        assert version == APIVersion.V2
    
    def test_get_latest_version(self, version_manager):
        """Test getting latest version"""
        def v1_handler():
            return "v1"
        
        def v2_handler():
            return "v2"
        
        version_manager.register_handler("/test", APIVersion.V1, v1_handler)
        version_manager.register_handler("/test", APIVersion.V2, v2_handler)
        
        handler = version_manager.get_handler("/test", APIVersion.LATEST)
        
        # Should return the latest registered version
        assert handler is not None



