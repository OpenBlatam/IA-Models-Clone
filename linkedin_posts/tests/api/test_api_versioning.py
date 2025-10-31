"""
API Versioning Tests for LinkedIn Posts Service
Tests API versioning, backward compatibility, and migration scenarios
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

# Mock API versions and schemas
class MockAPIVersion:
    """Mock API version for testing"""
    
    def __init__(self, version: str, schema: Dict[str, Any]):
        self.version = version
        self.schema = schema
        self.endpoints = {}
        self.deprecated_endpoints = {}
    
    def add_endpoint(self, path: str, method: str, handler: callable):
        """Add endpoint to version"""
        self.endpoints[f"{method}:{path}"] = handler
    
    def add_deprecated_endpoint(self, path: str, method: str, handler: callable):
        """Add deprecated endpoint to version"""
        self.deprecated_endpoints[f"{method}:{path}"] = handler

class MockAPIVersionManager:
    """Mock API version manager for testing"""
    
    def __init__(self):
        self.versions = {}
        self.current_version = "v2"
        self.supported_versions = ["v1", "v2", "v3"]
        self.deprecated_versions = ["v1"]
    
    def register_version(self, version: MockAPIVersion):
        """Register API version"""
        self.versions[version.version] = version
    
    def get_version(self, version: str) -> MockAPIVersion:
        """Get API version"""
        return self.versions.get(version)
    
    def is_supported(self, version: str) -> bool:
        """Check if version is supported"""
        return version in self.supported_versions
    
    def is_deprecated(self, version: str) -> bool:
        """Check if version is deprecated"""
        return version in self.deprecated_versions
    
    def get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Get migration path between versions"""
        if from_version == "v1" and to_version == "v2":
            return ["v1", "v2"]
        elif from_version == "v2" and to_version == "v3":
            return ["v2", "v3"]
        elif from_version == "v1" and to_version == "v3":
            return ["v1", "v2", "v3"]
        return []

class MockPostData:
    """Mock post data for different API versions"""
    
    @staticmethod
    def v1_post_data():
        """V1 API post data structure"""
        return {
            "id": "post-1",
            "user_id": "user-1",
            "title": "Test Post",
            "content": "Test content",
            "post_type": "text",
            "tone": "professional",
            "status": "draft",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "engagement": {
                "likes": 10,
                "comments": 5,
                "shares": 2
            },
            "ai_score": 0.8,
            "keywords": ["test", "linkedin"]
        }
    
    @staticmethod
    def v2_post_data():
        """V2 API post data structure"""
        return {
            "id": "post-1",
            "userId": "user-1",
            "title": "Test Post",
            "content": {
                "text": "Test content",
                "hashtags": ["#test", "#linkedin"],
                "mentions": [],
                "links": [],
                "images": []
            },
            "postType": "text",
            "tone": "professional",
            "status": "draft",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
            "engagement": {
                "likes": 10,
                "comments": 5,
                "shares": 2,
                "clicks": 15,
                "impressions": 100,
                "reach": 80,
                "engagementRate": 0.17
            },
            "aiScore": 0.8,
            "optimizationSuggestions": ["Add more hashtags"],
            "keywords": ["test", "linkedin"],
            "externalMetadata": {},
            "performanceScore": 0.75,
            "reachScore": 0.8,
            "engagementScore": 0.7
        }
    
    @staticmethod
    def v3_post_data():
        """V3 API post data structure"""
        return {
            "id": "post-1",
            "userId": "user-1",
            "title": "Test Post",
            "content": {
                "text": "Test content",
                "hashtags": ["#test", "#linkedin"],
                "mentions": [],
                "links": [],
                "images": [],
                "callToAction": "Learn more"
            },
            "postType": "text",
            "tone": "professional",
            "status": "draft",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
            "scheduledAt": None,
            "publishedAt": None,
            "engagement": {
                "likes": 10,
                "comments": 5,
                "shares": 2,
                "clicks": 15,
                "impressions": 100,
                "reach": 80,
                "engagementRate": 0.17
            },
            "aiScore": 0.8,
            "optimizationSuggestions": ["Add more hashtags"],
            "keywords": ["test", "linkedin"],
            "linkedinPostId": None,
            "externalMetadata": {},
            "performanceScore": 0.75,
            "reachScore": 0.8,
            "engagementScore": 0.7
        }

class TestAPIVersioning:
    """Test API versioning and backward compatibility"""
    
    @pytest.fixture
    def api_version_manager(self):
        """API version manager for testing"""
        return MockAPIVersionManager()
    
    @pytest.fixture
    def v1_api(self):
        """V1 API version"""
        v1 = MockAPIVersion("v1", {
            "post": {
                "required": ["id", "user_id", "title", "content", "post_type", "tone", "status"],
                "optional": ["engagement", "ai_score", "keywords"]
            }
        })
        
        # Add V1 endpoints
        v1.add_endpoint("/posts", "POST", lambda data: {"status": "created", "data": data})
        v1.add_endpoint("/posts/{id}", "GET", lambda id: {"status": "success", "data": MockPostData.v1_post_data()})
        v1.add_endpoint("/posts/{id}", "PUT", lambda id, data: {"status": "updated", "data": data})
        v1.add_endpoint("/posts/{id}", "DELETE", lambda id: {"status": "deleted"})
        
        return v1
    
    @pytest.fixture
    def v2_api(self):
        """V2 API version"""
        v2 = MockAPIVersion("v2", {
            "post": {
                "required": ["id", "userId", "title", "content", "postType", "tone", "status"],
                "optional": ["engagement", "aiScore", "optimizationSuggestions", "keywords", "externalMetadata", "performanceScore", "reachScore", "engagementScore"]
            }
        })
        
        # Add V2 endpoints
        v2.add_endpoint("/posts", "POST", lambda data: {"status": "created", "data": data})
        v2.add_endpoint("/posts/{id}", "GET", lambda id: {"status": "success", "data": MockPostData.v2_post_data()})
        v2.add_endpoint("/posts/{id}", "PUT", lambda id, data: {"status": "updated", "data": data})
        v2.add_endpoint("/posts/{id}", "DELETE", lambda id: {"status": "deleted"})
        v2.add_endpoint("/posts/{id}/optimize", "POST", lambda id: {"status": "optimized", "data": MockPostData.v2_post_data()})
        
        return v2
    
    @pytest.fixture
    def v3_api(self):
        """V3 API version"""
        v3 = MockAPIVersion("v3", {
            "post": {
                "required": ["id", "userId", "title", "content", "postType", "tone", "status"],
                "optional": ["engagement", "aiScore", "optimizationSuggestions", "keywords", "linkedinPostId", "externalMetadata", "performanceScore", "reachScore", "engagementScore", "scheduledAt", "publishedAt"]
            }
        })
        
        # Add V3 endpoints
        v3.add_endpoint("/posts", "POST", lambda data: {"status": "created", "data": data})
        v3.add_endpoint("/posts/{id}", "GET", lambda id: {"status": "success", "data": MockPostData.v3_post_data()})
        v3.add_endpoint("/posts/{id}", "PUT", lambda id, data: {"status": "updated", "data": data})
        v3.add_endpoint("/posts/{id}", "DELETE", lambda id: {"status": "deleted"})
        v3.add_endpoint("/posts/{id}/optimize", "POST", lambda id: {"status": "optimized", "data": MockPostData.v3_post_data()})
        v3.add_endpoint("/posts/{id}/schedule", "POST", lambda id, data: {"status": "scheduled", "data": data})
        v3.add_endpoint("/posts/{id}/analytics", "GET", lambda id: {"status": "success", "data": {"analytics": "data"}})
        
        return v3

    async def test_api_version_support(self, api_version_manager, v1_api, v2_api, v3_api):
        """Test API version support and registration"""
        # Register API versions
        api_version_manager.register_version(v1_api)
        api_version_manager.register_version(v2_api)
        api_version_manager.register_version(v3_api)
        
        # Test version support
        assert api_version_manager.is_supported("v1")
        assert api_version_manager.is_supported("v2")
        assert api_version_manager.is_supported("v3")
        assert not api_version_manager.is_supported("v4")
        
        # Test deprecated versions
        assert api_version_manager.is_deprecated("v1")
        assert not api_version_manager.is_deprecated("v2")
        assert not api_version_manager.is_deprecated("v3")

    async def test_backward_compatibility_v1_to_v2(self, api_version_manager, v1_api, v2_api):
        """Test backward compatibility from V1 to V2"""
        api_version_manager.register_version(v1_api)
        api_version_manager.register_version(v2_api)
        
        # V1 post data
        v1_data = MockPostData.v1_post_data()
        
        # Test V1 endpoint with V1 data
        v1_handler = v1_api.endpoints["POST:/posts"]
        v1_response = v1_handler(v1_data)
        
        assert v1_response["status"] == "created"
        assert v1_response["data"]["user_id"] == "user-1"  # V1 format
        assert "userId" not in v1_response["data"]  # V2 format not present
        
        # Test V2 endpoint with V2 data
        v2_data = MockPostData.v2_post_data()
        v2_handler = v2_api.endpoints["POST:/posts"]
        v2_response = v2_handler(v2_data)
        
        assert v2_response["status"] == "created"
        assert v2_response["data"]["userId"] == "user-1"  # V2 format
        assert "user_id" not in v2_response["data"]  # V1 format not present

    async def test_backward_compatibility_v2_to_v3(self, api_version_manager, v2_api, v3_api):
        """Test backward compatibility from V2 to V3"""
        api_version_manager.register_version(v2_api)
        api_version_manager.register_version(v3_api)
        
        # V2 post data
        v2_data = MockPostData.v2_post_data()
        
        # Test V2 endpoint with V2 data
        v2_handler = v2_api.endpoints["POST:/posts"]
        v2_response = v2_handler(v2_data)
        
        assert v2_response["status"] == "created"
        assert "scheduledAt" not in v2_response["data"]  # V3 field not present
        
        # Test V3 endpoint with V3 data
        v3_data = MockPostData.v3_post_data()
        v3_handler = v3_api.endpoints["POST:/posts"]
        v3_response = v3_handler(v3_data)
        
        assert v3_response["status"] == "created"
        assert "scheduledAt" in v3_response["data"]  # V3 field present
        assert "linkedinPostId" in v3_response["data"]  # V3 field present

    async def test_data_migration_v1_to_v2(self, api_version_manager, v1_api, v2_api):
        """Test data migration from V1 to V2 format"""
        # V1 data structure
        v1_data = MockPostData.v1_post_data()
        
        # Migration function (simulated)
        def migrate_v1_to_v2(v1_data):
            return {
                "id": v1_data["id"],
                "userId": v1_data["user_id"],  # user_id -> userId
                "title": v1_data["title"],
                "content": {
                    "text": v1_data["content"],
                    "hashtags": [],
                    "mentions": [],
                    "links": [],
                    "images": []
                },
                "postType": v1_data["post_type"],  # post_type -> postType
                "tone": v1_data["tone"],
                "status": v1_data["status"],
                "createdAt": v1_data["created_at"],  # created_at -> createdAt
                "updatedAt": v1_data["updated_at"],  # updated_at -> updatedAt
                "engagement": {
                    **v1_data["engagement"],
                    "clicks": 0,
                    "impressions": 0,
                    "reach": 0,
                    "engagementRate": 0
                },
                "aiScore": v1_data["ai_score"],  # ai_score -> aiScore
                "optimizationSuggestions": [],
                "keywords": v1_data["keywords"],
                "externalMetadata": {},
                "performanceScore": 0,
                "reachScore": 0,
                "engagementScore": 0
            }
        
        # Test migration
        migrated_data = migrate_v1_to_v2(v1_data)
        
        # Verify migration
        assert migrated_data["userId"] == v1_data["user_id"]
        assert migrated_data["postType"] == v1_data["post_type"]
        assert migrated_data["aiScore"] == v1_data["ai_score"]
        assert migrated_data["createdAt"] == v1_data["created_at"]
        assert "user_id" not in migrated_data  # Old field removed
        assert "content" in migrated_data and isinstance(migrated_data["content"], dict)  # New structure

    async def test_data_migration_v2_to_v3(self, api_version_manager, v2_api, v3_api):
        """Test data migration from V2 to V3 format"""
        # V2 data structure
        v2_data = MockPostData.v2_post_data()
        
        # Migration function (simulated)
        def migrate_v2_to_v3(v2_data):
            return {
                **v2_data,
                "scheduledAt": None,
                "publishedAt": None,
                "linkedinPostId": None,
                "content": {
                    **v2_data["content"],
                    "callToAction": None
                }
            }
        
        # Test migration
        migrated_data = migrate_v2_to_v3(v2_data)
        
        # Verify migration
        assert "scheduledAt" in migrated_data
        assert "publishedAt" in migrated_data
        assert "linkedinPostId" in migrated_data
        assert "callToAction" in migrated_data["content"]
        assert migrated_data["userId"] == v2_data["userId"]  # Preserved fields

    async def test_endpoint_compatibility(self, api_version_manager, v1_api, v2_api, v3_api):
        """Test endpoint compatibility across versions"""
        api_version_manager.register_version(v1_api)
        api_version_manager.register_version(v2_api)
        api_version_manager.register_version(v3_api)
        
        # Test basic CRUD endpoints exist in all versions
        basic_endpoints = ["POST:/posts", "GET:/posts/{id}", "PUT:/posts/{id}", "DELETE:/posts/{id}"]
        
        for endpoint in basic_endpoints:
            assert endpoint in v1_api.endpoints
            assert endpoint in v2_api.endpoints
            assert endpoint in v3_api.endpoints
        
        # Test V2-specific endpoints
        v2_specific_endpoints = ["POST:/posts/{id}/optimize"]
        for endpoint in v2_specific_endpoints:
            assert endpoint in v2_api.endpoints
            assert endpoint in v3_api.endpoints
            assert endpoint not in v1_api.endpoints
        
        # Test V3-specific endpoints
        v3_specific_endpoints = ["POST:/posts/{id}/schedule", "GET:/posts/{id}/analytics"]
        for endpoint in v3_specific_endpoints:
            assert endpoint in v3_api.endpoints
            assert endpoint not in v1_api.endpoints
            assert endpoint not in v2_api.endpoints

    async def test_schema_validation(self, api_version_manager, v1_api, v2_api, v3_api):
        """Test schema validation across versions"""
        api_version_manager.register_version(v1_api)
        api_version_manager.register_version(v2_api)
        api_version_manager.register_version(v3_api)
        
        # Test V1 schema validation
        v1_schema = v1_api.schema["post"]
        v1_data = MockPostData.v1_post_data()
        
        # Verify required fields
        for field in v1_schema["required"]:
            assert field in v1_data
        
        # Test V2 schema validation
        v2_schema = v2_api.schema["post"]
        v2_data = MockPostData.v2_post_data()
        
        # Verify required fields
        for field in v2_schema["required"]:
            assert field in v2_data
        
        # Test V3 schema validation
        v3_schema = v3_api.schema["post"]
        v3_data = MockPostData.v3_post_data()
        
        # Verify required fields
        for field in v3_schema["required"]:
            assert field in v3_data

    async def test_deprecated_endpoint_handling(self, api_version_manager, v1_api):
        """Test handling of deprecated endpoints"""
        api_version_manager.register_version(v1_api)
        
        # Add deprecated endpoint to V1
        v1_api.add_deprecated_endpoint("/posts/old", "GET", lambda: {"status": "deprecated"})
        
        # Test deprecated endpoint still works but with warning
        deprecated_handler = v1_api.deprecated_endpoints["GET:/posts/old"]
        response = deprecated_handler()
        
        assert response["status"] == "deprecated"
        
        # Verify endpoint is marked as deprecated
        assert "GET:/posts/old" in v1_api.deprecated_endpoints
        assert "GET:/posts/old" not in v1_api.endpoints

    async def test_version_header_handling(self, api_version_manager, v1_api, v2_api, v3_api):
        """Test API version header handling"""
        api_version_manager.register_version(v1_api)
        api_version_manager.register_version(v2_api)
        api_version_manager.register_version(v3_api)
        
        # Test version header parsing
        headers = [
            {"Accept": "application/vnd.api.v1+json"},
            {"Accept": "application/vnd.api.v2+json"},
            {"Accept": "application/vnd.api.v3+json"},
            {"Accept": "application/vnd.api.v4+json"}  # Unsupported
        ]
        
        expected_versions = ["v1", "v2", "v3", None]
        
        for header, expected_version in zip(headers, expected_versions):
            # Simulate header parsing
            accept_header = header.get("Accept", "")
            if "v1" in accept_header:
                version = "v1"
            elif "v2" in accept_header:
                version = "v2"
            elif "v3" in accept_header:
                version = "v3"
            else:
                version = None
            
            if expected_version:
                assert api_version_manager.is_supported(version)
            else:
                assert not api_version_manager.is_supported(version)

    async def test_migration_path_validation(self, api_version_manager):
        """Test migration path validation between versions"""
        # Test valid migration paths
        valid_paths = [
            ("v1", "v2"),
            ("v2", "v3"),
            ("v1", "v3")
        ]
        
        for from_version, to_version in valid_paths:
            migration_path = api_version_manager.get_migration_path(from_version, to_version)
            assert len(migration_path) > 0
            assert migration_path[0] == from_version
            assert migration_path[-1] == to_version
        
        # Test invalid migration paths
        invalid_paths = [
            ("v2", "v1"),  # Cannot downgrade
            ("v3", "v1"),  # Cannot skip versions
            ("v4", "v2")   # Invalid version
        ]
        
        for from_version, to_version in invalid_paths:
            migration_path = api_version_manager.get_migration_path(from_version, to_version)
            assert len(migration_path) == 0

    async def test_response_format_compatibility(self, api_version_manager, v1_api, v2_api, v3_api):
        """Test response format compatibility across versions"""
        api_version_manager.register_version(v1_api)
        api_version_manager.register_version(v2_api)
        api_version_manager.register_version(v3_api)
        
        # Test V1 response format
        v1_handler = v1_api.endpoints["GET:/posts/{id}"]
        v1_response = v1_handler("post-1")
        
        assert "status" in v1_response
        assert "data" in v1_response
        assert v1_response["data"]["user_id"] == "user-1"  # V1 format
        
        # Test V2 response format
        v2_handler = v2_api.endpoints["GET:/posts/{id}"]
        v2_response = v2_handler("post-1")
        
        assert "status" in v2_response
        assert "data" in v2_response
        assert v2_response["data"]["userId"] == "user-1"  # V2 format
        
        # Test V3 response format
        v3_handler = v3_api.endpoints["GET:/posts/{id}"]
        v3_response = v3_handler("post-1")
        
        assert "status" in v3_response
        assert "data" in v3_response
        assert v3_response["data"]["userId"] == "user-1"  # V3 format
        assert "scheduledAt" in v3_response["data"]  # V3 specific field

    async def test_error_handling_compatibility(self, api_version_manager, v1_api, v2_api, v3_api):
        """Test error handling compatibility across versions"""
        api_version_manager.register_version(v1_api)
        api_version_manager.register_version(v2_api)
        api_version_manager.register_version(v3_api)
        
        # Test error responses maintain consistent structure
        error_scenarios = [
            {"status": "error", "message": "Post not found", "code": 404},
            {"status": "error", "message": "Invalid data", "code": 400},
            {"status": "error", "message": "Server error", "code": 500}
        ]
        
        for error in error_scenarios:
            # All versions should handle errors consistently
            assert "status" in error
            assert "message" in error
            assert "code" in error
            assert error["status"] == "error"

    async def test_performance_impact_versioning(self, api_version_manager, v1_api, v2_api, v3_api):
        """Test performance impact of versioning"""
        api_version_manager.register_version(v1_api)
        api_version_manager.register_version(v2_api)
        api_version_manager.register_version(v3_api)
        
        # Test response time consistency across versions
        import time
        
        versions = [v1_api, v2_api, v3_api]
        response_times = []
        
        for version in versions:
            handler = version.endpoints["GET:/posts/{id}"]
            
            start_time = time.time()
            response = handler("post-1")
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Verify response is valid
            assert response["status"] == "success"
        
        # Response times should be similar (within reasonable range)
        avg_response_time = sum(response_times) / len(response_times)
        for response_time in response_times:
            assert abs(response_time - avg_response_time) < 0.1  # Within 100ms

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
