"""
BUL API Versioning System
========================

Advanced API versioning with backward compatibility and migration support.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from enum import Enum
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import json
import asyncio
from dataclasses import dataclass

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class APIVersion(str, Enum):
    """Supported API versions"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    LATEST = "latest"

class VersionStatus(str, Enum):
    """Version status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    BETA = "beta"
    STABLE = "stable"

@dataclass
class VersionInfo:
    """Version information"""
    version: str
    status: VersionStatus
    release_date: datetime
    sunset_date: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    changelog: List[str] = None
    breaking_changes: List[str] = None
    migration_guide: Optional[str] = None

class VersionManager:
    """API version management system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        
        # Version registry
        self.versions = {
            APIVersion.V1: VersionInfo(
                version="v1",
                status=VersionStatus.STABLE,
                release_date=datetime(2024, 1, 1),
                changelog=[
                    "Initial API release",
                    "Basic document generation",
                    "Simple agent selection"
                ],
                breaking_changes=[],
                migration_guide="https://docs.bul-system.com/migration/v1"
            ),
            APIVersion.V2: VersionInfo(
                version="v2",
                status=VersionStatus.STABLE,
                release_date=datetime(2024, 6, 1),
                changelog=[
                    "Enhanced agent selection algorithm",
                    "Improved caching system",
                    "Better error handling",
                    "Performance optimizations"
                ],
                breaking_changes=[
                    "Response format changes",
                    "New required fields in requests"
                ],
                migration_guide="https://docs.bul-system.com/migration/v2"
            ),
            APIVersion.V3: VersionInfo(
                version="v3",
                status=VersionStatus.BETA,
                release_date=datetime(2024, 12, 1),
                changelog=[
                    "Advanced analytics integration",
                    "Real-time streaming responses",
                    "Enhanced security features",
                    "Multi-tenant support"
                ],
                breaking_changes=[
                    "Authentication changes",
                    "New request/response schemas",
                    "Deprecated endpoints"
                ],
                migration_guide="https://docs.bul-system.com/migration/v3"
            }
        }
        
        # Set latest version
        self.latest_version = APIVersion.V3
    
    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get version information"""
        if version == "latest":
            version = self.latest_version.value
        
        return self.versions.get(APIVersion(version))
    
    def is_version_supported(self, version: str) -> bool:
        """Check if version is supported"""
        if version == "latest":
            return True
        
        try:
            version_info = self.get_version_info(version)
            return version_info is not None and version_info.status != VersionStatus.SUNSET
        except ValueError:
            return False
    
    def get_supported_versions(self) -> List[str]:
        """Get list of supported versions"""
        supported = []
        for version, info in self.versions.items():
            if info.status != VersionStatus.SUNSET:
                supported.append(version.value)
        supported.append("latest")
        return supported
    
    def get_version_headers(self, version: str) -> Dict[str, str]:
        """Get version-specific headers"""
        version_info = self.get_version_info(version)
        if not version_info:
            return {}
        
        headers = {
            "API-Version": version,
            "API-Status": version_info.status.value,
            "API-Latest": self.latest_version.value
        }
        
        if version_info.status == VersionStatus.DEPRECATED:
            headers["API-Deprecation-Date"] = version_info.deprecation_date.isoformat()
            headers["API-Sunset-Date"] = version_info.sunset_date.isoformat()
        
        return headers
    
    def validate_version_compatibility(self, version: str, endpoint: str) -> bool:
        """Validate version compatibility with endpoint"""
        version_info = self.get_version_info(version)
        if not version_info:
            return False
        
        # Check if endpoint is available in this version
        endpoint_versions = {
            "/generate": [APIVersion.V1, APIVersion.V2, APIVersion.V3],
            "/agents": [APIVersion.V1, APIVersion.V2, APIVersion.V3],
            "/stats": [APIVersion.V1, APIVersion.V2, APIVersion.V3],
            "/analytics": [APIVersion.V2, APIVersion.V3],
            "/stream": [APIVersion.V3],
            "/webhooks": [APIVersion.V3]
        }
        
        supported_versions = endpoint_versions.get(endpoint, [])
        return APIVersion(version) in supported_versions

# Global version manager
_version_manager: Optional[VersionManager] = None

def get_version_manager() -> VersionManager:
    """Get the global version manager"""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager

# Version dependency
def get_api_version(
    x_api_version: Optional[str] = Header(None, alias="X-API-Version"),
    accept: Optional[str] = Header(None, alias="Accept")
) -> str:
    """Extract API version from headers"""
    version_manager = get_version_manager()
    
    # Try to get version from X-API-Version header
    if x_api_version:
        if version_manager.is_version_supported(x_api_version):
            return x_api_version
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported API version: {x_api_version}. Supported versions: {version_manager.get_supported_versions()}"
            )
    
    # Try to extract from Accept header
    if accept and "version=" in accept:
        try:
            version = accept.split("version=")[1].split(";")[0].strip()
            if version_manager.is_version_supported(version):
                return version
        except (IndexError, ValueError):
            pass
    
    # Default to latest version
    return version_manager.latest_version.value

# Version-specific request/response models
class DocumentRequestV1(BaseModel):
    """Document request model for API v1"""
    query: str = Field(..., min_length=10, max_length=1000)
    business_area: str = Field(..., description="Business area")
    document_type: str = Field(..., description="Document type")
    language: str = Field("es", description="Language")
    format: str = Field("markdown", description="Output format")

class DocumentRequestV2(BaseModel):
    """Document request model for API v2"""
    query: str = Field(..., min_length=10, max_length=2000)
    business_area: BusinessArea = Field(..., description="Business area")
    document_type: DocumentType = Field(..., description="Document type")
    language: str = Field("es", description="Language")
    format: str = Field("markdown", description="Output format")
    context: Optional[str] = Field(None, description="Additional context")
    requirements: Optional[List[str]] = Field(None, description="Specific requirements")

class DocumentRequestV3(BaseModel):
    """Document request model for API v3"""
    query: str = Field(..., min_length=10, max_length=5000)
    business_area: BusinessArea = Field(..., description="Business area")
    document_type: DocumentType = Field(..., description="Document type")
    language: str = Field("es", description="Language")
    format: str = Field("markdown", description="Output format")
    context: Optional[str] = Field(None, description="Additional context")
    requirements: Optional[List[str]] = Field(None, description="Specific requirements")
    streaming: bool = Field(False, description="Enable streaming response")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for async processing")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DocumentResponseV1(BaseModel):
    """Document response model for API v1"""
    content: str
    business_area: str
    document_type: str
    success: bool
    document_id: str
    generated_at: str
    processing_time: float

class DocumentResponseV2(BaseModel):
    """Document response model for API v2"""
    content: str
    business_area: BusinessArea
    document_type: DocumentType
    success: bool
    document_id: str
    generated_at: datetime
    processing_time: float
    confidence_score: float
    agent_used: str
    cache_hit: bool

class DocumentResponseV3(BaseModel):
    """Document response model for API v3"""
    content: str
    business_area: BusinessArea
    document_type: DocumentType
    success: bool
    document_id: str
    generated_at: datetime
    processing_time: float
    confidence_score: float
    agent_used: str
    cache_hit: bool
    streaming_enabled: bool
    webhook_sent: bool
    metadata: Dict[str, Any]
    analytics: Dict[str, Any]

# Version-specific routers
def create_versioned_router(version: str) -> APIRouter:
    """Create version-specific router"""
    version_manager = get_version_manager()
    version_info = version_manager.get_version_info(version)
    
    if not version_info:
        raise ValueError(f"Unsupported version: {version}")
    
    router = APIRouter(
        prefix=f"/{version}",
        tags=[f"API {version.upper()}"],
        responses={
            400: {"description": "Bad Request - Unsupported version or invalid request"},
            404: {"description": "Not Found - Endpoint not available in this version"},
            410: {"description": "Gone - Version has been sunset"}
        }
    )
    
    # Add version-specific middleware
    @router.middleware("http")
    async def add_version_headers(request: Request, call_next):
        response = await call_next(request)
        
        # Add version headers
        headers = version_manager.get_version_headers(version)
        for key, value in headers.items():
            response.headers[key] = value
        
        # Add deprecation warning if applicable
        if version_info.status == VersionStatus.DEPRECATED:
            response.headers["Warning"] = f"299 - \"API version {version} is deprecated. Please migrate to {version_manager.latest_version.value}\""
        
        return response
    
    return router

# Version compatibility layer
class VersionCompatibility:
    """Handle version compatibility and migration"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def migrate_request(self, request_data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate request data between versions"""
        try:
            if from_version == to_version:
                return request_data
            
            # Migration rules
            migration_rules = {
                "v1_to_v2": self._migrate_v1_to_v2,
                "v2_to_v3": self._migrate_v2_to_v3,
                "v1_to_v3": self._migrate_v1_to_v3
            }
            
            migration_key = f"{from_version}_to_{to_version}"
            if migration_key in migration_rules:
                return migration_rules[migration_key](request_data)
            
            return request_data
        
        except Exception as e:
            self.logger.error(f"Error migrating request from {from_version} to {to_version}: {e}")
            return request_data
    
    def _migrate_v1_to_v2(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1 request to v2 format"""
        # Convert string enums to proper enum values
        if "business_area" in request_data:
            try:
                request_data["business_area"] = BusinessArea(request_data["business_area"])
            except ValueError:
                pass
        
        if "document_type" in request_data:
            try:
                request_data["document_type"] = DocumentType(request_data["document_type"])
            except ValueError:
                pass
        
        return request_data
    
    def _migrate_v2_to_v3(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v2 request to v3 format"""
        # Add new v3 fields with defaults
        request_data.setdefault("streaming", False)
        request_data.setdefault("webhook_url", None)
        request_data.setdefault("metadata", {})
        
        return request_data
    
    def _migrate_v1_to_v3(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1 request to v3 format"""
        # First migrate to v2, then to v3
        v2_data = self._migrate_v1_to_v2(request_data)
        return self._migrate_v2_to_v3(v2_data)
    
    def migrate_response(self, response_data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate response data between versions"""
        try:
            if from_version == to_version:
                return response_data
            
            # Response migration rules
            migration_rules = {
                "v3_to_v2": self._migrate_response_v3_to_v2,
                "v2_to_v1": self._migrate_response_v2_to_v1,
                "v3_to_v1": self._migrate_response_v3_to_v1
            }
            
            migration_key = f"{from_version}_to_{to_version}"
            if migration_key in migration_rules:
                return migration_rules[migration_key](response_data)
            
            return response_data
        
        except Exception as e:
            self.logger.error(f"Error migrating response from {from_version} to {to_version}: {e}")
            return response_data
    
    def _migrate_response_v3_to_v2(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v3 response to v2 format"""
        # Remove v3-specific fields
        response_data.pop("streaming_enabled", None)
        response_data.pop("webhook_sent", None)
        response_data.pop("metadata", None)
        response_data.pop("analytics", None)
        
        return response_data
    
    def _migrate_response_v2_to_v1(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v2 response to v1 format"""
        # Convert datetime to string
        if "generated_at" in response_data and isinstance(response_data["generated_at"], datetime):
            response_data["generated_at"] = response_data["generated_at"].isoformat()
        
        # Remove v2-specific fields
        response_data.pop("confidence_score", None)
        response_data.pop("agent_used", None)
        response_data.pop("cache_hit", None)
        
        # Convert enums to strings
        if "business_area" in response_data:
            response_data["business_area"] = str(response_data["business_area"])
        
        if "document_type" in response_data:
            response_data["document_type"] = str(response_data["document_type"])
        
        return response_data
    
    def _migrate_response_v3_to_v1(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v3 response to v1 format"""
        # First migrate to v2, then to v1
        v2_data = self._migrate_response_v3_to_v2(response_data)
        return self._migrate_response_v2_to_v1(v2_data)

# Global compatibility instance
_compatibility: Optional[VersionCompatibility] = None

def get_compatibility() -> VersionCompatibility:
    """Get the global compatibility instance"""
    global _compatibility
    if _compatibility is None:
        _compatibility = VersionCompatibility()
    return _compatibility

# Version info endpoints
version_router = APIRouter(prefix="/versions", tags=["API Versioning"])

@version_router.get("/")
async def get_version_info():
    """Get information about all API versions"""
    version_manager = get_version_manager()
    
    versions_info = {}
    for version, info in version_manager.versions.items():
        versions_info[version.value] = {
            "version": info.version,
            "status": info.status.value,
            "release_date": info.release_date.isoformat(),
            "sunset_date": info.sunset_date.isoformat() if info.sunset_date else None,
            "deprecation_date": info.deprecation_date.isoformat() if info.deprecation_date else None,
            "changelog": info.changelog,
            "breaking_changes": info.breaking_changes,
            "migration_guide": info.migration_guide
        }
    
    return {
        "supported_versions": version_manager.get_supported_versions(),
        "latest_version": version_manager.latest_version.value,
        "versions": versions_info
    }

@version_router.get("/{version}")
async def get_specific_version_info(version: str):
    """Get information about a specific API version"""
    version_manager = get_version_manager()
    
    if not version_manager.is_version_supported(version):
        raise HTTPException(
            status_code=404,
            detail=f"Version {version} not found or not supported"
        )
    
    version_info = version_manager.get_version_info(version)
    if not version_info:
        raise HTTPException(status_code=404, detail="Version information not found")
    
    return {
        "version": version_info.version,
        "status": version_info.status.value,
        "release_date": version_info.release_date.isoformat(),
        "sunset_date": version_info.sunset_date.isoformat() if version_info.sunset_date else None,
        "deprecation_date": version_info.deprecation_date.isoformat() if version_info.deprecation_date else None,
        "changelog": version_info.changelog,
        "breaking_changes": version_info.breaking_changes,
        "migration_guide": version_info.migration_guide
    }

@version_router.get("/{version}/compatibility/{endpoint}")
async def check_endpoint_compatibility(version: str, endpoint: str):
    """Check if an endpoint is compatible with a specific version"""
    version_manager = get_version_manager()
    
    if not version_manager.is_version_supported(version):
        raise HTTPException(
            status_code=404,
            detail=f"Version {version} not found or not supported"
        )
    
    is_compatible = version_manager.validate_version_compatibility(version, endpoint)
    
    return {
        "version": version,
        "endpoint": endpoint,
        "compatible": is_compatible,
        "status": version_manager.get_version_info(version).status.value if is_compatible else "not_supported"
    }


