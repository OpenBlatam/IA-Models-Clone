"""Core models and entities for Social Media Identity Clone AI."""

from .models import (
    SocialProfile,
    IdentityProfile,
    ContentAnalysis,
    GeneratedContent,
    VideoContent,
    PostContent,
    CommentContent,
    Platform,
    ContentType,
)
from .interfaces import (
    IProfileExtractor,
    IIdentityAnalyzer,
    IContentGenerator,
    IConnector,
    IStorageService,
    ICacheManager,
)
from .dependency_injection import (
    ServiceContainer,
    get_container,
    register_service,
    get_service,
    resolve_service,
)

__all__ = [
    # Models
    "SocialProfile",
    "IdentityProfile",
    "ContentAnalysis",
    "GeneratedContent",
    "VideoContent",
    "PostContent",
    "CommentContent",
    "Platform",
    "ContentType",
    # Interfaces
    "IProfileExtractor",
    "IIdentityAnalyzer",
    "IContentGenerator",
    "IConnector",
    "IStorageService",
    "ICacheManager",
    # Dependency Injection
    "ServiceContainer",
    "get_container",
    "register_service",
    "get_service",
    "resolve_service",
]




