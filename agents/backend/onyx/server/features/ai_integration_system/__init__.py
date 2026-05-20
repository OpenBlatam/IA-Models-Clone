"""
AI Integration System
====================

A comprehensive system for integrating AI-generated content with CMS, CRM, and marketing platforms.

This system provides:
- Multi-platform content distribution
- Automated workflow management
- Real-time integration monitoring
- Webhook handling for platform events
- Comprehensive error handling and retry mechanisms

Supported Platforms:
- Salesforce CRM
- Mailchimp Email Marketing
- WordPress CMS
- HubSpot CRM & Marketing
- Slack Communication
- Google Workspace
- Microsoft 365
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Comprehensive system for integrating AI-generated content with CMS, CRM, and marketing platforms"

# Try to import components with error handling
try:
    from .integration_engine import (
        AIIntegrationEngine,
        IntegrationRequest,
        IntegrationResult,
        ContentType,
        IntegrationStatus,
        PlatformConnector,
        integration_engine
    )
except ImportError:
    AIIntegrationEngine = None
    IntegrationRequest = None
    IntegrationResult = None
    ContentType = None
    IntegrationStatus = None
    PlatformConnector = None
    integration_engine = None

try:
    from .api_endpoints import router
except ImportError:
    router = None

try:
    from .config import (
        settings,
        get_platform_config,
        is_platform_enabled,
        get_enabled_platforms
    )
except ImportError:
    settings = None
    get_platform_config = None
    is_platform_enabled = None
    get_enabled_platforms = None

__all__ = [
    # Core engine
    "AIIntegrationEngine",
    "IntegrationRequest", 
    "IntegrationResult",
    "ContentType",
    "IntegrationStatus",
    "PlatformConnector",
    "integration_engine",
    
    # API
    "router",
    
    # Configuration
    "settings",
    "get_platform_config",
    "is_platform_enabled", 
    "get_enabled_platforms"
]





















