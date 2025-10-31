"""
AI Integration System
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

Author: AI Integration Team
Version: 1.0.0
"""

from .integration_engine import (
    AIIntegrationEngine,
    IntegrationRequest,
    IntegrationResult,
    ContentType,
    IntegrationStatus,
    PlatformConnector,
    integration_engine
)

from .api_endpoints import router

from .config import (
    settings,
    get_platform_config,
    is_platform_enabled,
    get_enabled_platforms
)

__version__ = "1.0.0"
__author__ = "AI Integration Team"
__email__ = "team@aiintegration.com"

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



























