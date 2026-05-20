"""
Social domain services
"""

from services.domains import register_service

try:
    from services.social_integration_service import SocialIntegrationService
    from services.social_relationships_service import SocialRelationshipsService
    from services.social_media_analysis_service import SocialMediaAnalysisService
    from services.advanced_social_network_analysis_service import AdvancedSocialNetworkAnalysisService
    from services.advanced_social_support_service import AdvancedSocialSupportService
    
    def register_services():
        register_service("social", "integration", SocialIntegrationService)
        register_service("social", "relationships", SocialRelationshipsService)
        register_service("social", "media_analysis", SocialMediaAnalysisService)
        register_service("social", "network_analysis", AdvancedSocialNetworkAnalysisService)
        register_service("social", "support", AdvancedSocialSupportService)
except ImportError:
    pass



