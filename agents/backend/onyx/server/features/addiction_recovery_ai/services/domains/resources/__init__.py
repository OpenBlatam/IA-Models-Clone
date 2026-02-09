"""
Resources domain services
"""

from services.domains import register_service

try:
    from services.resource_library_service import ResourceLibraryService
    from services.advanced_support_groups_service import AdvancedSupportGroupsService
    
    def register_services():
        register_service("resources", "library", ResourceLibraryService)
        register_service("resources", "support_groups", AdvancedSupportGroupsService)
except ImportError:
    pass



