"""
Recovery API Module
Feature-specific API module for recovery endpoints
"""

from typing import List
from modules.api_module import APIModule

logger = __import__("logging").getLogger(__name__)


class RecoveryAPIModule(APIModule):
    """Recovery API feature module"""
    
    def __init__(self):
        super().__init__()
        self._name = "recovery_api"
    
    @property
    def name(self) -> str:
        """Module name"""
        return self._name
    
    def get_dependencies(self) -> List[str]:
        """Recovery API depends on base API and other modules"""
        return ["api", "storage", "cache", "security", "observability"]
    
    def _register_routes(self) -> None:
        """Register recovery API routes"""
        try:
            # Import recovery router
            # Use canonical API router instead of deprecated recovery_api
            from api.recovery_api_refactored import router as recovery_router
            self.include_router(recovery_router, prefix="/recovery", tags=["Recovery"])
            logger.info("Recovery API routes registered")
        except ImportError as e:
            logger.warning(f"Failed to import recovery router: {str(e)}")















