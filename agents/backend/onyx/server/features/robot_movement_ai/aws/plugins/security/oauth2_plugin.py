"""
OAuth2 Security Plugin
======================
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Depends
from aws.core.interfaces import SecurityPlugin

logger = logging.getLogger(__name__)


class OAuth2SecurityPlugin(SecurityPlugin):
    """OAuth2/JWT security plugin."""
    
    def get_name(self) -> str:
        return "oauth2"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        security_config = config.get("security", {})
        return security_config.get("enable_oauth2", True)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup OAuth2 security."""
        try:
            from aws.security.oauth2_config import get_current_active_user
            
            # Add protected endpoints
            @app.get("/api/v1/protected/status")
            async def protected_status(current_user=Depends(get_current_active_user)):
                """Protected status endpoint."""
                return {
                    "status": "authenticated",
                    "user": current_user.username,
                    "scopes": current_user.scopes
                }
            
            logger.info("OAuth2 security enabled")
            
        except ImportError:
            logger.warning("OAuth2 dependencies not installed. Security disabled.")
        except Exception as e:
            logger.error(f"Failed to setup OAuth2: {e}")
        
        return app















