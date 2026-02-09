"""
Deprecation Manager
==================

API deprecation management.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeprecationNotice:
    """Deprecation notice."""
    endpoint: str
    version: str
    deprecation_date: datetime
    sunset_date: datetime
    alternative: Optional[str] = None
    migration_guide: Optional[str] = None


class DeprecationManager:
    """API deprecation manager."""
    
    def __init__(self):
        self._notices: Dict[str, DeprecationNotice] = {}  # endpoint -> notice
        self._warnings_sent: Dict[str, datetime] = {}
    
    def deprecate_endpoint(
        self,
        endpoint: str,
        version: str,
        sunset_days: int = 90,
        alternative: Optional[str] = None,
        migration_guide: Optional[str] = None
    ):
        """Deprecate endpoint."""
        deprecation_date = datetime.now()
        sunset_date = deprecation_date + timedelta(days=sunset_days)
        
        notice = DeprecationNotice(
            endpoint=endpoint,
            version=version,
            deprecation_date=deprecation_date,
            sunset_date=sunset_date,
            alternative=alternative,
            migration_guide=migration_guide
        )
        
        self._notices[endpoint] = notice
        logger.warning(f"Deprecated endpoint: {endpoint} (sunset: {sunset_date})")
    
    def get_notice(self, endpoint: str) -> Optional[DeprecationNotice]:
        """Get deprecation notice for endpoint."""
        return self._notices.get(endpoint)
    
    def is_deprecated(self, endpoint: str) -> bool:
        """Check if endpoint is deprecated."""
        return endpoint in self._notices
    
    def is_sunset(self, endpoint: str) -> bool:
        """Check if endpoint is sunset."""
        notice = self.get_notice(endpoint)
        if not notice:
            return False
        
        return datetime.now() >= notice.sunset_date
    
    def get_deprecation_headers(self, endpoint: str) -> Dict[str, str]:
        """Get deprecation headers for endpoint."""
        notice = self.get_notice(endpoint)
        if not notice:
            return {}
        
        headers = {
            "Deprecation": "true",
            "Sunset": notice.sunset_date.isoformat()
        }
        
        if notice.alternative:
            headers["Link"] = f'<{notice.alternative}>; rel="successor-version"'
        
        return headers
    
    def get_all_deprecated(self) -> List[DeprecationNotice]:
        """Get all deprecated endpoints."""
        return list(self._notices.values())
    
    def get_deprecation_stats(self) -> Dict[str, Any]:
        """Get deprecation statistics."""
        now = datetime.now()
        
        deprecated = [
            n for n in self._notices.values()
            if n.deprecation_date <= now < n.sunset_date
        ]
        
        sunset = [
            n for n in self._notices.values()
            if now >= n.sunset_date
        ]
        
        return {
            "total_deprecated": len(self._notices),
            "active_deprecations": len(deprecated),
            "sunset_endpoints": len(sunset)
        }















