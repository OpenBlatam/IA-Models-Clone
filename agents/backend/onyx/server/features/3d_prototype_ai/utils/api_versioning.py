"""
API Versioning - Sistema de versionado de API
==============================================
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
    """Versiones de API"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    LATEST = "latest"


class APIVersioning:
    """Sistema de versionado de API"""
    
    def __init__(self):
        self.versions: Dict[str, Dict[str, Any]] = {
            "v1": {
                "version": "1.0.0",
                "release_date": "2024-01-01",
                "status": "stable",
                "deprecated": False,
                "endpoints": []
            },
            "v2": {
                "version": "2.0.0",
                "release_date": datetime.now().strftime("%Y-%m-%d"),
                "status": "current",
                "deprecated": False,
                "endpoints": []
            }
        }
        self.default_version = "v2"
        self.deprecation_warnings: Dict[str, datetime] = {}
    
    def register_endpoint(self, version: str, endpoint: str, method: str = "GET"):
        """Registra un endpoint en una versión"""
        if version not in self.versions:
            self.versions[version] = {
                "version": version,
                "release_date": datetime.now().strftime("%Y-%m-%d"),
                "status": "beta",
                "deprecated": False,
                "endpoints": []
            }
        
        endpoint_info = {
            "path": endpoint,
            "method": method,
            "registered_at": datetime.now().isoformat()
        }
        
        self.versions[version]["endpoints"].append(endpoint_info)
    
    def deprecate_version(self, version: str, deprecation_date: Optional[datetime] = None):
        """Marca una versión como deprecada"""
        if version in self.versions:
            self.versions[version]["deprecated"] = True
            self.versions[version]["status"] = "deprecated"
            
            if deprecation_date:
                self.deprecation_warnings[version] = deprecation_date
                self.versions[version]["deprecation_date"] = deprecation_date.isoformat()
            
            logger.warning(f"Versión {version} marcada como deprecada")
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de una versión"""
        return self.versions.get(version)
    
    def get_all_versions(self) -> Dict[str, Any]:
        """Obtiene todas las versiones"""
        return {
            "default": self.default_version,
            "versions": self.versions,
            "deprecation_warnings": {
                v: d.isoformat() for v, d in self.deprecation_warnings.items()
            }
        }
    
    def check_deprecation(self, version: str) -> Optional[Dict[str, Any]]:
        """Verifica si una versión está deprecada"""
        version_info = self.versions.get(version)
        if not version_info:
            return None
        
        if version_info.get("deprecated"):
            return {
                "deprecated": True,
                "version": version,
                "deprecation_date": version_info.get("deprecation_date"),
                "message": f"La versión {version} está deprecada. Por favor migra a {self.default_version}",
                "migration_guide": f"/docs/migration/{version}-to-{self.default_version}"
            }
        
        return None
    
    def get_version_header(self, version: str) -> Dict[str, str]:
        """Obtiene headers para respuesta de versión"""
        headers = {
            "API-Version": version,
            "API-Default-Version": self.default_version
        }
        
        deprecation = self.check_deprecation(version)
        if deprecation:
            headers["API-Deprecated"] = "true"
            headers["Deprecation"] = f"version={version}"
            headers["Sunset"] = deprecation.get("deprecation_date", "")
        
        return headers




