"""
Versioning Service - Sistema de versionado de diseños
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from ..core.models import StoreDesign

logger = logging.getLogger(__name__)


class VersioningService:
    """Servicio para versionado de diseños"""
    
    def __init__(self):
        self.versions: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_version(
        self,
        original_design: StoreDesign,
        changes: Dict[str, Any],
        version_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear nueva versión de un diseño"""
        
        original_id = original_design.store_id
        
        # Obtener número de versión
        existing_versions = self.versions.get(original_id, [])
        version_number = len(existing_versions) + 1
        
        # Crear ID de versión
        version_id = f"{original_id}_v{version_number}"
        
        version = {
            "version_id": version_id,
            "original_store_id": original_id,
            "version_number": version_number,
            "created_at": datetime.now().isoformat(),
            "changes": changes,
            "notes": version_notes or f"Versión {version_number}",
            "status": "draft"  # "draft", "approved", "rejected"
        }
        
        if original_id not in self.versions:
            self.versions[original_id] = []
        
        self.versions[original_id].append(version)
        
        logger.info(f"Versión {version_number} creada para diseño {original_id}")
        return version
    
    def get_versions(self, store_id: str) -> List[Dict[str, Any]]:
        """Obtener todas las versiones de un diseño"""
        return self.versions.get(store_id, [])
    
    def get_latest_version(self, store_id: str) -> Optional[Dict[str, Any]]:
        """Obtener la última versión"""
        versions = self.get_versions(store_id)
        if not versions:
            return None
        return max(versions, key=lambda v: v["version_number"])
    
    def compare_versions(
        self,
        store_id: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """Comparar dos versiones"""
        versions = self.get_versions(store_id)
        
        v1 = next((v for v in versions if v["version_number"] == version1), None)
        v2 = next((v for v in versions if v["version_number"] == version2), None)
        
        if not v1 or not v2:
            raise ValueError("Una o ambas versiones no encontradas")
        
        return {
            "version1": v1,
            "version2": v2,
            "differences": self._find_differences(v1["changes"], v2["changes"]),
            "summary": f"Comparación entre versión {version1} y {version2}"
        }
    
    def _find_differences(
        self,
        changes1: Dict[str, Any],
        changes2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encontrar diferencias entre cambios"""
        differences = {}
        
        all_keys = set(changes1.keys()) | set(changes2.keys())
        
        for key in all_keys:
            val1 = changes1.get(key)
            val2 = changes2.get(key)
            
            if val1 != val2:
                differences[key] = {
                    "version1": val1,
                    "version2": val2
                }
        
        return differences
    
    def approve_version(
        self,
        store_id: str,
        version_number: int
    ) -> bool:
        """Aprobar una versión"""
        versions = self.get_versions(store_id)
        
        for version in versions:
            if version["version_number"] == version_number:
                version["status"] = "approved"
                version["approved_at"] = datetime.now().isoformat()
                return True
        
        return False
    
    def get_version_history(self, store_id: str) -> Dict[str, Any]:
        """Obtener historial completo de versiones"""
        versions = self.get_versions(store_id)
        
        return {
            "store_id": store_id,
            "total_versions": len(versions),
            "versions": versions,
            "latest_version": self.get_latest_version(store_id),
            "approved_versions": [v for v in versions if v["status"] == "approved"],
            "timeline": self._generate_timeline(versions)
        }
    
    def _generate_timeline(self, versions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generar línea de tiempo de versiones"""
        timeline = []
        
        for version in sorted(versions, key=lambda v: v["version_number"]):
            timeline.append({
                "version": version["version_number"],
                "date": version["created_at"],
                "status": version["status"],
                "notes": version.get("notes", "")
            })
        
        return timeline




