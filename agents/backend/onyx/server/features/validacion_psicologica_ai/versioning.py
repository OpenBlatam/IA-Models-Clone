"""
Sistema de Versionado para Validación Psicológica AI
=====================================================
Gestión de versiones de validaciones y migraciones
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
from enum import Enum
import structlog
import json

from .models import PsychologicalValidation, PsychologicalProfile, ValidationReport

logger = structlog.get_logger()


class VersionStatus(str, Enum):
    """Estado de versión"""
    CURRENT = "current"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ValidationVersion:
    """Versión de una validación"""
    
    def __init__(
        self,
        validation_id: UUID,
        version_number: int,
        validation_data: Dict[str, Any],
        created_at: datetime,
        changes: Optional[List[str]] = None
    ):
        self.validation_id = validation_id
        self.version_number = version_number
        self.validation_data = validation_data
        self.created_at = created_at
        self.changes = changes or []
        self.status = VersionStatus.CURRENT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "validation_id": str(self.validation_id),
            "version_number": self.version_number,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "changes": self.changes
        }


class VersionManager:
    """Gestor de versiones"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._versions: Dict[UUID, List[ValidationVersion]] = {}
        logger.info("VersionManager initialized")
    
    def create_version(
        self,
        validation: PsychologicalValidation,
        changes: Optional[List[str]] = None
    ) -> ValidationVersion:
        """
        Crear nueva versión de validación
        
        Args:
            validation: Validación a versionar
            changes: Lista de cambios (opcional)
            
        Returns:
            Nueva versión creada
        """
        validation_id = validation.id
        
        # Obtener versión actual
        if validation_id in self._versions:
            current_versions = self._versions[validation_id]
            # Marcar versión anterior como deprecada
            if current_versions:
                current_versions[-1].status = VersionStatus.DEPRECATED
            version_number = len(current_versions) + 1
        else:
            self._versions[validation_id] = []
            version_number = 1
        
        # Crear nueva versión
        validation_data = {
            "id": str(validation.id),
            "user_id": str(validation.user_id),
            "status": validation.status.value,
            "connected_platforms": [p.value for p in validation.connected_platforms],
            "profile": validation.profile.to_dict() if validation.profile else None,
            "report": validation.report.to_dict() if validation.report else None,
            "metadata": validation.metadata,
            "created_at": validation.created_at.isoformat(),
            "updated_at": validation.updated_at.isoformat(),
            "completed_at": validation.completed_at.isoformat() if validation.completed_at else None
        }
        
        version = ValidationVersion(
            validation_id=validation_id,
            version_number=version_number,
            validation_data=validation_data,
            created_at=datetime.utcnow(),
            changes=changes
        )
        
        self._versions[validation_id].append(version)
        
        logger.info(
            "Version created",
            validation_id=str(validation_id),
            version_number=version_number
        )
        
        return version
    
    def get_versions(
        self,
        validation_id: UUID,
        include_deprecated: bool = False
    ) -> List[ValidationVersion]:
        """
        Obtener versiones de una validación
        
        Args:
            validation_id: ID de la validación
            include_deprecated: Incluir versiones deprecadas
            
        Returns:
            Lista de versiones
        """
        if validation_id not in self._versions:
            return []
        
        versions = self._versions[validation_id]
        
        if not include_deprecated:
            versions = [v for v in versions if v.status != VersionStatus.DEPRECATED]
        
        return sorted(versions, key=lambda v: v.version_number, reverse=True)
    
    def get_current_version(
        self,
        validation_id: UUID
    ) -> Optional[ValidationVersion]:
        """
        Obtener versión actual de una validación
        
        Args:
            validation_id: ID de la validación
            
        Returns:
            Versión actual o None
        """
        versions = self.get_versions(validation_id, include_deprecated=True)
        current = [v for v in versions if v.status == VersionStatus.CURRENT]
        return current[0] if current else None
    
    def restore_version(
        self,
        validation_id: UUID,
        version_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Restaurar una versión específica
        
        Args:
            validation_id: ID de la validación
            version_number: Número de versión a restaurar
            
        Returns:
            Datos de la validación restaurada o None
        """
        versions = self.get_versions(validation_id, include_deprecated=True)
        target_version = next(
            (v for v in versions if v.version_number == version_number),
            None
        )
        
        if not target_version:
            logger.warning(
                "Version not found",
                validation_id=str(validation_id),
                version_number=version_number
            )
            return None
        
        logger.info(
            "Version restored",
            validation_id=str(validation_id),
            version_number=version_number
        )
        
        return target_version.validation_data
    
    def compare_versions(
        self,
        validation_id: UUID,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """
        Comparar dos versiones
        
        Args:
            validation_id: ID de la validación
            version1: Primera versión
            version2: Segunda versión
            
        Returns:
            Comparación de versiones
        """
        versions = self.get_versions(validation_id, include_deprecated=True)
        
        v1 = next((v for v in versions if v.version_number == version1), None)
        v2 = next((v for v in versions if v.version_number == version2), None)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        comparison = {
            "version1": v1.version_number,
            "version2": v2.version_number,
            "changes": [],
            "differences": {}
        }
        
        # Comparar perfiles
        profile1 = v1.validation_data.get("profile")
        profile2 = v2.validation_data.get("profile")
        
        if profile1 and profile2:
            traits1 = profile1.get("personality_traits", {})
            traits2 = profile2.get("personality_traits", {})
            
            trait_diffs = {}
            for trait in set(traits1.keys()) | set(traits2.keys()):
                val1 = traits1.get(trait, 0.0)
                val2 = traits2.get(trait, 0.0)
                if abs(val1 - val2) > 0.05:
                    trait_diffs[trait] = {
                        "version1": val1,
                        "version2": val2,
                        "change": val2 - val1
                    }
            
            if trait_diffs:
                comparison["differences"]["personality_traits"] = trait_diffs
        
        # Comparar confianza
        conf1 = profile1.get("confidence_score", 0.0) if profile1 else 0.0
        conf2 = profile2.get("confidence_score", 0.0) if profile2 else 0.0
        
        if abs(conf1 - conf2) > 0.05:
            comparison["differences"]["confidence_score"] = {
                "version1": conf1,
                "version2": conf2,
                "change": conf2 - conf1
            }
        
        return comparison
    
    def get_version_history(
        self,
        validation_id: UUID
    ) -> Dict[str, Any]:
        """
        Obtener historial completo de versiones
        
        Args:
            validation_id: ID de la validación
            
        Returns:
            Historial de versiones
        """
        versions = self.get_versions(validation_id, include_deprecated=True)
        
        return {
            "validation_id": str(validation_id),
            "total_versions": len(versions),
            "versions": [v.to_dict() for v in sorted(versions, key=lambda x: x.version_number)],
            "current_version": (
                max(v.version_number for v in versions if v.status == VersionStatus.CURRENT)
                if versions else None
            )
        }


# Instancia global del gestor de versiones
version_manager = VersionManager()




